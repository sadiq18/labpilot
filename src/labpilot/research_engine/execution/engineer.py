"""Research Engineer — deterministic plan orchestrator.

Owns queue, dispatch, evidence, verification, recovery, and resume.
Never calls an LLM to choose the next task.
"""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.research_engine.execution.codegen_strategy import (
    resolve_codegen_strategy,
    resolve_codegen_timeout_s,
)
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.evidence import read_evidence, write_evidence
from labpilot.research_engine.execution.recovery import RecoveryAction, decide_recovery
from labpilot.research_engine.execution.registry import CapabilityRegistry
from labpilot.research_engine.execution.schemas import ResearchExecution
from labpilot.research_engine.execution.store import (
    ExecutionStore,
    competition_workspace_path,
)
from labpilot.research_engine.execution.verification import verify_evidence
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskStatus, TaskType
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.planner.validator import topological_levels, validate_plan

logger = logging.getLogger(__name__)


class EngineerError(RuntimeError):
    """Fatal orchestration failure."""


class ResearchEngineer:
    """Walk an approved ResearchPlan via registered capabilities."""

    def __init__(
        self,
        *,
        knowledge_dir: Path,
        competition: str,
        registry: CapabilityRegistry,
        plan_store: PlanStore | None = None,
        execution_store: ExecutionStore | None = None,
        constraints: dict | None = None,
    ) -> None:
        self.knowledge_dir = knowledge_dir
        self.competition = competition
        self.registry = registry
        self.constraints = dict(constraints or {})
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()
        self._plan_store = plan_store or PlanStore(knowledge_dir, competition)
        self._owns_plan_store = plan_store is None
        self._exec_store = execution_store or ExecutionStore(knowledge_dir, competition)
        self._owns_exec_store = execution_store is None

    def close(self) -> None:
        if self._owns_plan_store:
            self._plan_store.close()
        if self._owns_exec_store:
            self._exec_store.close()

    def run_plan(
        self,
        plan_id: str,
        *,
        execution: ResearchExecution | None = None,
    ) -> ResearchExecution:
        """Create an execution (unless provided) and run all pending tasks."""
        plan = self._load_runnable_plan(plan_id)
        if execution is None:
            execution = self._exec_store.create_execution(plan_id)
        elif execution.plan_id != plan_id:
            raise EngineerError(
                f"execution {execution.id} plan_id={execution.plan_id} "
                f"does not match requested plan_id={plan_id}"
            )
        self._plan_store.update_plan_status(plan_id, PlanStatus.IN_PROGRESS)
        self._exec_store.update_status(execution.id, "running")
        if plan.hypothesis_id:
            from labpilot.research_engine.reflection.hypotheses import HypothesisEvaluator

            HypothesisEvaluator(self.knowledge_dir, self.competition).mark_testing(
                plan.hypothesis_id
            )
        return self._run_execution(execution.id)

    def resume(self, execution_id: str) -> ResearchExecution:
        """Continue from the first non-terminal task (retries failed tasks)."""
        execution = self._exec_store.get_execution(execution_id)
        if execution is None:
            raise EngineerError(f"unknown execution_id: {execution_id}")
        if execution.status in {"succeeded", "cancelled"}:
            return execution
        self._exec_store.update_status(execution_id, "running")
        plan = self._plan_store.get_plan(execution.plan_id)
        if plan is not None:
            # Re-queue train/eval spine when a downstream task failed without
            # metrics (false-success training marked done).
            self._reset_tasks_for_retry(plan)
            if plan.status != PlanStatus.IN_PROGRESS:
                self._plan_store.update_plan_status(execution.plan_id, PlanStatus.IN_PROGRESS)
        return self._run_execution(execution_id)

    def _run_execution(self, execution_id: str) -> ResearchExecution:
        execution = self._exec_store.get_execution(execution_id)
        assert execution is not None
        plan = self._plan_store.get_plan(execution.plan_id)
        if plan is None:
            raise EngineerError(f"plan missing for execution {execution_id}")

        try:
            for level in topological_levels(plan):
                for task_id in level:
                    # Reload plan so status updates are visible.
                    plan = self._plan_store.get_plan(execution.plan_id)
                    assert plan is not None
                    task = next(t for t in plan.tasks if t.id == task_id)
                    if task.status == TaskStatus.FAILED:
                        # Retry after a failed run (e.g. fixed train.py).
                        self._plan_store.update_task_status(
                            task.id,
                            TaskStatus.PENDING,
                            metadata_patch={"retried_after_failure": True},
                            error="",
                        )
                    plan = self._plan_store.get_plan(execution.plan_id)
                    assert plan is not None
                    task = next(t for t in plan.tasks if t.id == task_id)
                    if task.status in {TaskStatus.DONE, TaskStatus.SKIPPED}:
                        continue
                    self._run_task(plan, execution_id, task_id)
        except EngineerError as exc:
            self._exec_store.update_status(execution_id, "failed", error=str(exc))
            self._plan_store.update_plan_status(execution.plan_id, PlanStatus.ABANDONED)
            # Tell the hypothesis what happened. Without this it stays `testing`
            # forever — a failed execution writes no evidence card, and a card
            # is the only thing that ever moved it out. Measured on rogii
            # 2026-08-09: one stuck there, three historically, and a redundant
            # one re-selected on every step of four campaigns.
            self._record_hypothesis_attempt(plan, str(exc))
            result = self._exec_store.get_execution(execution_id)
            assert result is not None
            return result

        self._exec_store.update_status(execution_id, "succeeded")
        self._plan_store.update_plan_status(execution.plan_id, PlanStatus.DONE)
        result = self._exec_store.get_execution(execution_id)
        assert result is not None
        try:
            from labpilot.research_engine.execution.outcome import (
                record_successful_execution,
            )

            workspace = Path(
                result.workspace_path
                or competition_workspace_path(self.knowledge_dir, self.competition)
            )
            record_successful_execution(
                knowledge_dir=self.knowledge_dir,
                competition=self.competition,
                execution=result,
                plan=plan,
                workspace_root=workspace,
                llm_client=self.constraints.get("llm_client"),
            )
        except Exception:
            logger.exception("Failed to record execution learning for %s", execution_id)
        return result

    def _run_task(
        self,
        plan: ResearchPlan,
        execution_id: str,
        task_id: str,
    ) -> None:
        plan = self._plan_store.get_plan(plan.id)
        assert plan is not None
        task = next(t for t in plan.tasks if t.id == task_id)
        execution = self._exec_store.get_execution(execution_id)
        assert execution is not None

        capability = self.registry.require(task.type)
        attempt = int(task.metadata.get("attempt", 0))
        workspace = Path(
            execution.workspace_path
            or competition_workspace_path(self.knowledge_dir, self.competition)
        )

        while True:
            self._plan_store.update_task_status(
                task.id,
                TaskStatus.RUNNING,
                metadata_patch={"capability": capability.name, "attempt": attempt},
            )
            prior = read_evidence(self.paths, execution_id, task.id)
            # The funnel every CLI path goes through, so the default is named
            # here once rather than left to each caller to remember. A caller
            # that read the workspace config still wins — this only fills the
            # gap that three separate constructors fell into on PR #118.
            #
            # From the workspace this method already resolved, not from the
            # packaged default: `code_workspace_root()` *is* the slug folder,
            # so `configs/default.yaml` sits right there, and filling the gap
            # by ignoring the config would reproduce the flaw the rest of this
            # change removes. Reported on PR #118.
            constraints = dict(self.constraints)
            constraints.setdefault(
                "codegen_strategy",
                resolve_codegen_strategy(workspace / "configs" / "default.yaml"),
            )
            constraints.setdefault(
                "codegen_timeout_s",
                resolve_codegen_timeout_s(workspace / "configs" / "default.yaml"),
            )
            context = TaskContext(
                plan=plan,
                task=task,
                execution=execution,
                paths=self.paths,
                workspace_root=workspace,
                competition=self.competition,
                prior_evidence=prior,
                runtime_target=execution.runtime_target,
                attempt=attempt,
                constraints=constraints,
            )
            capability.prepare(context)
            evidence = capability.execute(context)
            evidence = capability.verify(context, evidence)
            evidence = capability.collect_evidence(context, evidence)
            write_evidence(self.paths, evidence)

            if verify_evidence(evidence):
                self._plan_store.update_task_status(
                    task.id,
                    TaskStatus.DONE,
                    metadata_patch={
                        "capability": capability.name,
                        "attempt": attempt,
                        "evidence": str(
                            self.paths.executions_dir
                            / execution_id
                            / "evidence"
                            / f"{task.id}.json"
                        ),
                    },
                )
                return

            decision = decide_recovery(task, evidence, attempt=attempt)
            if decision.action == RecoveryAction.RETRY:
                attempt += 1
                logger.info("Retrying task %s (%s)", task.id, decision.reason)
                continue
            if decision.action == RecoveryAction.SKIP:
                self._plan_store.update_task_status(
                    task.id,
                    TaskStatus.SKIPPED,
                    error=decision.reason,
                    metadata_patch={"attempt": attempt},
                )
                return

            self._plan_store.update_task_status(
                task.id,
                TaskStatus.FAILED,
                error=decision.reason,
                metadata_patch={"attempt": attempt},
            )
            raise EngineerError(f"task {task.id} failed: {decision.reason}")

    def _load_runnable_plan(self, plan_id: str) -> ResearchPlan:
        plan = self._plan_store.get_plan(plan_id)
        if plan is None:
            raise EngineerError(f"unknown plan_id: {plan_id}")
        if plan.status == PlanStatus.ABANDONED:
            # Re-run after a failed execution: clear failed tasks and reopen.
            # Also re-queue earlier "done" train/eval spine tasks — a no-op
            # training run can be marked done without writing metrics.json.
            self._reset_tasks_for_retry(plan)
            self._plan_store.update_plan_status(plan_id, PlanStatus.READY)
            plan = self._plan_store.get_plan(plan_id)
            assert plan is not None
            logger.info("Reopened abandoned plan %s for retry", plan_id)
        if plan.status not in {PlanStatus.READY, PlanStatus.IN_PROGRESS}:
            raise EngineerError(f"plan {plan_id} status={plan.status}; need ready or in_progress")
        validate_plan(plan)
        return plan

    #: Tasks whose whole purpose is to prove the generated code runs. When one
    #: of these fails, the code is the thing that is wrong.
    _CODE_VALIDATION_TASKS = frozenset({TaskType.RUN_SMOKE_TEST, TaskType.RUN_UNIT_TEST})

    @staticmethod
    def _training_produced_nothing(plan: ResearchPlan) -> str:
        """A training run that exited 0 and wrote no metrics — a code defect.

        `RUN_TRAINING` is deliberately not a code-validation task: training
        fails for reasons code cannot fix, a missing dataset or an OOM, and
        rebuilding then throws away a file that passed its gates. That rule is
        right and stays.

        This one case is different, because the script *reported success*. It
        did not crash; it simply did not do its job. Measured on rogii
        2026-08-09: training wrote to `./workspace/metrics.json`, a directory it
        invented, exited 0, and nothing read the result. `code_is_suspect`
        stayed false, so `retry_reason` stayed empty and three consecutive
        retries rebuilt blind while the error sat one field away.

        Keyed on the constant the training capability writes, so renaming the
        message breaks the import rather than silently disabling this.
        """
        from labpilot.research_engine.execution.capabilities.training.capability import (
            PRODUCED_NOTHING_MARKERS,
        )

        for task in plan.tasks:
            if task.type is not TaskType.RUN_TRAINING or task.status != TaskStatus.FAILED:
                continue
            error = str(task.metadata.get("error") or "")
            if any(marker in error for marker in PRODUCED_NOTHING_MARKERS):
                return str(task.metadata.get("error") or "")
        return ""

    def _record_hypothesis_attempt(self, plan: ResearchPlan, error: str) -> None:
        """Retire or re-queue the hypothesis behind a failed execution.

        Redundancy is decided here rather than left to the failure text, because
        "aider made no edit" and "the parent already does this" are opposite
        findings that otherwise look identical — one says the adapter failed,
        the other says the campaign chose work already done. Reading them as the
        same thing is what let four campaigns re-select P-021.

        Attempts are counted from the executions that exist, not from a stored
        counter: a counter is a derived value that drifts from its source the
        first time something writes one and not the other.
        """
        hypothesis_id = getattr(plan, "hypothesis_id", None)
        if not hypothesis_id:
            return
        try:
            from labpilot.accessor.common.provenance import classify_failure
            from labpilot.research_engine.reflection.hypotheses import HypothesisEvaluator

            HypothesisEvaluator(self.knowledge_dir, self.competition).record_failed_attempt(
                hypothesis_id,
                failure_reason=error,
                failure_kind=classify_failure(error),
                attempts=self._failed_attempts_for(hypothesis_id),
                # Redundancy is decided upstream, by `AiderAgent`, which holds
                # both the claim and the parent. Deciding it again from the
                # failure text here would be two answers to one question, and
                # the text is the wrong input: "aider made no edit" cannot tell
                # a redundant hypothesis from an adapter that failed.
            )
        except Exception as exc:  # noqa: BLE001 — bookkeeping must not mask the failure
            logger.warning("could not record attempt on %s: %s", hypothesis_id, exc)

    def _failed_attempts_for(self, hypothesis_id: str) -> int:
        """How many executions for this hypothesis have already failed."""
        try:
            plans = self._plan_store.list_plans(hypothesis_id=hypothesis_id)
            failed = sum(
                len(self._exec_store.list_executions(plan_id=p.id, status="failed")) for p in plans
            )
        except Exception:  # noqa: BLE001 — an unreadable store means "first attempt"
            return 1
        # At least 1: this is called *from* a failure, so the current one counts
        # even if the store has not recorded it yet.
        return max(1, failed)

    def _train_script_is_unrunnable(self) -> bool:
        """True when the script we are about to re-run cannot possibly work.

        Asking "which task failed?" is not enough, because the task that
        notices is not always the one scoped as a code check. Measured on rogii
        2026-08-08: codegen returned a 624-byte `train.py` — a docstring and
        half a `# requires-python` line — and **`run_smoke_test` passed it**.
        A docstring followed by comments executes fine and exits 0, so the gate
        whose whole purpose is "does this run" saw success. Only `run_training`
        failed, which is deliberately excluded from the code-suspect set
        because training also fails for reasons code cannot fix.

        So this asks about the artifact instead of the task: if the file on
        disk fails the same checks `apply_proposal` enforces, no retry of it
        can succeed and `write_code` has to run again. Reuses those validators
        rather than restating them — two copies of this rule would be the
        third instance of a guard whose input drifted from its twin.
        """
        from labpilot.research_engine.execution.capabilities.code_engineering.apply import (
            TRAIN_RELPATH,
            ApplyError,
            _check_dependency_block,
            _check_not_truncated,
            strip_stdlib_dependencies,
        )

        root = competition_workspace_path(self.knowledge_dir, self.competition)
        script = Path(root) / TRAIN_RELPATH
        try:
            content = script.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            # A script we cannot read is a script that cannot run, and this
            # method's whole question is "can the thing we are about to re-run
            # work?". Returning False here answered "yes" for a *missing*
            # `train.py`: `write_code` stayed `done`, the retry skipped it, and
            # the plan re-ran a file that was not there — the same
            # rebuild-never-happens loop this method was added to break, just
            # reached by a different door.
            logger.info("Re-queuing write_code: cannot read %s (%s)", TRAIN_RELPATH, exc)
            return True
        try:
            _check_dependency_block(TRAIN_RELPATH, content)
            _check_not_truncated(TRAIN_RELPATH, content)
        except ApplyError as exc:
            logger.info("Re-queuing write_code: %s", exc)
            return True
        # A stdlib name makes uv reject the *whole* dependency set, so a file
        # carrying one cannot run. Regenerating is the way out rather than
        # editing the file here: `apply_proposal` is the only writer, and it
        # strips these on the way in.
        _, stdlib_deps = strip_stdlib_dependencies(content)
        if stdlib_deps:
            logger.info(
                "Re-queuing write_code: %s declares stdlib module(s) as "
                "dependencies (%s), which makes uv reject every dependency",
                TRAIN_RELPATH,
                ", ".join(stdlib_deps),
            )
            return True
        return False

    def _reset_tasks_for_retry(self, plan: ResearchPlan) -> None:
        """Reset failed tasks and the train/eval spine so retries re-produce artifacts.

        `WRITE_CODE` is reset too when a code-validation task failed, and that
        case is why this method exists in its current form. The spine below
        deliberately covers only *run* artifacts, so a retry resumed after
        `write_code` and re-executed the identical file — which cannot pass a
        gate it has already failed. Measured on rogii 2026-08-08: `train.py`
        imported `catboost` with no dependency declaration, the smoke test
        failed, and 16 consecutive `run_experiment` dispatches re-ran that same
        file. Zero `write_code`. The fix for the underlying defect had shipped
        eight days earlier and could not be reached, because the only step that
        would have applied it was marked `done`.

        Scoped to smoke/unit failures on purpose. `run_training` can fail for
        reasons the code cannot fix — a missing dataset, an OOM — and
        regenerating a correct file in response would discard working code and
        spend a codegen call to do it.
        """
        spine = {
            TaskType.RUN_SMOKE_TEST,
            TaskType.RUN_TRAINING,
            TaskType.RUN_INFERENCE,
            TaskType.EVALUATE,
            TaskType.BUILD_SUBMISSION,
            TaskType.GENERATE_REPORT,
            TaskType.REFLECT,
            TaskType.UPDATE_BELIEF,
        }
        failed_ids = {t.id for t in plan.tasks if t.status == TaskStatus.FAILED}
        if not failed_ids:
            return

        code_is_suspect = (
            any(
                t.status == TaskStatus.FAILED and t.type in self._CODE_VALIDATION_TASKS
                for t in plan.tasks
            )
            or self._train_script_is_unrunnable()
            or bool(self._training_produced_nothing(plan))
        )
        if code_is_suspect:
            spine = spine | {TaskType.WRITE_CODE}

        # Transitive dependencies of every failed task (ancestors in the DAG).
        by_id = {t.id: t for t in plan.tasks}
        ancestors: set[str] = set()
        stack = list(failed_ids)
        while stack:
            tid = stack.pop()
            task = by_id.get(tid)
            if task is None:
                continue
            for dep in task.dependencies:
                if dep not in ancestors:
                    ancestors.add(dep)
                    stack.append(dep)

        # Why the code is being rebuilt, so codegen is not asked to try again
        # from the inputs that produced the broken file. Regenerating blind
        # reproduces the same mistake and burns a step doing it; naming the
        # failure is the mechanism that took prose-reply failures from
        # three-in-eight to 30 of 30.
        retry_reason = self._first_failure_reason(plan) if code_is_suspect else ""

        for task in plan.tasks:
            reset = task.id in failed_ids or (
                task.id in ancestors
                and task.type in spine
                and task.status in {TaskStatus.DONE, TaskStatus.FAILED}
            )
            if not reset:
                continue
            patch: dict[str, object] = {"retried_after_abandon": True}
            if retry_reason and task.type == TaskType.WRITE_CODE:
                patch["retry_reason"] = retry_reason
            self._plan_store.update_task_status(
                task.id,
                TaskStatus.PENDING,
                metadata_patch=patch,
                error="",
            )

    @staticmethod
    def _first_failure_reason(plan: ResearchPlan) -> str:
        """The error text from the failed task closest to the code.

        Prefers a code-validation failure over anything downstream: a smoke
        test names what would not run, while `evaluate` failing three steps
        later describes a consequence.
        """
        failed = [
            t
            for t in plan.tasks
            if t.status == TaskStatus.FAILED and str(t.metadata.get("error") or "").strip()
        ]
        if not failed:
            return ""
        code_first = sorted(
            failed,
            key=lambda t: (t.type not in ResearchEngineer._CODE_VALIDATION_TASKS, t.order),
        )
        return str(code_first[0].metadata.get("error") or "").strip()


def default_stub_registry() -> CapabilityRegistry:
    """Registry with a stub covering every TaskType (Plan 2 wiring)."""
    from labpilot.research_engine.execution.capabilities.stub import StubCapability

    registry = CapabilityRegistry()
    registry.register(StubCapability())
    return registry


def default_capability_registry(
    *,
    install_packages: bool = True,
    llm_client=None,
    dry_run: bool = False,
) -> CapabilityRegistry:
    """Full capability registry for the Research Engineer.

    Registers real capabilities for every TaskType. ``dry_run`` is not applied
    here — pass it via :class:`ResearchEngineer` ``constraints``.
    """
    from labpilot.research_engine.execution.capabilities.code_engineering import (
        CodeEngineeringCapability,
    )
    from labpilot.research_engine.execution.capabilities.dependency import (
        DependencyCapability,
    )
    from labpilot.research_engine.execution.capabilities.evaluation import (
        EvaluationCapability,
    )
    from labpilot.research_engine.execution.capabilities.reporting import (
        ReportingCapability,
    )
    from labpilot.research_engine.execution.capabilities.research_review import (
        ResearchReviewCapability,
    )
    from labpilot.research_engine.execution.capabilities.runtime import RuntimeCapability
    from labpilot.research_engine.execution.capabilities.submission import (
        SubmissionCapability,
    )
    from labpilot.research_engine.execution.capabilities.training import (
        TrainingCapability,
    )
    from labpilot.research_engine.execution.capabilities.verification import (
        VerificationCapability,
    )
    from labpilot.research_engine.execution.capabilities.workspace import (
        WorkspaceCapability,
    )

    _ = dry_run  # documented for callers; applied on Engineer.constraints
    registry = CapabilityRegistry()
    registry.register(WorkspaceCapability())
    registry.register(CodeEngineeringCapability(llm_client=llm_client))
    registry.register(ResearchReviewCapability(llm_client=llm_client))
    registry.register(DependencyCapability(install=install_packages))
    registry.register(VerificationCapability())
    registry.register(RuntimeCapability())
    registry.register(TrainingCapability())
    registry.register(EvaluationCapability())
    registry.register(SubmissionCapability())
    registry.register(ReportingCapability())
    return registry
