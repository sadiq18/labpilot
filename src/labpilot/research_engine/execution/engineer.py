"""Research Engineer — deterministic plan orchestrator.

Owns queue, dispatch, evidence, verification, recovery, and resume.
Never calls an LLM to choose the next task.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
                constraints=dict(self.constraints),
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
        except OSError:
            return False
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
