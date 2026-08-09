"""Code Engineering capability — LLM proposes full code; platform applies it.

Primary path: :class:`CodeEngineerAgent` → typed :class:`CodeProposal` →
deterministic apply under allow-list.

Baseline selection records ``baseline_choice.json`` (problem type, metric, and
the validation plan derived from the dataset profile). The LLM is the primary
author; when it yields nothing, the registered Jinja baseline template is
rendered as a fallback because it is real, tested code that honours that
validation plan. The tiny emergency stub is used only when no template matches,
and a non-dry run refuses to continue on it rather than emitting fake metrics
and an invalid submission.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from labpilot.accessor.common.micro_agents import StructuredContext, run_or_none
from labpilot.research_engine.execution.capabilities._helpers import (
    evidence,
    failure_excerpt,
    file_digest,
    is_dry_run,
)
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.capabilities.code_engineering.apply import (
    ALLOWED_ROOTS,
    ApplyError,
    apply_proposal,
)
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.delta import (
    check_delta_consistency,
    record_execution_source,
    snapshot_dir,
)
from labpilot.research_engine.execution.micro_agents.code_engineer import CodeEngineerAgent
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.execution.schemas.code_proposal import (
    CodeFileSpec,
    CodeProposal,
)
from labpilot.research_engine.execution.technique.resolver import (
    TechniqueResolution,
    prompt_technique_fields,
    resolve_technique,
)
from labpilot.research_engine.planner.schemas.task_types import TaskType

logger = logging.getLogger(__name__)

# Last-resort only — never the primary SoR for code generation.
_LAST_RESORT_TRAIN = '''"""Emergency fallback train scaffold (LLM unavailable)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metrics = {"cv_accuracy": 0.0, "status": "last_resort_scaffold"}
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\\n")
    sub = ROOT / "submission.csv"
    if not sub.is_file():
        sub.write_text("id,prediction\\n0,0\\n")
    print("last-resort scaffold complete", metrics)


if __name__ == "__main__":
    main()
'''

_LAST_RESORT_INFER = '''"""Emergency fallback inference module (keep separate from training)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def predict() -> None:
    sub = ROOT / "submission.csv"
    if not sub.is_file():
        sub.write_text("id,prediction\\n0,0\\n", encoding="utf-8")


if __name__ == "__main__":
    predict()
'''


#: Example filenames kept when summarising; enough to convey the naming
#: convention, far short of enumerating a partitioned dataset.
_FILE_SAMPLE = 5


def _summarise_profile(profile: dict) -> dict:
    """Drop the parts of `profile.json` the model cannot use.

    Measured on rogii 2026-08-07: the profile is 47% of the codegen prompt, and
    `files` is 61% of *that* — a flat list of 200 filenames costing ~1,770
    tokens. The generated code globs ``data/raw/<split>/*.csv`` at runtime, so
    it needs the naming *convention*, which five examples convey better than
    two hundred rows. Trimming is lossless for this consumer.

    It matters because a 14,437-token codegen call exceeded the 12,000 TPM
    ceiling of the free tier serving it and ended a campaign.

    Applied unconditionally rather than only when a budget is tight: a prompt
    that changes shape by provider would make two experiments differ in their
    *input*, and that difference would be read as a finding about the technique.
    Same reason `technique_origin` and the served-model stamp exist.
    """
    if not isinstance(profile, dict):
        return {}
    files = profile.get("files")
    if not isinstance(files, list) or len(files) <= _FILE_SAMPLE:
        return profile
    trimmed = dict(profile)
    trimmed["files"] = {
        "count": len(files),
        "sample": [str(f) for f in files[:_FILE_SAMPLE]],
        "note": "full list omitted; the generated code globs the data directory at runtime",
    }
    return trimmed


#: How much of a failure to hand back to codegen. Larger than the smoke gate's
#: own excerpt because the editor is being asked to *fix* the error, not just
#: to report it.
_RETRY_EXCERPT = 2000


def _observe_delta(prior_train: str, proposal: CodeProposal) -> dict[str, object]:
    """Check the change against the claim its own author made. Gates nothing.

    The claim comes from `proposal.kept` / `added` / `combined` — code
    identifiers named by the agent that wrote the file. Not from the plan's
    `technique` field, which cannot serve: `SWA` is not an importable symbol,
    and rogii's plans recorded `feature_engineering` (a category), `add+model`
    (two names concatenated) and once the bare word `the`. Passing those in
    would make a working guard fire on wrong input — the defect this module
    exists to catch, reproduced inside it.

    Self-reported, and that limit is real: a model that lies consistently is not
    caught. The failure that actually happens is carelessness — code that
    contradicts its author's own stated intent — and the gap between the
    declaration and the file is exactly what makes an evidence card wrong.

    Confinement runs regardless, because it needs no claim at all, and it covers
    the case §5 calls **the dangerous one**: a delta that added a technique *and*
    quietly retuned something else, where `technique_attribution` credits the
    whole `cv_gain` to one name.

    **Observe-only on purpose.** These three checks have only ever been
    calibrated against hand-written samples, and that is precisely how the two
    bugs in step 1a got in. The first real campaign supplies a false-positive
    rate; blocking is a one-line change after that, with evidence behind it.
    """
    train = next(
        (f for f in proposal.files if f.path.endswith("pipeline/train.py")),
        None,
    )
    if train is None:
        return {}
    report = check_delta_consistency(
        prior_train,
        train.content,
        keep=list(proposal.kept),
        add=list(proposal.added),
        combine=list(proposal.combined),
    )
    meta = report.as_metadata()
    claimed = bool(proposal.kept or proposal.added or proposal.combined)
    has_parent = bool(prior_train.strip())
    if not claimed:
        # Nothing was claimed, so `consistent: true` would be a pass nobody
        # earned — the fabricated-verdict failure, in the module written to
        # prevent it.
        meta.pop("consistent", None)
        meta.pop("violations", None)
    out: dict[str, object] = {f"delta_{k}": v for k, v in meta.items()}
    out["delta_claim"] = {
        "kept": list(proposal.kept),
        "added": list(proposal.added),
        "combined": list(proposal.combined),
    }
    # A baseline claims nothing because there is nothing to claim. A *delta*
    # that claims nothing was simply never checked — and without this, the two
    # are indistinguishable on the card: both show no verdict and no
    # violations, so an unchecked experiment reads exactly like a clean one.
    #
    # Recorded rather than refused, for the same reason the checks are
    # observe-only: the rate is the thing worth knowing first. If codegen
    # routinely omits the claim, that is a prompt problem to fix with a number
    # attached, not a reason to fail runs today.
    out["delta_claim_declared"] = claimed
    if has_parent and not claimed:
        out["delta_unchecked"] = True
        flags = list(out.get("delta_flags") or [])
        flags.append(
            "the change has a parent but declared no kept/added/combined, so "
            "preservation, addition and combination were not checked — this "
            "card carries no evidence that the delta tested its hypothesis"
        )
        out["delta_flags"] = flags
    return out


class RedundantHypothesisError(RuntimeError):
    """The parent already implements what this hypothesis proposed.

    Raised rather than returned, and deliberately *not* caught by the
    whole-file fallback: every other reason the delta path declines is a codegen
    problem, where falling back is right because the experiment is still worth
    running. This one says the experiment is not worth running at all.

    A failed step is the honest outcome. The campaign records it, the breaker
    counts it, and the hypothesis is already retired, so the next step chooses
    something else — which is what the detection was for.
    """


class CodeEngineeringCapability(BaseCapability):
    name = "code_engineering"

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client
        self._agent = CodeEngineerAgent(llm_client=llm_client)

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset(
            {
                TaskType.READ_CODE,
                TaskType.WRITE_CODE,
                TaskType.MODIFY_CONFIG,
            }
        )

    def execute(self, context: TaskContext) -> TaskEvidence:
        if context.task.type == TaskType.READ_CODE:
            return self._read(context)
        if context.task.type == TaskType.WRITE_CODE:
            return self._write(context)
        return self._modify_config(context)

    def _retire_redundant_hypothesis(self, context: TaskContext, reason: str) -> bool:
        """Retire a hypothesis whose change the parent already implements.

        Detected here rather than after the experiment because the evidence is
        already in hand — the claim and the parent — and because a failed
        execution cannot distinguish this from an adapter that broke. Retiring
        it is the whole point: `_next_hypothesis_id` lists only `proposed`, so
        rejection is what removes it from selection.

        Returns whether the retirement actually landed. The broad catch stays —
        a store problem must not take down the step — but swallowing it silently
        left the hypothesis `proposed` while the caller behaved as though it had
        been retired, so the campaign re-selected the same finished work. The
        caller now says so in the error it raises, which is the difference
        between a bad step and an invisible loop.
        """
        hypothesis_id = getattr(context.plan, "hypothesis_id", None)
        if not hypothesis_id:
            return False
        try:
            from labpilot.research_engine.reflection.hypotheses import HypothesisEvaluator

            # `redundant=True` alone: `classify_hypothesis_failure` short-circuits
            # on it before `failure_kind` is ever read, so passing the kind too
            # implied a second input that does nothing.
            HypothesisEvaluator(context.paths.base_dir, context.competition).record_failed_attempt(
                hypothesis_id,
                failure_reason=reason,
                redundant=True,
            )
        except Exception as exc:  # noqa: BLE001 — bookkeeping must not kill the step
            logger.warning("could not retire redundant hypothesis %s: %s", hypothesis_id, exc)
            return False
        logger.info("Retired %s: %s", hypothesis_id, reason)
        return True

    def _propose_delta(
        self,
        context: TaskContext,
        structured,
        prior_train: str,
    ) -> tuple[CodeProposal | None, str]:
        """Try the aider path. ``(None, "")`` means "use the whole-file path".

        Four ways this declines, and each is a routing decision rather than a
        failure:

        * not configured — `codegen.strategy` defaults to `whole_file` (§10:
          both paths coexist while the rate is measured);
        * no parent — a baseline has nothing to diff against, which is
          `WholeFileAgent`'s job by design;
        * no gateway — aider without the proxy bypasses the budget ledger, rate
          limiting and failover, which §4 calls a regression dressed as a
          feature. Declining beats routing around M10;
        * aider failed — recorded with its kind, then handed on.
        """
        if str(context.constraints.get("codegen_strategy") or "whole_file") != "delta":
            return None, ""
        if not prior_train.strip():
            return None, ""
        gateway = self._llm if callable(getattr(self._llm, "for_role", None)) else None
        if gateway is None:
            logger.info("codegen.strategy=delta ignored: no gateway, so aider would bypass M10")
            return None, ""

        from labpilot.research_engine.execution.delta.aider_agent import (
            AiderAgent,
            AiderError,
        )

        try:
            agent = AiderAgent(gateway)
            proposal = agent.propose(structured, Path(context.workspace_root))
        except AiderError as exc:
            # Already recorded to `agent_invocations` with its kind by the
            # agent itself; this only decides what happens next.
            if exc.kind == "hypothesis_redundant":
                # Not a codegen failure, and so not a fallback. The campaign
                # chose work already done: retiring the hypothesis stops it
                # being chosen again — four campaigns re-selected P-021 because
                # nothing did this — and raising stops *this* step from doing
                # it anyway.
                #
                # Falling through to whole-file was the gap: the hypothesis was
                # marked already-implemented and then the LLM rewrote train.py
                # from scratch and the runner trained it, spending the full
                # experiment the retirement existed to avoid. Worse, a
                # successful run reaches `HypothesisEvaluator.evaluate`, which
                # writes the critic's verdict with no settled-status guard — so
                # a hypothesis just retired as redundant could come back
                # `confirmed`, carrying evidence from a run that tested
                # something else entirely.
                retired = self._retire_redundant_hypothesis(context, str(exc))
                raise RedundantHypothesisError(
                    f"{exc}"
                    + (
                        ""
                        if retired
                        else " (and the hypothesis could not be retired, so it "
                        "may be selected again — see the warning above)"
                    )
                ) from exc
            logger.warning("aider proposal failed (%s); falling back: %s", exc.kind, exc)
            return None, ""
        except Exception:  # noqa: BLE001 — a codegen path must not kill the step
            logger.exception("aider proposal raised; falling back to whole-file")
            return None, ""
        return (proposal, "aider") if proposal.files else (None, "")

    def _read(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        notes: list[str] = []
        paths: list[str] = []
        for rel in ("src", "pipeline", "configs"):
            d = root / rel
            if d.is_dir():
                for path in sorted(d.rglob("*")):
                    if path.is_file() and len(paths) < 20:
                        paths.append(str(path))
                        notes.append(str(path.relative_to(root)))
        notes_path = root / "artifacts" / "code_notes.json"
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(
            json.dumps({"files": notes, "competition": context.competition}, indent=2) + "\n",
            encoding="utf-8",
        )
        return evidence(
            context,
            capability=self.name,
            passed=True,
            summary=f"inspected {len(paths)} files",
            checks=["read_code"],
            paths=paths + [str(notes_path)],
            metadata={"file_count": len(paths)},
        )

    def _write(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        (root / "pipeline").mkdir(parents=True, exist_ok=True)
        train_path = root / "pipeline" / "train.py"

        prior_train = ""
        backup_path: Path | None = None
        if train_path.is_file():
            try:
                prior_train = train_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                prior_train = ""
            backup_dir = root / "artifacts" / "code_backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"train_{context.execution.id}.py"
            try:
                backup_path.write_text(prior_train, encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not backup train.py: %s", exc)
                backup_path = None

        profile_path = root / "profile.json"
        if not profile_path.is_file() and not is_dry_run(context):
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary="code write blocked: missing dataset profile",
                checks=["write_code", "profile_required"],
                error=(
                    "profile.json missing under the competition workspace. "
                    "prepare_workspace must download and profile data before write_code."
                ),
            )

        choice = self._select_baseline(context, root)
        problem_type = (
            choice.problem_type
            if choice is not None
            else str(context.constraints.get("problem_type") or "unknown")
        )
        profile_summary = self._profile_summary(root)
        data_inventory = self._data_inventory(root)

        brief = ""
        brief_path = context.paths.brief_path
        if brief_path.is_file():
            brief = brief_path.read_text(encoding="utf-8")[:3000]

        hyp_fields = self._hypothesis_fields(context)
        plan_meta = dict(context.plan.metadata or {})
        # Resolved *before* the prompt is built, not after. A rejected label is
        # worse than no label — rogii asked codegen to implement "hyp:H-010",
        # which no model can do, and the triad below already carries the real
        # intent. Resolution is also needed by the fallback and by provenance.
        resolution = resolve_technique(
            plan_meta, hyp_fields, choice=choice, profile=profile_summary
        )
        if resolution.status != "none":
            logger.info("technique %s: %s", resolution.status, resolution.reason)
        self._stamp_technique(root, resolution)
        technique_fields = prompt_technique_fields(resolution, plan_meta, hyp_fields)
        structured = StructuredContext(
            competition=context.competition,
            question=context.plan.goal or context.task.description,
            text=brief,
            data={
                "task_id": context.task.id,
                "task_type": str(context.task.type),
                "task_description": context.task.description,
                "plan_id": context.plan.id,
                "plan_goal": context.plan.goal,
                "plan_kind": plan_meta.get("plan_kind", ""),
                "hypothesis_id": context.plan.hypothesis_id,
                "problem_type": problem_type,
                "baseline_choice": choice.model_dump(mode="json") if choice else {},
                "profile_summary": profile_summary,
                "data_inventory": data_inventory,
                "allowed_roots": list(ALLOWED_ROOTS),
                "existing_files": self._inventory(root),
                "brief_excerpt": brief,
                "dry_run": is_dry_run(context),
                "workspace_root": str(root),
                "skill_agent_key": "code_engineer",
                "prior_train_py": prior_train[:120_000],
                # Why the last attempt failed, set by `_reset_tasks_for_retry`
                # when it re-queues this task. Without it the model is asked to
                # try again from the inputs that produced the broken file, and
                # reproduces the same mistake — measured on rogii 2026-08-08,
                # where a `Geology: object` column was handed to LightGBM.
                # Shared with the smoke gate, which learned the same lesson
                # first: keep the *tail*, and collapse tqdm's `\r` frames so a
                # progress bar cannot fill the budget. Two implementations of
                # one idiom is how the `\r` fix lands in only one of them.
                "retry_reason": failure_excerpt(
                    str(context.task.metadata.get("retry_reason") or ""),
                    "",
                    limit=_RETRY_EXCERPT,
                ),
                "parent_hypothesis_id": plan_meta.get("parent_hypothesis_id"),
                "parent_metrics": plan_meta.get("parent_metrics") or {},
                **technique_fields,
                "observation": hyp_fields.get("observation", ""),
                "reason": hyp_fields.get("reason", ""),
                "prediction": hyp_fields.get("prediction", ""),
                "evidence": hyp_fields.get("evidence") or [],
                "improve_on_prior": bool(
                    prior_train
                    or plan_meta.get("parent_hypothesis_id")
                    or context.plan.hypothesis_id
                ),
            },
        )
        # Delta first when configured and a parent exists, falling back to the
        # whole-file path rather than failing the step. §10 requires both paths
        # to coexist while the failure rate is measured, and a campaign that
        # cannot produce code because aider had a bad day measures nothing.
        proposal, origin = self._propose_delta(context, structured, prior_train)
        if proposal is None:
            raw = run_or_none(self._agent, structured)
            proposal = raw if isinstance(raw, CodeProposal) else CodeProposal()
            origin = "llm" if self._agent.last_used_llm else "last_resort"

        if not proposal.files:
            # Prefer the deterministic baseline template over the emergency
            # stub. The stub writes fake metrics and a wrong-header submission
            # while reporting success, so a failed LLM silently produced a
            # garbage leaderboard entry. A rendered template is real, tested
            # code that honours the derived validation plan.
            rendered = self._render_template_fallback(root, choice, resolution)
            if rendered is not None:
                proposal = rendered
                origin = "template"
            else:
                proposal = CodeProposal(
                    summary="last-resort scaffold",
                    rationale="LLM produced no files and no template matched",
                    files=[
                        CodeFileSpec(
                            path="pipeline/train.py",
                            content=_LAST_RESORT_TRAIN,
                            action="write",
                        ),
                        CodeFileSpec(
                            path="pipeline/infer.py",
                            content=_LAST_RESORT_INFER,
                            action="write",
                        ),
                    ],
                )
                origin = "last_resort"

        # A dry run only checks wiring, so the stub is fine there. A real run
        # must not continue on it: the stub writes fake metrics and a
        # wrong-header submission, which evaluate/submit would then dress up as
        # a genuine leaderboard result.
        if origin == "last_resort" and not is_dry_run(context):
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary="no usable training code (LLM produced none, no template matched)",
                checks=["write_code"],
                error=(
                    "Code generation produced no files and no baseline template "
                    "matched this problem type. Check `research doctor` (LLM "
                    "provider) or add a template for this problem type."
                ),
                metadata={"origin": origin, "problem_type": problem_type},
            )

        try:
            written = apply_proposal(root, proposal)
        except ApplyError as exc:
            logger.warning("Code proposal apply failed: %s", exc)
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary="code proposal rejected",
                checks=["write_code", "apply"],
                error=str(exc),
                metadata={"origin": origin, "problem_type": problem_type},
            )

        digests = {str(p): file_digest(p) for p in written}
        paths = [str(p) for p in written]
        delta = _observe_delta(prior_train, proposal)
        # M19 §6: record what *this* execution ran, keyed by this execution, so
        # a child can address its parent's code instead of inferring it from
        # write order. Snapshotted after apply, so it records what landed.
        snapshot = record_execution_source(root, str(context.execution.id), written)
        if backup_path is not None:
            paths.append(str(backup_path))
        return evidence(
            context,
            capability=self.name,
            passed=train_path.is_file(),
            summary=f"code written via {origin} ({len(written)} files; overridden)",
            checks=["write_code", "apply", origin, "override"],
            paths=paths,
            metadata={
                "digests": digests,
                "origin": origin,
                "problem_type": problem_type,
                "summary": proposal.summary,
                "rationale": proposal.rationale,
                "used_llm": self._agent.last_used_llm,
                "dry_run": is_dry_run(context),
                "used_jinja": False,
                "overrode_existing": bool(prior_train),
                "backup": str(backup_path) if backup_path else None,
                "code_snapshot": str(snapshot_dir(root, str(context.execution.id)))
                if snapshot
                else None,
                # F5. Reflection must be able to tell three outcomes apart that
                # today all read as "technique X did not help": never applied,
                # applied with no effect, and applied-and-worse. Only the last
                # is evidence about the technique.
                "technique": resolution.requested or None,
                "technique_canonical": resolution.canonical,
                "technique_status": resolution.status,
                "technique_reason": resolution.reason,
                **delta,
                "technique_origin": (
                    "registry"
                    if resolution.changes_rendering and origin == "template"
                    else "llm"
                    if origin == "llm"
                    else "none"
                ),
            },
            error=None if train_path.is_file() else "train.py missing after apply",
        )

    def _modify_config(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        config_dir = root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "baseline.yaml"
        existed = config_path.is_file()
        if not existed:
            config_path.write_text(
                f"competition: {context.competition}\n"
                f"plan_id: {context.plan.id}\n"
                "epochs: 1\n"
                "smoke: true\n",
                encoding="utf-8",
            )
        pipeline_cfg = root / "pipeline" / "config.yaml"
        if pipeline_cfg.parent.is_dir() and not pipeline_cfg.is_file():
            pipeline_cfg.write_text(
                f"competition: {context.competition}\nmetric: accuracy\n",
                encoding="utf-8",
            )
        return evidence(
            context,
            capability=self.name,
            passed=config_path.is_file(),
            summary="config written" if not existed else "config already present",
            checks=["modify_config"],
            paths=[str(config_path)],
            metadata={"digest": file_digest(config_path), "idempotent": existed},
        )

    def _hypothesis_fields(self, context: TaskContext) -> dict:
        hyp_id = context.plan.hypothesis_id
        if not hyp_id:
            return {}
        try:
            from labpilot.research_engine.shared.experiments.hypothesis import (
                HypothesisStore,
            )

            hyp = HypothesisStore(context.paths.base_dir, context.competition).get(hyp_id)
        except Exception:
            return {}
        if hyp is None:
            return {}
        return {
            "observation": hyp.observation,
            "reason": hyp.reason,
            "prediction": hyp.prediction,
            "technique": hyp.technique,
            "technique_stack": list(hyp.technique_stack),
            "combo_techniques": list(hyp.combo_techniques),
            "evidence": [e.model_dump(mode="json") for e in hyp.evidence],
        }

    def _render_template_fallback(
        self,
        root: Path,
        choice: object | None,
        resolution: TechniqueResolution | None = None,
    ) -> CodeProposal | None:
        """Render the registered baseline template as a CodeProposal, or None.

        Used when LLM codegen yields nothing. The template already encodes the
        validation plan derived from the dataset profile, so this fallback is a
        real baseline rather than a placeholder.

        ``resolution`` carries the technique the plan is testing. Without it
        this path renders the *same* baseline for every hypothesis, which is
        what made twelve distinct hypotheses score identically: the run looked
        healthy and tested nothing.
        """
        resolution = resolution or TechniqueResolution()
        if choice is None:
            return None
        try:
            from labpilot.config import load_config
            from labpilot.research_engine.execution.baseline.registry import get_template
            from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen.renderer import (  # noqa: E501
                CodeRenderer,
            )

            template = get_template(choice.problem_type, template_name=choice.template_name)
            if template is None:
                return None
            config = load_config()
            # The renderer has always accepted these; every one of them was
            # dropped here, so the fallback rendered the same baseline whatever
            # the hypothesis asked for — 12 hypotheses, MSE 194.80 identically.
            CodeRenderer(config.training).render(
                template,
                choice,
                root,
                feature_recipes=resolution.feature_recipes or None,
                model_params=resolution.model_params or None,
            )
        except Exception as exc:  # noqa: BLE001 — fall through to the stub
            logger.warning("Template fallback render failed: %s", exc)
            return None

        produced = [p for p in (root / "pipeline").glob("*") if p.is_file()]
        if not produced:
            return None
        # Files are already on disk; describe them so the apply step records
        # digests and the run stays auditable.
        return CodeProposal(
            summary=f"rendered baseline template {template.name}",
            rationale="LLM produced no files; used the deterministic template",
            files=[
                CodeFileSpec(
                    path=f"pipeline/{p.name}",
                    content=p.read_text(encoding="utf-8"),
                    action="write",
                )
                for p in produced
            ],
        )

    def _inventory(self, root: Path) -> list[str]:
        items: list[str] = []
        for rel in ALLOWED_ROOTS:
            d = root / rel
            if not d.is_dir():
                continue
            for path in sorted(d.rglob("*")):
                if path.is_file() and len(items) < 30:
                    items.append(str(path.relative_to(root)))
        return items

    def _profile_summary(self, root: Path) -> dict:
        path = root / "profile.json"
        if not path.is_file():
            return {}
        try:
            return _summarise_profile(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return {}

    def _data_inventory(self, root: Path) -> list[str]:
        raw = root / "data" / "raw"
        if not raw.is_dir():
            return []
        items: list[str] = []
        for path in sorted(raw.rglob("*")):
            if path.is_file() or (path.is_dir() and path.name.endswith(".zarr")):
                items.append(str(path.relative_to(root)))
            if len(items) >= 80:
                break
        return items

    def _select_baseline(self, context: TaskContext, root: Path) -> object | None:
        """Record baseline_choice.json for problem-type / metric hints (no Jinja)."""
        try:
            from labpilot.accessor.profiler.tabular import DatasetProfile
            from labpilot.research_engine.execution.baseline.selector import (
                BaselineSelector,
            )
            from labpilot.research_engine.intelligence.competition.models import (
                CompetitionSpec,
            )
        except Exception:
            return None

        competition = CompetitionSpec(slug=context.competition)
        comp_path = root / "competition.json"
        if comp_path.is_file():
            try:
                competition = CompetitionSpec.model_validate_json(
                    comp_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        profile = DatasetProfile(competition=context.competition)
        profile_path = root / "profile.json"
        if profile_path.is_file():
            try:
                profile = DatasetProfile.model_validate_json(
                    profile_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        try:
            choice = BaselineSelector().select(competition, profile)
        except Exception as exc:
            logger.info("Baseline selection deferred to LLM: %s", exc)
            return None

        try:
            (root / "baseline_choice.json").write_text(
                choice.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass
        return choice

    def _stamp_technique(self, root: Path, resolution: TechniqueResolution) -> None:
        """Record the resolution on ``baseline_choice.json`` (F5).

        Written after the fact rather than by the selector, which derives from
        **data only** and never sees a plan — a property worth keeping
        (design §8.5). This is the artifact an operator reads to answer "what
        did this run actually apply?" without replaying the log.
        """
        path = root / "baseline_choice.json"
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["applied_technique"] = {
                "requested": resolution.requested,
                "canonical": resolution.canonical,
                "status": resolution.status,
                "reason": resolution.reason,
                "feature_recipes": list(resolution.feature_recipes),
                "model_params": dict(resolution.model_params),
            }
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — provenance must not fail a run
            logger.warning("Could not stamp applied_technique: %s", exc)
