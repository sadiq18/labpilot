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

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.execution.capabilities._helpers import (
    evidence,
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
            json.dumps({"files": notes, "competition": context.competition}, indent=2)
            + "\n",
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
        proposal = self._agent.run(structured)
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
                # F5. Reflection must be able to tell three outcomes apart that
                # today all read as "technique X did not help": never applied,
                # applied with no effect, and applied-and-worse. Only the last
                # is evidence about the technique.
                "technique": resolution.requested or None,
                "technique_canonical": resolution.canonical,
                "technique_status": resolution.status,
                "technique_reason": resolution.reason,
                "technique_origin": (
                    "registry" if resolution.changes_rendering and origin == "template"
                    else "llm" if origin == "llm"
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

            hyp = HypothesisStore(context.paths.base_dir, context.competition).get(
                hyp_id
            )
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

            template = get_template(
                choice.problem_type, template_name=choice.template_name
            )
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
            return json.loads(path.read_text(encoding="utf-8"))
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
