"""Code Engineering capability — LLM proposes full code; platform applies it.

Primary path: :class:`CodeEngineerAgent` → typed :class:`CodeProposal` →
deterministic apply under allow-list.

Offline / CI: Jinja baseline templates as ``rule_engine`` (full scaffold code).
A tiny stub is only the last resort when Jinja cannot render.
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
from labpilot.research_engine.planner.schemas.task_types import TaskType

logger = logging.getLogger(__name__)

# Last-resort only — never the primary SoR for code generation.
_LAST_RESORT_TRAIN = '''"""Emergency fallback train scaffold (Jinja/LLM unavailable)."""
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

        if train_path.is_file() and not context.task.metadata.get("force_rewrite"):
            return evidence(
                context,
                capability=self.name,
                passed=True,
                summary="train.py already present",
                checks=["write_code", "idempotent"],
                paths=[str(train_path)],
                metadata={"idempotent": True, "digest": file_digest(train_path)},
            )

        jinja_baseline = self._render_jinja_baseline(context, root)
        brief = ""
        brief_path = context.paths.brief_path
        if brief_path.is_file():
            brief = brief_path.read_text(encoding="utf-8")[:3000]

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
                "plan_kind": context.plan.metadata.get("plan_kind", ""),
                "hypothesis_id": context.plan.hypothesis_id,
                "problem_type": context.constraints.get(
                    "problem_type", "tabular_classification"
                ),
                "allowed_roots": list(ALLOWED_ROOTS),
                "existing_files": self._inventory(root),
                "jinja_baseline": jinja_baseline,
                "brief_excerpt": brief,
                "dry_run": is_dry_run(context),
            },
        )
        proposal = self._agent.run(structured)
        origin = "llm" if self._agent.last_used_llm else "rule_engine"

        if not proposal.files:
            # Last resort only.
            proposal = CodeProposal(
                summary="last-resort scaffold",
                rationale="Jinja and LLM produced no files",
                files=[
                    CodeFileSpec(
                        path="pipeline/train.py",
                        content=_LAST_RESORT_TRAIN,
                        action="write",
                    )
                ],
            )
            origin = "last_resort"

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
                metadata={"origin": origin},
            )

        digests = {str(p): file_digest(p) for p in written}
        return evidence(
            context,
            capability=self.name,
            passed=train_path.is_file(),
            summary=f"code written via {origin} ({len(written)} files)",
            checks=["write_code", "apply", origin],
            paths=[str(p) for p in written],
            metadata={
                "digests": digests,
                "origin": origin,
                "summary": proposal.summary,
                "rationale": proposal.rationale,
                "used_llm": self._agent.last_used_llm,
                "dry_run": is_dry_run(context),
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

    def _render_jinja_baseline(
        self, context: TaskContext, root: Path
    ) -> dict[str, str]:
        """Render full Jinja templates into memory (path → content)."""
        try:
            from labpilot.config import TrainingConfig
            from labpilot.accessor.profiler.tabular import DatasetProfile
            from labpilot.research_engine.execution.baseline.registry import get_template
            from labpilot.research_engine.execution.baseline.selector import (
                BaselineChoice,
                BaselineSelector,
            )
            from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen.renderer import CodeRenderer
            from labpilot.research_engine.intelligence.competition.models import (
                CompetitionSpec,
            )
        except Exception:
            return {}

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
        except Exception:
            problem_type = str(
                context.constraints.get("problem_type", "tabular_classification")
            )
            if problem_type not in {
                "tabular_classification",
                "tabular_regression",
                "text_classification",
                "image_classification",
            }:
                problem_type = "tabular_classification"
            template = get_template(problem_type)
            if template is None:
                return {}
            choice = BaselineChoice(
                problem_type=problem_type,
                template_name=template.name,
                rationale="Research Engineer baseline scaffold",
                metric_name="accuracy" if "regression" not in problem_type else "rmse",
            )

        template = get_template(choice.problem_type, template_name=choice.template_name)
        if template is None:
            return {}

        # Persist choice for inspectability (idempotent overwrite).
        try:
            (root / "baseline_choice.json").write_text(
                choice.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

        # Render into a temp sibling then read back — CodeRenderer writes to disk.
        scratch = root / ".codegen_scratch"
        try:
            if scratch.exists():
                import shutil

                shutil.rmtree(scratch)
            scratch.mkdir(parents=True, exist_ok=True)
            CodeRenderer(TrainingConfig()).render(template, choice, scratch)
            files: dict[str, str] = {}
            pipeline = scratch / "pipeline"
            if pipeline.is_dir():
                for path in pipeline.rglob("*"):
                    if path.is_file():
                        rel = f"pipeline/{path.relative_to(pipeline).as_posix()}"
                        files[rel] = path.read_text(encoding="utf-8")
            return files
        except Exception as exc:
            logger.info("Jinja baseline render failed: %s", exc)
            return {}
        finally:
            import shutil

            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)
