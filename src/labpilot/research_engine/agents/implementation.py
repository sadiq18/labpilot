"""Implementation specialist — code via CodingTool (EDA/features as code tasks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.agents.events import EventEmitter, noop_emit
from labpilot.research_engine.agents.models import AgentTask, as_agent_task
from labpilot.research_engine.agents.ports import CodingTool
from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.context.models import ContextBundle
from labpilot.research_engine.workspace_facade import Workspace

_INFER_STUB = '''"""Inference entry — keep scoring/prediction separate from training."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def predict() -> None:
    """Load a trained artifact and write submission/predictions (hook for runners)."""
    sub = ROOT / "submission.csv"
    if not sub.is_file():
        sub.write_text("id,prediction\\n0,0\\n", encoding="utf-8")


if __name__ == "__main__":
    predict()
'''

_CODE_SCHEMA = "labpilot.artifact.code/v1"


def _has_existing_code(root: Path) -> bool:
    train = root / "pipeline" / "train.py"
    if train.is_file() and train.stat().st_size > 0:
        return True
    src = root / "src"
    if src.is_dir() and any(src.rglob("*.py")):
        return True
    return False


def _ref_for(path: Path, *, competition: str, task_id: str, index: int) -> ArtifactRef:
    return ArtifactRef(
        kind="code",
        id=f"code:{task_id}:{index}",
        schema_id=_CODE_SCHEMA,
        path=str(path),
        competition=competition,
    )


def ensure_separable_layout(
    workspace: Workspace,
    *,
    task_id: str,
) -> list[ArtifactRef]:
    """Ensure train entrypoint and a separate inference module exist."""
    root = workspace.root
    pipeline = root / "pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    refs: list[ArtifactRef] = []
    train = pipeline / "train.py"
    infer = pipeline / "infer.py"
    idx = 0
    if train.is_file():
        refs.append(_ref_for(train, competition=workspace.competition, task_id=task_id, index=idx))
        idx += 1
    if not infer.is_file():
        infer.write_text(_INFER_STUB, encoding="utf-8")
    refs.append(_ref_for(infer, competition=workspace.competition, task_id=task_id, index=idx))
    return refs


class ImplementationSpecialist:
    """Write/update code through CodingTool; never calls peer specialists."""

    name = "implementation"
    capabilities = [
        "implement",
        "write_code",
        "read_code",
        "modify_config",
        "eda",
        "features",
    ]

    def __init__(
        self,
        coding: CodingTool,
        *,
        on_event: EventEmitter | None = None,
    ) -> None:
        self._coding = coding
        self._emit = on_event or noop_emit

    async def execute(
        self,
        task: object,
        workspace: Workspace,
        context: ContextBundle,
    ) -> list[ArtifactRef]:
        agent_task = as_agent_task(task)
        if agent_task.capability in {"eda", "features"}:
            agent_task = agent_task.model_copy(update={"capability": "implement"})

        existing = _has_existing_code(workspace.root)
        meta: dict[str, Any] = dict(agent_task.metadata)
        meta.setdefault("prefer_separate_inference", True)
        if existing:
            meta.setdefault("prefer_patch", True)

        prefer_patch = bool(meta.get("prefer_patch")) and not bool(meta.get("force_rewrite"))
        refs: list[ArtifactRef] = []

        if existing and prefer_patch and agent_task.capability in {
            "implement",
            "write_code",
            "write",
        }:
            # Preserve existing train/src; only ensure separable inference layout.
            refs = ensure_separable_layout(workspace, task_id=agent_task.id)
        else:
            patched = AgentTask(
                id=agent_task.id,
                capability=agent_task.capability,
                description=agent_task.description
                or (
                    "Update code with separable pipeline/train.py and pipeline/infer.py"
                    if meta.get("prefer_separate_inference")
                    else agent_task.description
                ),
                metadata=meta,
            )
            refs = list(await self._coding.implement(patched, workspace, context))
            layout_refs = ensure_separable_layout(workspace, task_id=agent_task.id)
            seen = {r.path for r in refs}
            for ref in layout_refs:
                if ref.path not in seen:
                    refs.append(ref)

        self._emit(
            "ImplementationFinished",
            {
                "task_id": agent_task.id,
                "competition": workspace.competition,
                "paths": [r.path for r in refs],
                "patched_existing": bool(existing and prefer_patch),
            },
        )
        return refs
