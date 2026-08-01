"""Unit tests for Research OS specialist agent runtime."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from labpilot.research_engine.agents import (
    AgentTask,
    SpecialistDescriptor,
    SpecialistRegistry,
    V1CodeEngineeringCodingTool,
    execute_agent_sync,
)
from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.context.models import ContextBundle, ContextRequest
from labpilot.research_engine.workspace_facade import Workspace

_FORBIDDEN_IMPORT_PREFIXES = (
    "labpilot.research_engine.conductor",
)


class _EchoAgent:
    """Test double — not a peer specialist module."""

    name = "echo"
    capabilities = ["echo", "test"]

    async def execute(
        self,
        task: object,
        workspace: Workspace,
        context: ContextBundle,
    ) -> list[ArtifactRef]:
        note = ""
        if isinstance(task, AgentTask):
            note = task.description
        elif isinstance(task, dict):
            note = str(task.get("description", ""))
        path = workspace.root / "echo.txt"
        path.write_text(note or context.request.goal, encoding="utf-8")
        return [
            ArtifactRef(
                kind="echo",
                id=f"echo:{workspace.competition}",
                schema_id="labpilot.artifact.echo/v1",
                path=str(path),
                competition=workspace.competition,
            )
        ]


def _empty_bundle(competition: str = "demo") -> ContextBundle:
    return ContextBundle(request=ContextRequest(competition=competition, goal="test"))


def test_registry_register_lookup_and_candidates() -> None:
    registry = SpecialistRegistry()
    cheap = SpecialistDescriptor(
        name="echo-cheap",
        capabilities=["echo"],
        cost_hint=1.0,
        agent=_EchoAgent(),
    )
    pricey = SpecialistDescriptor(
        name="echo-pricey",
        capabilities=["echo"],
        cost_hint=10.0,
        agent=_EchoAgent(),
    )
    other = SpecialistDescriptor(
        name="other",
        capabilities=["other"],
        cost_hint=0.5,
        agent=_EchoAgent(),
    )
    registry.register(cheap)
    registry.register(pricey)
    registry.register(other)

    assert registry.get("echo-cheap") is not None
    assert registry.get("missing") is None
    assert registry.require("echo-pricey").name == "echo-pricey"

    all_echo = registry.candidates(capability="echo")
    assert [s.name for s in all_echo] == ["echo-cheap", "echo-pricey"]

    budgeted = registry.candidates(capability="echo", budget=5.0)
    assert [s.name for s in budgeted] == ["echo-cheap"]


def test_execute_agent_sync_no_event_loop(tmp_path: Path) -> None:
    ws = Workspace.from_competition(
        tmp_path / "knowledge", "demo", code_root=tmp_path / "ws"
    )
    ws.ensure_roots()
    bundle = _empty_bundle("demo")
    refs = execute_agent_sync(
        _EchoAgent(),
        AgentTask(id="T-1", capability="echo", description="hello"),
        ws,
        bundle,
    )
    assert len(refs) == 1
    assert Path(refs[0].path or "").read_text(encoding="utf-8") == "hello"


def test_coding_tool_v1_read_smoke(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    ws = Workspace.from_competition(knowledge, "demo", code_root=tmp_path / "ws")
    ws.ensure_roots()
    (ws.root / "src").mkdir(parents=True)
    (ws.root / "src" / "hello.py").write_text("x = 1\n", encoding="utf-8")

    tool = V1CodeEngineeringCodingTool(llm_client=None)
    bundle = _empty_bundle("demo")
    refs = execute_agent_sync(
        _CodingAgent(tool),
        AgentTask(id="T-read", capability="read_code", description="inspect"),
        ws,
        bundle,
    )
    assert refs
    assert any(r.kind == "code" for r in refs)
    notes = ws.root / "artifacts" / "code_notes.json"
    assert notes.is_file()


def test_coding_tool_v1_write_smoke(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    ws = Workspace.from_competition(knowledge, "demo", code_root=tmp_path / "ws")
    ws.ensure_roots()
    (ws.root / "profile.json").write_text(
        json.dumps(
            {
                "competition": "demo",
                "files": ["train.csv"],
                "train_file": "train.csv",
                "test_file": "test.csv",
                "sample_submission_file": "sample_submission.csv",
                "row_count": 100,
                "column_count": 3,
                "columns": [
                    {"name": "id", "dtype": "int"},
                    {"name": "a", "dtype": "float"},
                    {"name": "target", "dtype": "int"},
                ],
            }
        ),
        encoding="utf-8",
    )

    tool = V1CodeEngineeringCodingTool(llm_client=None)
    bundle = ContextBundle(
        request=ContextRequest(competition="demo", goal="baseline"),
        items=[],
    )
    refs = execute_agent_sync(
        _CodingAgent(tool),
        AgentTask(id="T-write", capability="implement", description="write baseline"),
        ws,
        bundle,
    )
    assert refs
    assert (ws.root / "pipeline" / "train.py").is_file()


class _CodingAgent:
    """Thin Agent that delegates to CodingTool (smoke double)."""

    name = "coding-smoke"
    capabilities = ["implement", "read_code"]

    def __init__(self, coding: V1CodeEngineeringCodingTool) -> None:
        self._coding = coding

    async def execute(
        self,
        task: object,
        workspace: Workspace,
        context: ContextBundle,
    ) -> list[ArtifactRef]:
        return await self._coding.implement(task, workspace, context)


def test_agents_package_importable() -> None:
    import labpilot.research_engine.agents as agents

    assert agents.SpecialistRegistry is SpecialistRegistry
    assert agents.V1CodeEngineeringCodingTool is V1CodeEngineeringCodingTool


def test_agents_do_not_import_conductor() -> None:
    """Specialists must not reach into Conductor for peer control flow."""
    agents_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "labpilot"
        / "research_engine"
        / "agents"
    )
    for path in sorted(agents_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                        assert not alias.name.startswith(prefix), (
                            f"{path.name} imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                    assert not node.module.startswith(prefix), (
                        f"{path.name} imports from {node.module}"
                    )


def test_context_bundle_required_type(tmp_path: Path) -> None:
    """execute_agent_sync accepts ContextBundle (M4 handoff)."""
    ws = Workspace.from_competition(
        tmp_path / "knowledge", "demo", code_root=tmp_path / "ws"
    )
    ws.ensure_roots()
    bundle = _empty_bundle()
    assert isinstance(bundle, ContextBundle)
    refs = execute_agent_sync(
        _EchoAgent(),
        AgentTask(id="T-2", capability="echo"),
        ws,
        bundle,
    )
    assert refs[0].competition == "demo"
