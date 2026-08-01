"""Unit tests for Research OS ToolRegistry (M1 plan-2a)."""

from __future__ import annotations

import ast
from pathlib import Path

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.tools import (
    ToolDescriptor,
    ToolRegistry,
    ToolResult,
    build_default_tool_registry,
)
from labpilot.research_engine.tools.handlers.submit import submit
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace

_EXPECTED_TOOLS = {
    "analyze_competition",
    "search_papers",
    "generate_plan",
    "implement",
    "run_plan",
    "run_experiment",
    "reflect",
    "submit",
    "submit_learn",
    "query_memory",
}

_FORBIDDEN_FROM_ENGINE = (
    "labpilot.research_engine.artifacts",
    "labpilot.research_engine.tools",
    "labpilot.research_engine.conductor",
)


def test_default_registry_lists_catalog_tools() -> None:
    registry = build_default_tool_registry()
    names = set(registry.names())
    assert _EXPECTED_TOOLS <= names
    for name in _EXPECTED_TOOLS:
        tool = registry.require(name)
        assert tool.handler is not None
        assert "competition" in tool.required_workspace_fields


def test_registry_lookup_and_fake_tool_invoke(tmp_path: Path) -> None:
    registry = ToolRegistry()

    def echo_tool(workspace: Workspace, *, note: str = "") -> ToolResult:
        ref = ArtifactRef(
            kind="echo",
            id=f"echo:{workspace.competition}",
            schema_id="labpilot.artifact.echo/v1",
            path=str(workspace.root / "echo.txt"),
            competition=workspace.competition,
        )
        (workspace.root / "echo.txt").write_text(note, encoding="utf-8")
        return ToolResult(refs=[ref], data={"note": note})

    registry.register(
        ToolDescriptor(
            name="echo",
            description="test double",
            output_artifacts=["echo"],
            handler=echo_tool,
        )
    )
    assert registry.get("echo") is not None
    assert registry.get("missing") is None

    ws = Workspace.from_competition(tmp_path / "knowledge", "demo", code_root=tmp_path / "ws")
    ws.ensure_roots()
    result = registry.invoke("echo", ws, note="hello")
    assert len(result.refs) == 1
    assert result.refs[0].kind == "echo"
    assert result.data["note"] == "hello"
    assert Path(result.refs[0].path or "").read_text(encoding="utf-8") == "hello"


def test_submit_handler_accepts_workspace(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "pack-comp", "pack-comp")
    ws = Workspace.from_client(client)
    ws.ensure_roots()
    (ws.root / "submission.csv").write_text("id,prediction\n1,0.5\n", encoding="utf-8")

    result = submit(ws, execution_id="E-001")
    assert result.refs[0].kind == "submission"
    assert Path(result.data["csv_path"]).is_file()
    assert "E-001" in result.data["csv_path"]

    # Same path via registry.invoke (Workspace-aware).
    registry = build_default_tool_registry()
    via_registry = registry.invoke("submit", ws, execution_id="E-001")
    assert via_registry.refs[0].kind == "submission"


def test_handlers_do_not_import_tool_registry() -> None:
    """Handlers wrap libraries; they must not orchestrate via the registry."""
    handlers_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "labpilot"
        / "research_engine"
        / "tools"
        / "handlers"
    )
    violations: list[str] = []
    forbidden = ("ToolRegistry", "build_default_tool_registry", "registry.invoke")
    for path in handlers_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.name}: {token}")
    assert not violations, "handlers must not chain tools:\n" + "\n".join(violations)


def test_engine_packages_do_not_import_tools_or_artifacts() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "labpilot" / "research_engine"
    package_dirs = ("intelligence", "planner", "execution", "evidence", "reflection")
    violations: list[str] = []
    for name in package_dirs:
        for path in (root / name).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    for forbidden in _FORBIDDEN_FROM_ENGINE:
                        if module == forbidden or module.startswith(forbidden + "."):
                            violations.append(f"{path}:{node.lineno}: {module}")
    assert not violations, "engine packages must not import tools/artifacts:\n" + "\n".join(
        violations
    )
