"""When the aider path runs, and — more importantly — when it declines.

Each decline is a design decision, not a fallback of convenience, and each one
would be invisible if it regressed: the campaign would still produce code, just
by the wrong route and at the wrong price.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
    CodeEngineeringCapability,
)
from labpilot.research_engine.execution.schemas.code_proposal import (
    CodeFileSpec,
    CodeProposal,
)


class _Gateway:
    """Quacks like an LLMGateway: `for_role` is callable."""

    def for_role(self, role):  # pragma: no cover - identity only
        return self


class _NotAGateway:
    """A plain client. `for_role` absent — aider would bypass M10."""


class _Ctx:
    def __init__(self, root, **constraints):
        self.workspace_root = root
        self.constraints = constraints


PROPOSAL = CodeProposal(files=[CodeFileSpec(path="pipeline/train.py", content="x = 1\n")])


def _capability(llm=None) -> CodeEngineeringCapability:
    cap = CodeEngineeringCapability.__new__(CodeEngineeringCapability)
    cap._llm = llm
    return cap


def _patch_agent(monkeypatch, *, result=None, raises=None):
    import labpilot.research_engine.execution.delta.aider_agent as mod

    class _Fake:
        def __init__(self, gateway, **kw):
            self.gateway = gateway

        def propose(self, ctx, parent):
            if raises is not None:
                raise raises
            return result

    monkeypatch.setattr(mod, "AiderAgent", _Fake)


def test_the_delta_path_runs_when_configured_with_a_parent(tmp_path, monkeypatch):
    _patch_agent(monkeypatch, result=PROPOSAL)
    cap = _capability(_Gateway())

    proposal, origin = cap._propose_delta(
        _Ctx(tmp_path, codegen_strategy="delta"), object(), "prior code\n"
    )

    assert origin == "aider"
    assert proposal is PROPOSAL


def test_an_unset_constraint_follows_the_configured_default(tmp_path, monkeypatch):
    """The fallback is `CodegenConfig().strategy`, not a literal repeated here.

    This asserted `whole_file` for an unset constraint, which was the default
    until this milestone changed it — so the stale literal in the capability
    had a test holding it in place while `research resume`, which never set the
    constraint, quietly took the whole-file path. Reported on PR #118.
    """
    from labpilot.config import CodegenConfig

    _patch_agent(monkeypatch, result=PROPOSAL)
    cap = _capability(_Gateway())

    proposal, origin = cap._propose_delta(_Ctx(tmp_path), object(), "prior\n")

    took_delta = proposal is not None
    assert took_delta is (CodegenConfig().strategy == "delta")
    assert origin == ("aider" if took_delta else "")


def test_it_declines_when_whole_file_is_configured(tmp_path, monkeypatch):
    """§10: both paths coexist while the rate is measured, and asking for the
    whole-file path still gets it."""
    _patch_agent(monkeypatch, result=PROPOSAL)
    cap = _capability(_Gateway())

    assert cap._propose_delta(
        _Ctx(tmp_path, codegen_strategy="whole_file"), object(), "prior\n"
    ) == (None, "")


@pytest.mark.parametrize("prior", ["", "   \n"])
def test_it_declines_for_a_baseline(tmp_path, monkeypatch, prior):
    """No parent means nothing to diff against or preserve — the whole-file
    agent's job by design, and raising instead would hide the routing."""
    _patch_agent(monkeypatch, result=PROPOSAL)
    cap = _capability(_Gateway())

    assert cap._propose_delta(_Ctx(tmp_path, codegen_strategy="delta"), object(), prior) == (
        None,
        "",
    )


def test_it_declines_without_a_gateway(tmp_path, monkeypatch):
    """aider outside the proxy bypasses the ledger, rate limiting and failover
    — §4 calls that a regression dressed as a feature. Declining beats routing
    around M10."""
    _patch_agent(monkeypatch, result=PROPOSAL)
    cap = _capability(_NotAGateway())

    assert cap._propose_delta(_Ctx(tmp_path, codegen_strategy="delta"), object(), "prior\n") == (
        None,
        "",
    )


def test_a_gateway_is_detected_by_callable_not_by_attribute(tmp_path, monkeypatch):
    """A stub carrying `for_role` as a plain value is not a gateway."""
    _patch_agent(monkeypatch, result=PROPOSAL)
    stub = _NotAGateway()
    stub.for_role = "not callable"
    cap = _capability(stub)

    assert cap._propose_delta(_Ctx(tmp_path, codegen_strategy="delta"), object(), "prior\n") == (
        None,
        "",
    )


def test_an_aider_failure_falls_back_rather_than_killing_the_step(tmp_path, monkeypatch):
    """A campaign that cannot produce code because aider had a bad day measures
    nothing, and measurement is what step 2 decides on."""
    from labpilot.research_engine.execution.delta.aider_agent import AiderError

    _patch_agent(monkeypatch, raises=AiderError("no edit", kind="aider_no_edit"))
    cap = _capability(_Gateway())

    assert cap._propose_delta(_Ctx(tmp_path, codegen_strategy="delta"), object(), "prior\n") == (
        None,
        "",
    )


def test_an_unexpected_error_also_falls_back(tmp_path, monkeypatch):
    _patch_agent(monkeypatch, raises=RuntimeError("uvx exploded"))
    cap = _capability(_Gateway())

    assert cap._propose_delta(_Ctx(tmp_path, codegen_strategy="delta"), object(), "prior\n") == (
        None,
        "",
    )


def test_an_empty_proposal_is_not_accepted(tmp_path, monkeypatch):
    """ "aider returned something" is not "aider produced files"."""
    _patch_agent(monkeypatch, result=CodeProposal())
    cap = _capability(_Gateway())

    assert cap._propose_delta(_Ctx(tmp_path, codegen_strategy="delta"), object(), "prior\n") == (
        None,
        "",
    )


def test_every_task_context_sets_the_codegen_strategy():
    """The same bug three times on PR #118 — the capability's stale fallback,
    then `research resume`, then the Conductor's specialist path — because each
    `TaskContext` constructor owes this constraint and nothing said so.

    Enumerated from the source rather than listed here: a fourth constructor
    fails this test instead of quietly regenerating whole files for a week.

    Scoped to the **enclosing function**, not the file. The first version asked
    whether `"codegen_strategy"` appeared anywhere in a file that built a
    `TaskContext`, which held only because each file happened to have one
    construction — a second one added to either file would have passed on its
    neighbour's constraint. Reported on PR #118, and it is the same mistake the
    test it replaced made: checking that an identifier is present somewhere
    rather than that it is used where it matters.
    """
    import ast
    from pathlib import Path

    def enclosing_functions(tree: ast.Module) -> dict[int, ast.AST]:
        owner: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                for inner in ast.walk(node):
                    owner.setdefault(id(inner), node)
        return owner

    sites: list[tuple[str, str]] = []
    for path in Path("src/labpilot").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "TaskContext(" not in source:
            continue
        tree = ast.parse(source)
        owner = enclosing_functions(tree)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "TaskContext"
            ):
                continue
            function = owner.get(id(node))
            scope = ast.unparse(function) if function is not None else source
            sites.append((f"{path}:{node.lineno}", scope))

    assert sites, "no TaskContext construction found — has it been renamed?"
    missing = [where for where, scope in sites if "codegen_strategy" not in scope]
    assert not missing, f"TaskContext built without codegen_strategy at: {missing}"


def test_the_legacy_layout_reads_the_config_it_was_given():
    """Without a `labpilot.yaml` there is no workspace to read a
    per-competition config from, and passing `""` fell back to the packaged
    default instead of the config the CLI had already loaded. Reported on
    PR #118."""
    from labpilot.cli.run_engineer import _engineer_constraints
    from labpilot.config import AppConfig

    config = AppConfig()
    config.codegen.strategy = "whole_file"

    constraints = _engineer_constraints(config=config, workspace=None, dry_run=False, submit=False)

    assert constraints["codegen_strategy"] == "whole_file"


def test_the_funnel_default_reads_the_workspace_config():
    """Reported on PR #118: `ResearchEngineer`'s fill-in resolved the packaged
    default while the workspace it had just computed sat sixteen lines above —
    reproducing, in the one spot left open, the ignore-the-real-config flaw the
    rest of this change removes."""
    import inspect

    from labpilot.research_engine.execution.engineer import ResearchEngineer

    source = inspect.getsource(ResearchEngineer._run_task)

    assert 'resolve_codegen_strategy(workspace / "configs" / "default.yaml")' in source


def test_a_workspace_root_path_is_not_duck_typed():
    """`Path("/a/b").root` is `"/"`, so reading `.root` off a path turned a
    reasonable argument into the filesystem root. Reported on PR #118."""
    from pathlib import Path

    from labpilot.research_engine.execution.codegen_strategy import workspace_config_path

    assert workspace_config_path(Path("/a/b")) == Path("/a/b/configs/default.yaml")
    assert workspace_config_path("/a/b") == Path("/a/b/configs/default.yaml")
    assert workspace_config_path(None) is None
