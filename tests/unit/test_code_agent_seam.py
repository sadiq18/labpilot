"""The seam exists so a second implementation can land without a rewrite.

These tests pin the contract, not the wrapper: what `propose` must return, what
it must *not* do (touch the workspace), and that the existing whole-file path
still satisfies it.
"""

from __future__ import annotations

from pathlib import Path

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.execution.delta import CodeAgent, WholeFileAgent
from labpilot.research_engine.execution.schemas.code_proposal import (
    CodeFileSpec,
    CodeProposal,
)


class _StubEngineer:
    """Stands in for `CodeEngineerAgent`, whose `run` is an LLM call."""

    def __init__(self, result=None):
        self.result = result if result is not None else CodeProposal(summary="ok")
        self.seen: list[StructuredContext] = []

    def run(self, context):
        self.seen.append(context)
        return self.result


def _ctx(**data):
    return StructuredContext(competition="rogii", question="beat the baseline", data=data)


def test_the_whole_file_agent_satisfies_the_protocol():
    assert isinstance(WholeFileAgent(agent=_StubEngineer()), CodeAgent)


def test_it_returns_the_underlying_proposal():
    proposal = CodeProposal(
        summary="add catboost",
        files=[CodeFileSpec(path="pipeline/train.py", content="x = 1\n")],
    )
    agent = WholeFileAgent(agent=_StubEngineer(proposal))
    assert agent.propose(_ctx(), None) is proposal


def test_a_soft_failure_becomes_an_empty_proposal_not_an_exception():
    """The agent soft-fails when the LLM is unavailable; the caller decides
    whether to stub or abandon. Raising here would move that decision."""
    agent = WholeFileAgent(agent=_StubEngineer(result=object()))
    assert agent.propose(_ctx(), None).files == []


def test_the_context_reaches_the_underlying_agent_unchanged():
    engineer = _StubEngineer()
    ctx = _ctx(prior_train_py="import lightgbm\n", technique="catboost")
    WholeFileAgent(agent=engineer).propose(ctx, None)
    assert engineer.seen == [ctx]


def test_the_parent_is_accepted_and_ignored_by_the_whole_file_agent(tmp_path: Path):
    """Prior code reaches the model as `prior_train_py`, already compressed
    upstream. Reading it off `parent` too would give one string two sources."""
    engineer = _StubEngineer()
    parent = tmp_path / "workspace"
    (parent / "pipeline").mkdir(parents=True)
    (parent / "pipeline" / "train.py").write_text("import lightgbm\n")

    WholeFileAgent(agent=engineer).propose(_ctx(), parent)

    assert engineer.seen[0].data.get("prior_train_py") is None


def test_proposing_never_writes_to_the_workspace(tmp_path: Path):
    """The invariant everything else rests on: a bad proposal is rejected
    before it touches anything, and `apply_proposal` stays the only writer."""
    parent = tmp_path / "workspace"
    (parent / "pipeline").mkdir(parents=True)
    train = parent / "pipeline" / "train.py"
    train.write_text("original\n")
    before = sorted(p.relative_to(parent) for p in parent.rglob("*"))

    WholeFileAgent(
        agent=_StubEngineer(
            CodeProposal(files=[CodeFileSpec(path="pipeline/train.py", content="new\n")])
        )
    ).propose(_ctx(), parent)

    assert train.read_text() == "original\n"
    assert sorted(p.relative_to(parent) for p in parent.rglob("*")) == before


def test_the_agent_is_named_for_attribution():
    """Two experiments whose code came from different agents are not
    comparable without the producer recorded alongside the result."""
    assert WholeFileAgent(agent=_StubEngineer()).name == "whole_file"


def test_an_agent_that_only_edits_files_does_not_satisfy_the_protocol():
    class DirectEditor:
        name = "bad"

        def apply(self, ctx, parent):  # not `propose`
            ...

    assert not isinstance(DirectEditor(), CodeAgent)
