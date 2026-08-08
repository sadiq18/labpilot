"""The seam: one way to ask for code, two ways to produce it.

The design (§4) writes the protocol as ``propose(ctx: CodegenContext, parent)``.
There is no ``CodegenContext`` in this codebase — every micro-agent takes
`StructuredContext`, and inventing a second envelope for one caller would split
the input type the whole agent layer shares. The protocol below uses the real
one.

Two implementations sit behind it: `WholeFileAgent`, wrapping today's
`CodeEngineerAgent`, and (next) `AiderAgent`. The point of the boundary is
reversibility — the same reason `fitroute` was put behind one. If aider turns
out to be a bad bet, what gets deleted is one class, not the execution path.

`parent` is passed separately from `ctx` rather than stuffed into `ctx.data`
because it is a *path to a tree*, not a compressed signal. A delta-based agent
copies and edits it; a whole-file agent reads one file out of it and otherwise
ignores it. Making that difference explicit in the signature is what keeps the
whole-file path from quietly depending on the workspace layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.execution.schemas.code_proposal import CodeProposal


@runtime_checkable
class CodeAgent(Protocol):
    """Produce a `CodeProposal` for one experiment.

    Returning a proposal rather than editing the workspace is the invariant the
    rest of the system is built on: a bad proposal is rejected before it touches
    anything, and `apply_proposal` stays the only writer. An implementation that
    edits files directly satisfies the type and breaks the design.
    """

    #: Names the producer on the evidence card and in `agent_invocations`, so a
    #: result can be attributed to the agent that generated it. Two experiments
    #: whose code came from different agents are not comparable without it.
    name: str

    def propose(self, ctx: StructuredContext, parent: Path | None) -> CodeProposal:
        """Propose code for `ctx`, given the parent workspace (None for a baseline)."""
        ...


class WholeFileAgent:
    """Today's generate-from-scratch path, behind the new protocol.

    A thin wrapper, deliberately: it exists so that "which agent produced this?"
    becomes a recorded fact rather than an assumption, *before* a second
    implementation lands. Introducing the seam and the second implementation in
    one change would leave no way to tell a protocol bug from an aider bug.

    Kept even after `AiderAgent` ships, because baselines have no parent to diff
    against and not every workspace will have aider installed.
    """

    name = "whole_file"

    def __init__(self, agent=None, llm_client=None) -> None:
        if agent is None:
            from labpilot.research_engine.execution.micro_agents.code_engineer.agent import (
                CodeEngineerAgent,
            )

            agent = CodeEngineerAgent(llm_client=llm_client)
        self._agent = agent

    def propose(self, ctx: StructuredContext, parent: Path | None) -> CodeProposal:
        """`parent` is unused: this agent regenerates the file from scratch.

        The prior code still reaches the model — as `prior_train_py` in
        `ctx.data`, already compressed upstream. Reading it off `parent` here
        would give the same string a second, divergent source of truth.
        """
        result = self._agent.run(ctx)
        if isinstance(result, CodeProposal):
            return result
        # A soft-failed agent returns an empty model rather than raising; the
        # caller decides whether to stub or abandon. Not re-raising here keeps
        # that decision where it already lives.
        return CodeProposal()
