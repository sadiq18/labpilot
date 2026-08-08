"""``DeltaBriefAgent`` — turn a hypothesis into an instruction plus a claim.

Runs **before** the editor, which is the whole point: M19 §5's checks need a
claim that is independent of the code, and a claim written after the diff cannot
test the diff. This produces intent; aider produces execution; the gap between
them is the finding.

It is deliberately a separate call from codegen rather than an extra field on
`CodeProposal`. aider is a subprocess that returns a diff and nothing else, so
there is no response to attach a claim to — and asking aider for the claim would
put it downstream of the code, which is the circularity §5 rules out.

Soft-fails: a brief that cannot be produced leaves the delta `delta_unchecked`,
which is exactly the honest state 1b already defines for a delta that claimed
nothing. Losing an experiment because the *metadata* call failed would be the
worse trade.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.execution.schemas.delta_brief import DeltaBrief

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

#: The parent is included so the model can name symbols that actually exist,
#: rather than inventing plausible ones. Bounded because this is a cheap
#: metadata call and the whole file is rarely needed to identify a model symbol.
_PARENT_BUDGET = 12_000


class DeltaBriefAgent(BaseMicroAgent):
    name = "DeltaBriefAgent"
    output_model = DeltaBrief
    #: `reasoning`, not `codegen`. This writes no code — it reads a hypothesis
    #: and names symbols — and pinning it to the codegen role would make every
    #: delta pay codegen prices for a short structured answer.
    llm_role = "reasoning"

    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client)
        self._prompts_dir = _PROMPTS_DIR
        self._env = Environment(
            loader=FileSystemLoader(self._prompts_dir),
            autoescape=select_autoescape(default=False),
        )

    def system_prompt(self) -> str:
        return (self._prompts_dir / "delta_brief_system.md").read_text(encoding="utf-8")

    def user_prompt(self, context: StructuredContext) -> str:
        data = context.data
        return self._env.get_template("delta_brief_user.j2").render(
            competition=context.competition,
            problem_type=str(data.get("problem_type") or "unknown"),
            plan_goal=str(data.get("plan_goal") or context.question or ""),
            observation=str(data.get("observation") or ""),
            reason=str(data.get("reason") or ""),
            prediction=str(data.get("prediction") or ""),
            technique=str(data.get("technique") or ""),
            prior_train_py=str(data.get("prior_train_py") or "")[:_PARENT_BUDGET],
            retry_reason=str(data.get("retry_reason") or ""),
        )
