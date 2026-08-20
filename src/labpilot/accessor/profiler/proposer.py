"""The model proposes; the data vetoes.

M22 step 6. A reader of the competition description can sometimes see what the
files cannot — that `churn_flag` is the label, that `(store, date)` is the key —
and that is worth having. What it must never be is an *answer*, because a model
asked "which column is the target?" will always name one.

So the contract is narrow, and every clause of it is a guard:

* **It never writes a value.** `apply_proposal` touches `inferences` only. The
  value plane is byte-identical whether the proposer is absent, right, or wrong
  on every field — which is what makes "propose-only" a mechanism rather than a
  comment, and it is checked by a test rather than promised here.
* **It does not see our answer.** The prompt carries the description, the goal
  and the column *names*; not `target_column`, not the candidates, not a
  confidence. Withholding it keeps agreement worth something: a model shown the
  answer agrees with it, and that agreement is worth nothing.
* **Every claim faces a structural verifier**, and the three outcomes are all
  recorded — confirmed adds 0.10 and never more, contradicted goes to
  `rejected` and never near the value, and a claim about a field the profiler
  could not resolve becomes an *alternative*, so it reaches the human deciding
  the question instead of quietly becoming the decision.
* **Off by default** (`profiler.llm_proposals`).

Design: ``docs/research-os/autonomy-roadmap/design/17-dataset-understanding.md``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.accessor.profiler.evidence import Alternative, Inference, RejectedClaim, Signal

if TYPE_CHECKING:
    from labpilot.accessor.profiler.tabular import DatasetProfile

__all__ = [
    "SchemaProposal",
    "SchemaProposalAgent",
    "apply_proposal",
    "propose_schema",
]


class SchemaProposal(BaseModel):
    """What a reader of the description thinks the schema is."""

    target_column: str | None = None
    id_columns: list[str] = Field(default_factory=list)
    #: Why, in the model's words. Recorded for a human reading the question, and
    #: never parsed — a rationale that had to be understood would be a second
    #: inference channel with no verifier behind it.
    reasoning: str = ""


class SchemaProposalAgent(BaseMicroAgent):
    """Reads the description and names columns. Sees no inference of ours."""

    name = "SchemaProposalAgent"
    output_model = SchemaProposal
    llm_role = "reasoning"

    def system_prompt(self) -> str:
        return (
            "You read a dataset description and say which column is the "
            "prediction target and which column(s) identify a row. Reply ONLY "
            'with JSON: {"target_column": str|null, "id_columns": [str], '
            '"reasoning": str}. Use only column names from the list given. If '
            "the description does not say, answer null and an empty list — a "
            "guess is worse than nothing here, because a human will be asked "
            "either way."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        columns = ", ".join(str(name) for name in context.data.get("columns") or [])
        return (
            f"Competition: {context.competition}\n"
            f"Goal: {context.question}\n"
            f"Description:\n{context.text}\n\n"
            f"Columns: {columns}"
        )


def propose_schema(
    profile: DatasetProfile,
    *,
    llm_client: object,
    title: str = "",
    description: str = "",
    goal: str = "",
) -> SchemaProposal | None:
    """Ask for a proposal, or None if the model could not give one.

    Deliberately forgiving: a proposer that fails must leave the profile exactly
    as it would have been without one, and an exception here is not a reason to
    fail a profile.
    """
    agent = SchemaProposalAgent(llm_client=llm_client)  # type: ignore[arg-type]
    context = StructuredContext(
        competition=profile.competition,
        question=goal or title,
        text=description,
        data={"columns": [column.name for column in profile.columns]},
    )
    try:
        result = agent.run(context)
    except Exception:  # noqa: BLE001 — a proposal is optional by construction
        return None
    return result if isinstance(result, SchemaProposal) else None


def _column(profile: DatasetProfile, name: str):
    return next((column for column in profile.columns if column.name == name), None)


def _verify(profile: DatasetProfile, field: str, name: str) -> tuple[bool, str]:
    """Does the data support this claim? Returns (verified, why).

    A verifier that *cannot run* is not a pass: `dtype_matches_metric` needs a
    metric, and a dataset without one leaves the claim resting on the checks
    that did run. What no claim may do is arrive with nothing checked at all,
    which is why `column_exists` runs for every field.
    """
    column = _column(profile, name)
    if column is None:
        return False, "names no column in this dataset"
    if field == "target_column":
        if profile.train_only_columns and name not in profile.train_only_columns:
            return False, "is present in the scoring input, so it cannot be the label"
        # The shape the metric needs, as the registry states it — not a list of
        # keys kept here, which would be a second vocabulary drifting from the
        # first. `any` and an undeclared metric both mean "cannot check".
        kind = profile.metric.target_kind if profile.metric else None
        metric_name = profile.metric.name if profile.metric else "the metric"
        if kind == "continuous" and not column.is_numeric:
            return False, f"is not numeric, and {metric_name} scores a quantity"
        if kind == "discrete" and column.is_numeric and column.unique_count > 50:
            return False, (
                f"has {column.unique_count} distinct values, and {metric_name} scores classes"
            )
        return True, "withheld at scoring, and its dtype suits the metric"
    if field == "id_columns":
        units = profile.column_stats_rows or profile.row_count
        if units and column.unique_count != units:
            return False, f"has {column.unique_count} distinct values in {units} rows"
        return True, "one distinct value per row"
    return True, "the column exists"


def apply_proposal(profile: DatasetProfile, proposal: SchemaProposal) -> None:
    """Fold a proposal into the evidence plane. Never into the value plane.

    Three outcomes per claim, all recorded:

    * the claim names what the profiler already resolved and verifies —
      `llm_proposal_confirmed` (0.10), once, however emphatic the model was;
    * a verifier refuses it — a :class:`RejectedClaim`, which is evidence about
      the *source* and is kept rather than dropped;
    * the field is unresolved and the claim verifies — an alternative, so it
      reaches the person answering the question rather than answering it.
    """
    claimed: dict[str, str | None] = {
        "target_column": proposal.target_column,
        "id_columns": proposal.id_columns[0] if proposal.id_columns else None,
    }
    resolved: dict[str, str | None] = {
        "target_column": profile.target_column,
        "id_columns": profile.id_columns[0] if profile.id_columns else None,
    }

    for field, name in claimed.items():
        if not name:
            continue
        inference = profile.inferences.get(field)
        if inference is None:
            continue
        verified, why = _verify(profile, field, name)
        if not verified:
            profile.inferences[field] = inference.model_copy(
                update={
                    "rejected": [
                        *inference.rejected,
                        RejectedClaim(claim=name, source="llm", refuted_by=why),
                    ]
                }
            )
            continue
        signal = Signal(id="llm_proposal_confirmed", detail=f"a model proposed {name!r}: {why}")
        if name == resolved[field]:
            # Agreement with what the data already said. Ten points, and the
            # profile keeps showing which of the two came from evidence.
            profile.inferences[field] = Inference.of(
                [*inference.signals, signal],
                alternatives=inference.alternatives,
                rejected=inference.rejected,
            )
        else:
            # A different column, or a field nothing resolved. An alternative —
            # visible in the question, never the answer.
            profile.inferences[field] = Inference.of(
                inference.signals,
                alternatives=[*inference.alternatives, Alternative.of(name, [signal])],
                rejected=inference.rejected,
            )
