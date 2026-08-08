"""What to ask an editor for, and what that edit must be checkable against.

M19 §5's checks compare the code to a claim, and the claim has to be
**independent of the code** — a claim read off the diff cannot test the diff.
For whole-file codegen the author declares it (`CodeProposal.kept/added/
combined`). aider returns a diff and no structured claim, so an aider proposal
arrived with nothing to check and every delta landed `delta_unchecked`: three of
the four checks going dark precisely when deltas became the thing they were
built for.

This closes it without breaking independence. The brief is produced **before**
aider runs, from the hypothesis alone, so it is a statement of intent that the
resulting edit is then measured against. The order is what makes it legitimate:
intent first, execution second, and the gap between them is the finding.

Deriving it from the diff afterwards, or asking aider to report what it did,
would both put the claim downstream of the code and make the check circular —
§5 rules out both, and this is the third mechanism proposed for the job after
`technique` metadata and a technique→symbol map, each of which failed on the
same point.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class DeltaBrief(BaseModel):
    """An editing instruction plus the claim its result will be checked against."""

    #: What to tell the editor. Prose, imperative, one change.
    instruction: str = ""

    #: Symbols that must survive the edit — catches substitution disguised as
    #: addition, §5's first row.
    kept: list[str] = Field(default_factory=list)
    #: Symbols the edit must introduce, as called or imported names.
    added: list[str] = Field(default_factory=list)
    #: Symbols whose outputs must be blended into one prediction. Distinct from
    #: `added`: "added but never averaged" is the quietest failure, because the
    #: constructor is present while the score reflects the parent alone.
    combined: list[str] = Field(default_factory=list)

    @field_validator("kept", "added", "combined", mode="before")
    @classmethod
    def _tolerate_loose_shapes(cls, value: object) -> object:
        """Same leniency as `CodeProposal`, for the same reason.

        A `ValidationError` here is read by the retry path as a malformed
        response and re-asked, costing a step from a 30-step campaign over
        optional metadata. Names are stripped because the checks match them
        against symbols the AST parser extracts without surrounding whitespace.
        """
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [
                item.strip() if isinstance(item, str) else item
                for item in value
                if not isinstance(item, str) or item.strip()
            ]
        return value

    def claims_anything(self) -> bool:
        """True when the brief gives the checks something to verify."""
        return bool(self.kept or self.added or self.combined)
