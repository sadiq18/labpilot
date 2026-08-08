"""Typed artifact for Code Engineering micro-agent proposals."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CodeFileSpec(BaseModel):
    """One file the agent wants written under the competition workspace."""

    path: str
    content: str
    action: Literal["write"] = "write"


class CodeProposal(BaseModel):
    """LLM/rule_engine output — deterministic code applies this to disk."""

    summary: str = ""
    rationale: str = ""
    files: list[CodeFileSpec] = Field(default_factory=list)

    # --- what the author says the change did (M19 §5) ----------------------
    #
    # Code identifiers, named by the agent that wrote the file. The point is to
    # get a claim that can be checked *mechanically* against the code, which
    # technique names cannot be: `SWA` is not an importable symbol, and rogii's
    # plans recorded `feature_engineering` (a category) and once the bare word
    # `the`. Nothing maps a technique to a symbol, and hand-maintaining that map
    # is the curated-set pattern already rejected three times.
    #
    # Self-reported, and deliberately so. This cannot catch a model that lies
    # consistently — but the real failure is carelessness, not deception. The
    # declaration records *intent*, the file records *execution*, and the gap
    # between them is exactly the thing that makes an evidence card wrong: a
    # card reading "ensembling improved MSE" when the delta quietly substituted.
    #
    # Empty is honest and normal — a baseline keeps nothing and combines
    # nothing. An empty claim yields no verdict rather than a fabricated pass.

    #: Symbols the change was supposed to preserve from the parent.
    kept: list[str] = Field(default_factory=list)
    #: Symbols the change was supposed to introduce.
    added: list[str] = Field(default_factory=list)
    #: Symbols whose outputs were supposed to be blended into one prediction.
    #: Distinct from `added`: "added but never averaged" is the quietest
    #: failure, because the constructor is present while the score reflects the
    #: parent alone.
    combined: list[str] = Field(default_factory=list)

    @field_validator("kept", "added", "combined", mode="before")
    @classmethod
    def _tolerate_null_claim(cls, value: object) -> object:
        """`null` means "nothing", not "invalid".

        Models emit `"kept": null` about as readily as `[]`. Rejecting it raises
        a `ValidationError`, which the retry path reads as a malformed response
        and re-asks — burning a step from a 30-step campaign over a field that
        is optional metadata. A single string is coerced for the same reason:
        the failure mode of being strict here is losing an experiment, while
        the failure mode of being lenient is an accurate claim in a slightly
        different shape.

        Each name is stripped, because the check matches it against symbols the
        AST parser extracts *without* surrounding whitespace: a stray `" lgb "`
        would never match the `lgb` in the file and would report a correct
        experiment as inconsistent. Blanks drop out for the same reason `""`
        does — they name nothing.
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
