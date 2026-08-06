"""What a technique *is*, declaratively.

A spec says what a technique changes about training — never how to apply it
(templates own that) and never whether it is a good idea (the Conductor owns
that). Adding a technique is then a registry entry plus a template gate.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TechniqueSpec(BaseModel):
    """One executable technique in the published vocabulary."""

    #: Canonical name, used as the stable key for attribution and dedup —
    #: the reason identity exists at all (design §8.7). Matches the labels the
    #: recipe miner already emits
    #: (`intelligence/feature_recipes.py::_name_from_sentence`), so mined text
    #: and this registry cannot drift into two vocabularies for one idea.
    name: str
    #: Spellings that mean the same technique. The normaliser maps these to
    #: `name`, which is what keeps "mean encoding" and "target encoding" from
    #: being recorded as two different findings about the same method.
    aliases: list[str] = Field(default_factory=list)

    #: Passed to `CodeRenderer.render(feature_recipes=...)`, where each entry
    #: switches on a `{% if %}` gate in the template.
    feature_recipes: list[str] = Field(default_factory=list)
    model_family: str | None = None
    #: Merged over DEFAULT_TABULAR_MODEL_PARAMS at render time.
    model_params: dict[str, Any] = Field(default_factory=dict)

    #: Problem types this applies to; empty means any. Checked at plan time so
    #: an inapplicable technique never consumes a training run.
    applies_to: list[str] = Field(default_factory=list)
    #: Data preconditions — `categorical_columns`, `numeric_columns`,
    #: `partitioned`. Checked against the DatasetProfile.
    requires: list[str] = Field(default_factory=list)
    description: str = ""

    def has_recipe(self) -> bool:
        """True when this technique can be executed deterministically.

        Every entry in the shipped set satisfies this — a spec that changes
        nothing is a dead promise. The method stays because a spec can be built
        outside the registry (e.g. from a store record) where the distinction
        between "recognised" and "executable" is real.
        """
        return bool(self.feature_recipes or self.model_family or self.model_params)
