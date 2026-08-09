"""Techniques the templates can execute deterministically.

**Not a vocabulary.** This is the narrow, code-side set of techniques with a
recipe a template gate can run — bounded by how many gates someone wrote, so a
constant is the right shape here and nowhere else. An earlier version of this
module was called `KNOWN_TECHNIQUES`, which claimed to answer "is this a
technique?" — an open-world question that does not belong in a Python tuple
(design §8.7).

The three consumers, and which one this serves:

* **LLM codegen** — needs a *description*, and already receives the hypothesis
  triad in its prompt. It does not consult this module at all.
* **Template fallback** — needs a discrete switch (`{% if "target_encoding" in
  feature_recipes %}`). **This module exists for that consumer.**
* **Attribution / beliefs / dedup** — needs stable identity so evidence can be
  grouped. That vocabulary lives in the `techniques` store with a
  confirmed/candidate/rejected status, because it grows by learning.

So a name absent from here means "no deterministic recipe", never "not a
technique" — the run proceeds on its description via LLM codegen. Conflating
the two would discard exactly the novel techniques the system exists to find.

Canonical names match the labels `intelligence/feature_recipes.py` already
mines, so mined text and this registry cannot drift apart for the same idea.
"""

from __future__ import annotations

from labpilot.research_engine.execution.technique.models import TechniqueSpec

EXECUTABLE_TECHNIQUES: tuple[TechniqueSpec, ...] = (
    # --- feature recipes: mined vocabulary, implemented by codegen ---
    TechniqueSpec(
        name="target_encoding",
        aliases=["target encoding", "mean encoding", "likelihood encoding"],
        feature_recipes=["target_encoding"],
        applies_to=[],
        requires=["categorical_columns"],
        description="Replace a categorical level with a smoothed target statistic.",
    ),
    TechniqueSpec(
        name="log1p_transform",
        aliases=["log1p", "log transform", "log_numeric"],
        feature_recipes=["log_numeric"],
        requires=["numeric_columns"],
        description="log1p-compress skewed numeric columns.",
    ),
    TechniqueSpec(
        name="feature_interactions",
        aliases=["interactions", "interaction features", "crossed features"],
        feature_recipes=["feature_interactions"],
        requires=["numeric_columns"],
        description="Pairwise products of numeric drivers.",
    ),
    TechniqueSpec(
        name="polynomial_features",
        aliases=["polynomial", "poly features"],
        feature_recipes=["polynomial_features"],
        requires=["numeric_columns"],
        description="Polynomial expansion of numeric columns.",
    ),
    TechniqueSpec(
        name="lag_features",
        aliases=["lag", "lagged features", "shifted features"],
        feature_recipes=["lag_features"],
        requires=["partitioned"],
        description="Values from earlier rows within the same partition.",
    ),
    TechniqueSpec(
        name="rolling_features",
        aliases=["rolling", "rolling window", "moving average features"],
        feature_recipes=["rolling_features"],
        requires=["partitioned"],
        description="Rolling-window statistics within a partition.",
    ),
    TechniqueSpec(
        name="aggregation_features",
        aliases=["aggregation", "group statistics", "groupby features"],
        feature_recipes=["aggregation_features"],
        requires=["partitioned"],
        description="Per-partition aggregate statistics broadcast back to rows.",
    ),
    TechniqueSpec(
        name="frequency_encoding",
        aliases=["frequency encoding", "count encoding"],
        feature_recipes=["frequency_encoding"],
        requires=["categorical_columns"],
        description="Replace a category with how often it occurs.",
    ),
    TechniqueSpec(
        name="one_hot_encoding",
        aliases=["one hot", "one-hot", "dummies"],
        feature_recipes=["one_hot_encoding"],
        requires=["categorical_columns"],
        description="Indicator column per category level.",
    ),
    TechniqueSpec(
        name="binning",
        aliases=["bucketing", "discretisation", "discretization"],
        feature_recipes=["binning"],
        requires=["numeric_columns"],
        description="Discretise a numeric column into bins.",
    ),
    # --- model-family / hyperparameter techniques ---
    # `catboost` deliberately absent: it would set `model_family`, which
    # `CodeRenderer.render` does not accept (design §9.4 says it should). The
    # resolution would report `applied` while the rendered bytes were unchanged
    # — provenance asserting work that did not happen. Re-add it together with
    # the renderer change, not before.
    TechniqueSpec(
        name="deeper_trees",
        aliases=["deeper trees", "deeper tree depth"],
        model_params={"max_depth": 10, "num_leaves": 127},
        description="Raise tree depth and leaf count.",
    ),
    TechniqueSpec(
        name="more_estimators",
        aliases=["more estimators", "more boosting rounds"],
        model_params={"n_estimators": 2000},
        description="Train more boosting rounds.",
    ),
)


_BY_NAME: dict[str, TechniqueSpec] = {}
for _spec in EXECUTABLE_TECHNIQUES:
    _BY_NAME[_spec.name.lower()] = _spec
    for _alias in _spec.aliases:
        _BY_NAME[_alias.strip().lower()] = _spec


def canonical_name(label: str) -> str | None:
    """Map a spelling to a canonical *executable* technique name, else None.

    None means "no deterministic recipe for this", not "this is not a
    technique" — that judgement belongs to the vocabulary store. Reading None
    as rejection is how a genuinely new method from a paper would end up in the
    same bucket as `hyp:H-010`.
    """
    spec = _BY_NAME.get(str(label).strip().lower())
    return spec.name if spec else None


def get_technique(label: str) -> TechniqueSpec | None:
    """Resolve a spelling to its spec, following aliases."""
    return _BY_NAME.get(str(label).strip().lower())


def executable_names() -> tuple[str, ...]:
    """Canonical names of the recipe-backed set — what the templates can run."""
    return tuple(spec.name for spec in EXECUTABLE_TECHNIQUES)
