"""The published technique vocabulary — step 1 of the producer/consumer contract.

The registry exists so a producer can ask "will the executor understand this?"
*before* writing it. Every property below is one a producer will rely on.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.execution.technique import (
    EXECUTABLE_TECHNIQUES,
    canonical_name,
    executable_names,
    get_technique,
)
from labpilot.research_engine.execution.technique.registry import gated_recipes


def test_every_entry_declares_something_or_is_named_as_llm_only():
    """An entry that changes nothing and has no description is a dead promise."""
    for spec in EXECUTABLE_TECHNIQUES:
        assert spec.description, f"{spec.name} has no description"
        assert spec.has_recipe(), (
            f"{spec.name} is in the registry but changes nothing; either give it "
            "a recipe/model change or drop it from the vocabulary"
        )


def test_canonical_names_are_unique():
    names = [spec.name for spec in EXECUTABLE_TECHNIQUES]
    assert len(names) == len(set(names))


def test_no_alias_collides_across_techniques():
    """A colliding alias would silently merge two techniques into one finding."""
    seen: dict[str, str] = {}
    for spec in EXECUTABLE_TECHNIQUES:
        for alias in [spec.name, *spec.aliases]:
            key = alias.strip().lower()
            assert key not in seen or seen[key] == spec.name, (
                f"alias {alias!r} maps to both {seen.get(key)} and {spec.name}"
            )
            seen[key] = spec.name


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("target_encoding", "target_encoding"),
        ("Target Encoding", "target_encoding"),
        ("mean encoding", "target_encoding"),
        ("  LIKELIHOOD ENCODING  ", "target_encoding"),
        ("log1p", "log1p_transform"),
        ("rolling", "rolling_features"),
    ],
)
def test_aliases_resolve_to_one_canonical_name(spelling, expected):
    """Synonyms must collapse, or the same method accumulates evidence under
    several names and none of them reaches significance."""
    assert canonical_name(spelling) == expected


@pytest.mark.parametrize(
    "label",
    [
        "hyp:H-010",          # record reference — the fabricated class
        "add",                # regex fallback junk: "add features"
        "computed",           # "we computed features"
        "dataset",            # "the dataset features"
        "context",
        "built",
        "feature_engineering",  # the miner's catch-all, a category not a method
        "vit",                # real technique, wrong modality — still unknown here
        "",
    ],
)
def test_unknown_labels_resolve_to_none_not_to_a_guess(label):
    """None means *not in the vocabulary* — a candidate for review.

    Every string here was observed in rogii's knowledge base being treated as a
    technique. The registry's job is to not recognise them; deciding what
    happens next is the normaliser's, and it must be able to tell "unknown"
    from "rejected".
    """
    assert canonical_name(label) is None
    assert get_technique(label) is None


def test_executable_names_are_the_contract_surface():
    names = executable_names()
    assert "target_encoding" in names
    assert len(names) == len(EXECUTABLE_TECHNIQUES)
    # Canonical names only — a producer validating against aliases would accept
    # spellings the executor's own lookups do not use.
    assert all(canonical_name(n) == n for n in names)


def test_recipe_backed_techniques_name_their_gate_requirement():
    """Which recipes no tabular template can execute yet.

    Gates are read from template source, so this cannot drift silently: adding
    a gate shrinks the list and fails here until someone updates it — which is
    the point, since a registry implying a capability the templates lack is how
    `applied` gets recorded for a run that changed nothing.
    """
    tabular = gated_recipes("tabular_regression") | gated_recipes(
        "tabular_regression_partitioned"
    )
    ungated = sorted(
        {r for spec in EXECUTABLE_TECHNIQUES for r in spec.feature_recipes} - tabular
    )
    assert ungated == [
        "aggregation_features",
        "binning",
        "feature_interactions",
        "frequency_encoding",
        "lag_features",
        "one_hot_encoding",
        "polynomial_features",
        "rolling_features",
    ], "update this list as template gates are added, so the gap stays visible"


def test_no_entry_declares_model_family_until_the_renderer_accepts_it():
    """`CodeRenderer.render` has no `model_family` argument (design §9.4 says
    it should). An entry setting it would resolve as `applied` while the
    rendered bytes were unchanged — provenance asserting work that did not
    happen. `catboost` is held back for exactly this reason."""
    import inspect

    from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen.renderer import (  # noqa: E501
        CodeRenderer,
    )

    accepted = inspect.signature(CodeRenderer.render).parameters
    offenders = [s.name for s in EXECUTABLE_TECHNIQUES if s.model_family]
    assert not offenders or "model_family" in accepted, (
        f"{offenders} declare model_family, which the renderer cannot apply"
    )
