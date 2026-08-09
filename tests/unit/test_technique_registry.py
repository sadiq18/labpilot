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



