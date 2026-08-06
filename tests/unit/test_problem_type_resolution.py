"""The cross-modality filter must know what kind of problem this is.

Measured on rogii 2026-08-07: `vit` (a vision transformer) was proposed for a
tabular regression, executed as E-022, and then propagated into every
subsequent `technique_stack`.

`filter_incompatible_techniques` was correctly configured the whole time —
`vit` is in its vision token set and tabular allows no vision tokens. It never
fired because it returns early on an empty modality, and `_resolve_problem_type`
read only `competition.json`, which says ``unknown``. Two other sources knew:
`baseline_choice.json` says `tabular_regression` and `profile.json` says
`tabular`. The guard was reading the one source of three that did not know —
the same shape as the `hyp:` guard defeated by normalisation.
"""

from __future__ import annotations

import json

import pytest

from labpilot.research_engine.intelligence.hypothesis.assistant import _resolve_problem_type
from labpilot.research_engine.intelligence.hypothesis.candidates import (
    filter_incompatible_techniques,
)


class _Candidate:
    def __init__(self, technique: str) -> None:
        self.technique = technique
        self.title = technique
        self.key = technique


def _workspace(tmp_path, *, competition_type=None, baseline_type=None):
    """A workspace root with `knowledge/` beneath it, as the resolver expects."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    if competition_type is not None:
        (tmp_path / "competition.json").write_text(
            json.dumps({"slug": "demo", "problem_type": competition_type}), encoding="utf-8"
        )
    if baseline_type is not None:
        (tmp_path / "baseline_choice.json").write_text(
            json.dumps({"problem_type": baseline_type}), encoding="utf-8"
        )
    return knowledge


def test_unknown_competition_type_falls_through_to_the_derived_one(tmp_path):
    """rogii's exact shape: the spec says unknown, the profile-derived choice knows."""
    knowledge = _workspace(tmp_path, competition_type="unknown", baseline_type="tabular_regression")
    assert _resolve_problem_type(knowledge, "demo") == "tabular_regression"


def test_a_known_competition_type_still_wins(tmp_path):
    """The spec is authoritative when it actually knows something."""
    knowledge = _workspace(
        tmp_path, competition_type="image_classification", baseline_type="tabular_regression"
    )
    assert _resolve_problem_type(knowledge, "demo") == "image_classification"


@pytest.mark.parametrize("empty", ["", "unknown", "UNKNOWN", "  "])
def test_every_flavour_of_not_knowing_falls_through(tmp_path, empty):
    knowledge = _workspace(tmp_path, competition_type=empty, baseline_type="tabular_regression")
    assert _resolve_problem_type(knowledge, "demo") == "tabular_regression"


def test_no_sources_resolves_empty_rather_than_raising(tmp_path):
    """Filtering is an optimisation, not a gate — it must never break generation."""
    assert _resolve_problem_type(tmp_path / "knowledge", "demo") == ""


def test_vision_techniques_are_dropped_once_the_type_resolves(tmp_path):
    """The effect, not the lookup: this is what stops another `vit` experiment."""
    knowledge = _workspace(tmp_path, competition_type="unknown", baseline_type="tabular_regression")
    problem_type = _resolve_problem_type(knowledge, "demo")

    kept, dropped = filter_incompatible_techniques(
        [_Candidate("vit"), _Candidate("cnn"), _Candidate("rolling_features"), _Candidate("SWA")],
        problem_type,
    )

    assert sorted(dropped) == ["cnn", "vit"]
    assert [c.technique for c in kept] == ["rolling_features", "SWA"]


def test_an_unresolved_type_disables_the_filter(tmp_path):
    """The failure mode itself, pinned: with no resolvable type the filter is a
    no-op, which is how `vit` got through. Kept as a test so nobody 'fixes' the
    early return without realising it is load-bearing."""
    kept, dropped = filter_incompatible_techniques([_Candidate("vit")], "")
    assert dropped == []
    assert len(kept) == 1
