"""`is_planner_visible` is the single shared status-filter predicate for four
call sites (filter_by_technique_status, the stagnation minter, fetchers.py's
retrieval, generate_candidates). Its contract must match what `dict.get(key,
"candidate")` gave every one of them before migration: default only on a
*missing* key, not on a present-but-falsy value.
"""

from __future__ import annotations

from labpilot.research_engine.execution.technique.status_constants import (
    PLANNER_VISIBLE_STATUSES,
    is_planner_visible,
)


def test_a_missing_status_defaults_to_candidate_and_is_visible() -> None:
    assert is_planner_visible(None) is True


def test_an_explicit_empty_status_is_not_silently_treated_as_candidate() -> None:
    """`str(status or "candidate")` would collapse "" into "candidate" (visible)
    -- but `dict.get(key, "candidate")`, the pattern this replaces, only
    applies that default when the key itself is absent. A present-but-empty
    value must stay filtered out, not leak through as if unset.
    """
    assert is_planner_visible("") is False


def test_confirmed_and_candidate_are_visible() -> None:
    assert is_planner_visible("candidate") is True
    assert is_planner_visible("confirmed") is True


def test_rejected_and_dormant_are_not_visible() -> None:
    assert is_planner_visible("rejected") is False
    assert is_planner_visible("dormant") is False


def test_a_custom_visible_set_is_honored() -> None:
    assert is_planner_visible("dormant", visible=frozenset({"dormant"})) is True
    assert is_planner_visible("candidate", visible=frozenset({"dormant"})) is False


def test_default_visible_set_is_the_shared_constant() -> None:
    assert is_planner_visible("candidate") == ("candidate" in PLANNER_VISIBLE_STATUSES)
