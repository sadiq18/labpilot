"""A declared defect has to stay a claim someone is still willing to make.

M24 exit criterion 5: *"Every `known_failure` carries a reason and a date; a
stale xfail is a lie about intent."*

The **reason** half was already enforced by the schema, and the *goes-green*
half by `test_competition_corpus.py` — the scorecard's red cells and the
fixture's declared ones must be the same set, so a defect that starts passing
fails the corpus. What was missing is the date, and what the date buys: an
xfail that has outlived its cause is indistinguishable from a live one, and
reads to the next person as a thing that was checked.

These tests exist because the corpus cannot exercise the boundary. One live
declaration, two days old, cannot tell a `>` from a `>=` — the same reason the
noise margin and the ratchet needed hand-built tests.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from labpilot.accessor.benchmark.fixture import STALE_AFTER_DAYS, KnownFailure

DECLARED = date(2026, 1, 1)


def _entry(**overrides: object) -> KnownFailure:
    return KnownFailure.model_validate(
        {"reason": "the spec is stale", "declared": DECLARED} | overrides
    )


# --- both halves are required -----------------------------------------------


def test_a_bare_reason_is_refused_by_name() -> None:
    """The pre-M24 shape was `criterion → reason`, and coercing it would have
    been the easy migration: every fixture written afterwards would carry a
    reason and no date, which is the state this model exists to end.
    """
    with pytest.raises(ValidationError) as caught:
        KnownFailure.model_validate("the spec is stale")

    assert "date" in str(caught.value), "the error should name what is missing"


def test_a_date_without_a_reason_is_refused() -> None:
    """A red cell with a timestamp and no cause is a defect nobody noticed."""
    with pytest.raises(ValidationError):
        KnownFailure.model_validate({"declared": DECLARED})


def test_an_empty_reason_is_refused() -> None:
    """`""` satisfies "has a reason" and answers nothing."""
    with pytest.raises(ValidationError):
        _entry(reason="")


def test_a_reason_without_a_date_is_refused() -> None:
    with pytest.raises(ValidationError):
        KnownFailure.model_validate({"reason": "the spec is stale"})


# --- the date is load-bearing -----------------------------------------------


def test_age_is_measured_from_the_declaration() -> None:
    assert _entry().age_days(DECLARED + timedelta(days=37)) == 37


def test_a_declaration_made_today_is_not_stale() -> None:
    assert not _entry().is_stale(DECLARED)


def test_the_ceiling_itself_is_still_current() -> None:
    """Stale is *past* the ceiling, not at it.

    Asserted because the corpus cannot: its one live declaration is days old,
    so `>` and `>=` agree there and a mutation between them would survive the
    whole integration suite.
    """
    assert not _entry().is_stale(DECLARED + timedelta(days=STALE_AFTER_DAYS))


def test_a_day_past_the_ceiling_is_stale() -> None:
    assert _entry().is_stale(DECLARED + timedelta(days=STALE_AFTER_DAYS + 1))


def test_staleness_is_about_the_claim_not_the_defect() -> None:
    """A reason may name a date older than the declaration — s6e7's spec was
    written three weeks before anyone recorded that it was wrong. Ageing the
    claim from the defect would restart every clock at the bug's birth.
    """
    entry = _entry(reason="written 2020-01-01 by a mapper since replaced")

    assert entry.age_days(DECLARED + timedelta(days=1)) == 1


def test_today_is_passed_in_rather_than_read() -> None:
    """No hidden clock: the same entry answers differently for two `today`s, so
    a caller can ask about a past corpus without patching anything.
    """
    entry = _entry()
    far = DECLARED + timedelta(days=STALE_AFTER_DAYS + 1)

    assert entry.is_stale(far)
    assert not entry.is_stale(DECLARED + timedelta(days=1))
