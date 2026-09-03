"""Tier 3 — what licenses the hermetic corpus to stand in for real data.

The plan's own words: *"Tier 3 is the single most important test here… Tier 1 is
the gate, tier 2 is the truth, tier 3 keeps tier 1 honest about tier 2."*

Every other test of the corpus asks whether the system reads a **fixture**
correctly. This asks whether the fixture reads like the **dataset** — same
profiler, same expectations, same scorer, so a difference in the result is a
difference in the capture and nothing else.

A disagreement is not a bug in the profiler. It means that fixture's capture mode
is wrong for that criterion, and the remedy the plan names is to move the
criterion to `unverifiable`: a fixture that quietly answers differently from the
dataset it stands for is worse than one that admits it cannot answer.

**Skipped loudly**, per exit criterion 7 — it names every path it looked in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import labpilot.accessor.benchmark.score as score_module
from labpilot.accessor.benchmark.fixture import CompetitionFixture, load_fixture
from labpilot.accessor.benchmark.score import (
    disagreements,
    profile_and_score,
    score_full_dataset,
)

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "competitions"

#: Where a full dataset might be, in the order a machine is likely to have it.
#: An explicit override first, because a corpus that only works on the machine
#: that captured it is not a corpus.
_ENV_ROOT = "LABPILOT_CORPUS_FULL_DATA"
#: The default Kaggle cache, hardcoded rather than read from
#: `LABPILOT_KAGGLE_CACHE_DIR`. That variable *cannot* be read here: the autouse
#: `_no_real_dotenv_in_tests` fixture deletes it from the environment so
#: `Settings` cannot pick up a developer's real cache, so a test reading it
#: always saw `None`. The first version of this file read it anyway and printed
#: it in the skip message, which advertised a knob that silently did nothing —
#: an operator who set it got skips naming the very path they had overridden.
#: `_ENV_ROOT` is not on that list and does work, so it is the knob offered.
_DEFAULT_CACHE = Path.home() / "workspace" / ".labpilot-cache" / "kaggle"


def corpus_slugs() -> list[str]:
    return sorted(path.name for path in CORPUS.iterdir() if (path / "fixture.json").is_file())


def _candidates(fixture: CompetitionFixture) -> list[Path]:
    """Every place this dataset could be, most explicit first."""
    found: list[Path] = []
    override = os.environ.get(_ENV_ROOT)
    if override:
        found.append(Path(override) / fixture.slug)
    cache = _DEFAULT_CACHE
    found.append(cache / fixture.slug)
    # What the capture recorded. Absolute and machine-specific, so it is the
    # last resort rather than the first — but it is the only pointer for a
    # fixture taken from a workspace rather than the download cache.
    if fixture.source and not fixture.source.startswith("kaggle"):
        found.append(Path(fixture.source))
    return found


def _full_dataset(fixture: CompetitionFixture) -> Path:
    for candidate in _candidates(fixture):
        if candidate.is_dir() and any(candidate.rglob("*.csv")):
            return candidate
    pytest.skip(
        f"no full dataset for {fixture.slug!r}; looked in "
        + ", ".join(str(c) for c in _candidates(fixture))
        + f". Set {_ENV_ROOT} to a directory holding <slug>/ per competition."
    )


@pytest.mark.slow
@pytest.mark.parametrize("slug", corpus_slugs())
def test_the_fixture_agrees_with_the_dataset_it_stands_for(slug: str, tmp_path: Path) -> None:
    """Every criterion the fixture *claims* must read the same on real data.

    `unverifiable` and `not_applicable` are the fixture saying it cannot speak to
    a criterion, so they are exactly the ones it must not be held to — holding
    them would be asserting that a truncation preserved what it declared it
    destroyed.
    """
    fixture = load_fixture(CORPUS / slug)
    data = _full_dataset(fixture)

    hermetic = profile_and_score(CORPUS / slug, tmp_path / "expanded")
    # Through the module, so the injection test below can replace it — a
    # direct reference would bind at import and the patch would do nothing,
    # which is how a test comes to assert something it never exercised.
    full = score_module.score_full_dataset(fixture, data, declared_from=tmp_path / "expanded")

    differing = disagreements(hermetic, full)

    assert not differing, (
        f"{slug}: the fixture and the real dataset disagree — "
        + "; ".join(
            f"{criterion}: fixture={was!r} dataset={now!r}"
            for criterion, (was, now) in sorted(differing.items())
        )
        + ". Move the criterion to `unverifiable`; the capture mode is wrong for it."
    )


@pytest.mark.slow
def test_a_disagreement_actually_fails_this_test(tmp_path: Path) -> None:
    """The check has to be able to say no.

    All five fixtures agree, which is the answer you want and no evidence at all
    that the assertion works. This injects a disagreement into the full run and
    asserts the failure names the criterion and the remedy, so a green tier 3
    means the verdicts matched rather than that nothing was compared.
    """
    from unittest import mock

    import labpilot.accessor.benchmark.score as score_module

    slug = corpus_slugs()[0]
    fixture = load_fixture(CORPUS / slug)
    _full_dataset(fixture)

    def _mangled(*args, **kwargs):
        card = score_full_dataset(*args, **kwargs)
        for result in card.results:
            if result.verdict == "pass":
                result.verdict = "fail"
                break
        return card

    with mock.patch.object(score_module, "score_full_dataset", _mangled):
        with pytest.raises(AssertionError) as raised:
            test_the_fixture_agrees_with_the_dataset_it_stands_for(slug, tmp_path)

    assert "disagree" in str(raised.value)
    assert "`unverifiable`" in str(raised.value), "the failure must name the remedy"


@pytest.mark.slow
@pytest.mark.parametrize("slug", corpus_slugs())
def test_the_fixture_claims_something_about_the_dataset(slug: str, tmp_path: Path) -> None:
    """Agreement over nothing is not agreement.

    A fixture whose every criterion is `unverifiable` would pass the check above
    by claiming nothing at all, which is exactly how a corpus rots into a suite
    that cannot fail.
    """
    fixture = load_fixture(CORPUS / slug)
    _full_dataset(fixture)

    hermetic = profile_and_score(CORPUS / slug, tmp_path / "expanded")
    claimed = [r.criterion for r in hermetic.results if r.verdict in ("pass", "fail")]

    assert claimed, f"{slug} asserts nothing about the dataset it stands for"
