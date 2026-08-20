"""Tier 1 — every captured competition, scored on the shipped path.

M24. `tests/integration/` has held only stale `.pyc` since `109745c`; this is
what goes in it. The suite is hermetic: fixtures carry header sets and a
listing, the expander rebuilds the shape, and the profiler that runs is the one
the product runs.

**What tier 1 does not claim.** Row counts, cardinality, distributions and
generic-beats-dummy need the real bytes and belong to tier 2. Asserting them
here would be asserting a truncation artifact — which is why each fixture names
what its capture destroyed and the scorer refuses to score those.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from labpilot.accessor.benchmark.fixture import load_fixture
from labpilot.accessor.benchmark.score import CRITERIA, profile_and_score

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "competitions"


def corpus_slugs() -> list[str]:
    return sorted(path.name for path in CORPUS.iterdir() if (path / "fixture.json").is_file())


@pytest.mark.parametrize("slug", corpus_slugs())
def test_a_captured_competition_is_understood(slug: str, tmp_path: Path) -> None:
    """No criterion fails, and at least one passes.

    `fail` is the only verdict that counts against a fixture. `unverifiable` and
    `not_applicable` are answers about the *capture* and the *competition*, and
    a fixture whose every criterion were one of those would pass by describing
    nothing — hence the second half.
    """
    fixture = load_fixture(CORPUS / slug)
    card = profile_and_score(CORPUS / slug, tmp_path)
    failures = [
        f"{result.criterion}: expected {result.expected}, observed {result.observed}"
        for result in card.results
        if result.verdict == "fail"
    ]

    assert not failures, f"{slug}\n" + "\n".join(failures)
    assert any(result.verdict == "pass" for result in card.results), "nothing was scored"
    # Two properties, not one derived from the other. Tying `understood` to the
    # presence of a declared defect made a merely-unscoreable fixture fail a
    # test about correctness.
    reds = [r.criterion for r in card.results if r.verdict == "known_failure"]
    assert sorted(reds) == sorted(fixture.known_failures), (
        f"{slug}: the fixture and the scorecard disagree about what ships red"
    )
    assert card.understood == (not reds)


def test_the_corpus_scores_every_criterion(tmp_path: Path) -> None:
    """Every criterion is answered for every fixture, with no silent omissions.

    A scorer that skipped a criterion would report a corpus cleaner than it is.
    """
    for slug in corpus_slugs():
        card = profile_and_score(CORPUS / slug, tmp_path / slug)
        assert [result.criterion for result in card.results] == list(CRITERIA), slug


def test_the_hardest_fixture_asks_rather_than_guesses(tmp_path: Path) -> None:
    """rogii's submission names an `id` column that exists in no table.

    The expectation is not an answer. A corpus that demanded one here would be
    rewarding the guess that cost this workspace eleven days, so the fixture
    expects a *question* and the criterion scores whether it was asked.
    """
    slug = "rogii-wellbore-geology-prediction"
    fixture = load_fixture(CORPUS / slug)

    card = profile_and_score(CORPUS / slug, tmp_path)

    assert fixture.expected.must_ask == ["id_columns"]
    assert card.verdict_for("abstention") == "pass"
    assert card.verdict_for("target_column") == "pass", "and the target is still resolved"


def test_a_known_failure_is_red_on_purpose(tmp_path: Path) -> None:
    """The metric-synonym fixture, and the day it goes green.

    `balanced_accuracy_score` is recorded in the captured spec with
    `key: accuracy`. The fixture declares it, so the corpus reports
    `known_failure` rather than `fail` — and this test fails the moment it
    starts passing, which is how a fixed defect announces itself instead of
    quietly turning a red cell green.
    """
    slug = "playground-series-s6e7"

    card = profile_and_score(CORPUS / slug, tmp_path)

    assert card.verdict_for("metric_name") == "known_failure", (
        "metric_name now passes — remove it from the fixture's known_failures"
    )


# --- the corpus says what it is ---------------------------------------------


@pytest.mark.parametrize("slug", corpus_slugs())
def test_every_captured_file_is_accounted_for(slug: str) -> None:
    """The manifest and the directory agree, in both directions.

    A file on disk that the manifest does not name has no provenance, and a
    manifest entry with no file is a claim about something that is not there.
    """
    directory = CORPUS / slug
    fixture = load_fixture(directory)
    on_disk = {
        str(path.relative_to(directory / "data"))
        for path in (directory / "data").rglob("*")
        if path.is_file()
    }
    declared = {entry.path for entry in fixture.files}

    assert declared == on_disk, f"{slug}: manifest and data/ disagree"
    assert fixture.files, "a fixture with no files describes nothing"


@pytest.mark.parametrize("slug", corpus_slugs())
def test_a_capture_says_what_it_kept(slug: str) -> None:
    """`fixture_rows <= source_rows`, and a headers-only file kept none.

    The number that matters is the *source* row count: it is what a re-download
    is checked against, and `playground-series-s6e7` is in this corpus because
    690,088 rows were once reported as 100,000.
    """
    fixture = load_fixture(CORPUS / slug)

    for entry in fixture.files:
        if entry.source_rows is None:
            assert entry.fixture_rows is None, f"{entry.path}: rows on a non-tabular file"
            continue
        assert entry.fixture_rows is not None
        assert entry.fixture_rows <= entry.source_rows, entry.path
        if entry.mode == "headers_only":
            assert entry.fixture_rows == 0, entry.path
        assert len(entry.source_sha256) == 64, entry.path


@pytest.mark.parametrize("slug", corpus_slugs())
def test_a_truncated_fixture_declares_what_it_cannot_prove(slug: str) -> None:
    """Anything a capture destroyed is named, with a reason.

    Silence here is the failure mode: the scorer would report `fail` on a
    criterion the fixture was never able to speak to, and the number would be
    measuring the truncation.
    """
    fixture = load_fixture(CORPUS / slug)

    if fixture.provenance == "verbatim":
        return
    assert fixture.unverifiable, f"{slug}: a derived capture that claims to prove everything"
    for criterion, reason in fixture.unverifiable.items():
        assert reason.strip(), f"{slug}: {criterion} unverifiable with no reason"


@pytest.mark.parametrize("slug", corpus_slugs())
def test_a_fixture_honours_the_licence_it_declares(slug: str) -> None:
    """`redistribution: forbidden` is a constraint on the fixture, not a note.

    Both fixtures in this corpus carry column names and no data rows, which is
    why headers-only is the default. Left unchecked, the field would be a claim
    the commit containing it contradicts — and a provenance record that
    contradicts itself is not one.
    """
    fixture = load_fixture(CORPUS / slug)

    assert fixture.honours_its_licence, (
        f"{slug}: redistribution is {fixture.redistribution} and the fixture carries rows"
    )


def test_the_manifest_names_every_fixture() -> None:
    """The prose and the directory do not drift.

    `real_failures/MANIFEST.md` exists because a fixture and its description
    disagreed for a day; the same check, one corpus over.
    """
    manifest = (CORPUS / "MANIFEST.md").read_text(encoding="utf-8")

    for slug in corpus_slugs():
        assert slug in manifest, f"{slug} is in the corpus and not in MANIFEST.md"


@pytest.mark.parametrize("slug", corpus_slugs())
def test_expansion_is_deterministic(slug: str, tmp_path: Path) -> None:
    """Two expansions of one fixture are the same tree.

    The corpus is only allowed to stand in for real data if it is the same data
    every time it is asked.
    """
    from labpilot.accessor.benchmark.expand import expand_fixture

    def digest(root: Path) -> str:
        payload = sorted(
            (str(p.relative_to(root)), p.read_bytes()) for p in root.rglob("*") if p.is_file()
        )
        return hashlib.sha256(
            json.dumps([(n, b.decode(errors="replace")) for n, b in payload]).encode()
        ).hexdigest()

    expand_fixture(CORPUS / slug, tmp_path / "one")
    expand_fixture(CORPUS / slug, tmp_path / "two")

    assert digest(tmp_path / "one") == digest(tmp_path / "two")
