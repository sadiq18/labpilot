"""The rate §3 decides on has to be recomputable, and honest when it is empty."""

from __future__ import annotations

from labpilot.research_engine.telemetry.delta_rate import DeltaRate, delta_rate

_COMP = "demo"


def _record(
    tmp_path,
    *,
    failure_reason=None,
    failure_kind=None,
    created_at="2026-08-09T01:00:00+00:00",
    agent="aider",
):
    """Insert a row directly.

    `record_invocation` takes neither the competition nor a timestamp — the
    sink supplies both — and this is a test of the *query*, so the rows are
    written the way the sink writes them.
    """
    from labpilot.accessor.sqlite import SqliteClient
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    paths = ResearchPaths(tmp_path, _COMP).ensure()
    client = SqliteClient(paths.db_path, allow_cross_thread=True)
    try:
        client.conn.execute(
            """
            INSERT INTO agent_invocations
                (competition_slug, agent, llm_role, generated_by,
                 failure_reason, failure_kind, latency_ms, created_at)
            VALUES (?, ?, 'codegen', 'aider', ?, ?, 1000, ?)
            """,
            (_COMP, agent, failure_reason, failure_kind, created_at),
        )
        client.conn.commit()
    finally:
        client.close()


def test_an_empty_history_has_no_rate_rather_than_a_perfect_one():
    """Zero from zero reads as a flawless record and would justify flipping the
    default on no evidence — the vacuous measurement this project already paid
    for once."""
    assert DeltaRate().failure_rate is None


def test_a_success_and_a_failure_make_one_half(tmp_path):
    _record(tmp_path)
    _record(tmp_path, failure_reason="boom", failure_kind="aider_no_edit")

    rate = delta_rate(tmp_path, _COMP)

    assert (rate.succeeded, rate.failed) == (1, 1)
    assert rate.failure_rate == 0.5


def test_a_redundant_hypothesis_is_not_an_adapter_failure(tmp_path):
    """Declining work the parent already does is the correct answer. Counting
    it here would conclude delta does not work from evidence that it does."""
    _record(tmp_path)
    _record(tmp_path, failure_reason="already implemented", failure_kind="hypothesis_redundant")

    rate = delta_rate(tmp_path, _COMP)

    assert rate.failure_rate == 0.0
    assert rate.excused == 1
    assert rate.by_kind["hypothesis_redundant"] == 1


def test_excused_failures_stay_visible(tmp_path):
    """Excused is not hidden — a run that is all redundancy is a finding about
    the selector, and dropping it would make that invisible."""
    _record(tmp_path, failure_reason="x", failure_kind="no_parent")

    rate = delta_rate(tmp_path, _COMP)

    assert rate.attempts == 1
    assert rate.judged == 0
    assert rate.failure_rate is None


def test_an_unclassified_failure_counts_against_the_adapter(tmp_path):
    """A failure with no kind is not excused by its own vagueness."""
    _record(tmp_path, failure_reason="something broke")

    rate = delta_rate(tmp_path, _COMP)

    assert rate.failed == 1
    assert rate.by_kind["unclassified"] == 1


def test_since_excludes_older_runs(tmp_path):
    """§3's decision is about the system as it is now, not as it was before the
    fixes being judged."""
    _record(
        tmp_path,
        failure_reason="old",
        failure_kind="aider_no_edit",
        created_at="2026-01-01T00:00:00+00:00",
    )
    _record(tmp_path, created_at="2026-08-09T00:00:00+00:00")

    rate = delta_rate(tmp_path, _COMP, since="2026-06-01T00:00:00+00:00")

    assert (rate.attempts, rate.succeeded, rate.failed) == (1, 1, 0)


def test_every_raised_kind_is_classified():
    """The two lists must cover what `AiderAgent` actually raises.

    Reported on PR #118 and true at the time: `no_gateway` was excused and is
    raised nowhere, while `no_source`, `aider_timeout`, `aider_missing` and
    `aider_failed` were raised and classified by neither list. A rate computed
    from a stale vocabulary is wrong in a way nothing surfaces — the number
    still prints.

    Read from the source rather than declared twice, so adding a kind fails
    here until someone decides which side it belongs on.
    """
    import re
    from pathlib import Path

    from labpilot.research_engine.telemetry.delta_rate import COUNTED_KINDS, EXCUSED_KINDS

    source = (
        Path(__file__).resolve().parents[2]
        / "src/labpilot/research_engine/execution/delta/aider_agent.py"
    ).read_text()
    raised = set(re.findall(r'kind="([a-z_]+)"', source))

    assert raised, "no kinds found — this guard would pass vacuously"
    unclassified = raised - EXCUSED_KINDS - COUNTED_KINDS
    assert not unclassified, f"classify these in delta_rate: {sorted(unclassified)}"
    phantom = (EXCUSED_KINDS | COUNTED_KINDS) - raised
    assert not phantom, f"classified but never raised: {sorted(phantom)}"


# --- PR #118 round 3 ---------------------------------------------------------


def test_reading_a_rate_does_not_create_a_database(tmp_path):
    """Reported on PR #118. `SqliteClient.__init__` mkdirs, connects and
    migrates, so *constructing* one writes. `research conduct status` calls this
    for every competition, and a status query that leaves a `knowledge.db`
    behind for a competition that never had one is a write dressed as a read."""
    knowledge = tmp_path / "never-existed"

    rate = delta_rate(knowledge, "never-touched")

    assert rate.attempts == 0
    assert not knowledge.exists()


def test_a_failure_kind_without_a_reason_is_not_a_success(tmp_path):
    """Reported on PR #118: success was read off a blank `failure_reason`
    alone, so a row carrying only a kind counted as a success and inflated the
    rate a default-flip is decided on."""
    _record(tmp_path, failure_reason="", failure_kind="aider_no_edit")

    rate = delta_rate(tmp_path, _COMP)

    assert rate.succeeded == 0
    assert rate.failed == 1


def test_an_unrecognised_kind_is_counted_and_flagged(tmp_path):
    """`COUNTED_KINDS` was decorative — `if excused else failed` never read it,
    so a kind added to `AiderAgent` and to neither list moved the rate with
    nothing to show for it. It still counts against the adapter, which is the
    safe direction, and now says it was not recognised."""
    _record(tmp_path, failure_reason="something new", failure_kind="aider_exploded")

    rate = delta_rate(tmp_path, _COMP)

    assert rate.failed == 1
    assert rate.unclassified == 1
    assert rate.by_kind["aider_exploded"] == 1


def test_a_recognised_failure_is_not_flagged_as_unclassified(tmp_path):
    _record(tmp_path, failure_reason="no edit", failure_kind="aider_no_edit")

    rate = delta_rate(tmp_path, _COMP)

    assert rate.failed == 1
    assert rate.unclassified == 0
