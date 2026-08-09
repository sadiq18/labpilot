"""The delta path's failure rate, read from provenance.

M19 §2 measures whether `codegen.strategy: delta` works often enough to become
the default, and §3 flips the default *when the rate justifies it*. A rate
nobody can recompute is not evidence, so the reading lives here rather than in
a one-off query: the number in the design doc and the number a reviewer gets
have to come from the same place.

Every aider run is recorded by `AiderAgent.propose`, success and failure alike,
with `failure_kind` naming the cause. That distinction is the whole point — a
*redundant hypothesis* is the selector choosing work already done, and counting
it as an adapter failure would conclude delta does not work from evidence that
it does.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

#: Failures that say nothing about whether the adapter can edit code.
#:
#: `hypothesis_redundant` — the parent already implements the change, so
#: declining is the correct answer and the hypothesis selector owns it.
#: `no_parent` — a baseline has nothing to diff against, which is the
#: whole-file agent's job by design.
#: `no_source` — nothing editable under the parent tree; a workspace shape,
#: not an edit the adapter got wrong.
EXCUSED_KINDS: frozenset[str] = frozenset(
    {"hypothesis_redundant", "no_parent", "no_source"}
)

#: Failures the adapter *is* answerable for. Declared rather than implied by
#: "everything else", so `test_every_raised_kind_is_classified` can hold both
#: lists against the kinds `AiderAgent` actually raises.
#:
#: They drifted immediately: `no_gateway` was excused here and is raised
#: nowhere — the constructor refuses a missing gateway outright — while
#: `no_source`, `aider_timeout`, `aider_missing` and `aider_failed` were raised
#: and classified by neither list. A rate computed from a stale vocabulary is
#: wrong in a way nothing surfaces.
COUNTED_KINDS: frozenset[str] = frozenset(
    {"aider_no_edit", "aider_failed", "aider_missing", "aider_timeout"}
)


@dataclass
class DeltaRate:
    """How often the delta path produced a usable proposal."""

    attempts: int = 0
    succeeded: int = 0
    #: Failures that count against the adapter.
    failed: int = 0
    #: Failures excused by `EXCUSED_KINDS`, kept visible rather than dropped.
    excused: int = 0
    #: Failures whose kind is in neither `EXCUSED_KINDS` nor `COUNTED_KINDS`.
    #: A subset of `failed`, not a third bucket: a failure is not excused by
    #: being unrecognised, and understating the rate is the dangerous
    #: direction for a decision about flipping the default. Counted separately
    #: so a drifted vocabulary is visible in the report itself.
    unclassified: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[int] = field(default_factory=list)

    @property
    def judged(self) -> int:
        """Attempts the adapter is actually answerable for."""
        return self.succeeded + self.failed

    @property
    def failure_rate(self) -> float | None:
        """Failures over judged attempts, or None when nothing was judged.

        `None`, not `0.0`. A rate of zero from zero attempts reads as a perfect
        record and would justify flipping the default on no evidence at all —
        the vacuous-measurement failure this project has already paid for once.
        """
        if self.judged == 0:
            return None
        return self.failed / self.judged

    @property
    def median_latency_ms(self) -> int | None:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        return ordered[len(ordered) // 2]


def _rows(knowledge_dir: Path, competition: str, agent: str) -> Iterable[dict]:
    from labpilot.accessor.sqlite import SqliteClient
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    paths = ResearchPaths(Path(knowledge_dir), competition)
    if not paths.db_path.is_file():
        # `SqliteClient.__init__` unconditionally mkdirs, connects and migrates,
        # so constructing one *creates* the database. `research conduct status`
        # calls this for every competition, and a status query that leaves a
        # `knowledge.db` behind for a competition that never had one is a write
        # dressed as a read. `agent_provenance._rows` already guards this way.
        # Reported on PR #118.
        return []
    client = SqliteClient(paths.db_path, allow_cross_thread=True)
    try:
        return [
            dict(row)
            for row in client.conn.execute(
                """
                SELECT failure_kind, failure_reason, latency_ms, created_at
                FROM agent_invocations
                WHERE agent = ? AND competition_slug = ?
                ORDER BY id
                """,
                (agent, competition),
            ).fetchall()
        ]
    finally:
        client.close()


def delta_rate(
    knowledge_dir: Path,
    competition: str,
    *,
    agent: str = "aider",
    since: str = "",
) -> DeltaRate:
    """Count aider outcomes for one competition.

    `since` is an ISO timestamp: the rate that decides §3 is the rate *after*
    the fixes being judged, and mixing in runs from before them would measure a
    version of the system that no longer exists.
    """
    rate = DeltaRate()
    for row in _rows(Path(knowledge_dir), competition, agent):
        if since and str(row.get("created_at") or "") < since:
            continue
        rate.attempts += 1
        kind = str(row.get("failure_kind") or "").strip()
        reason = str(row.get("failure_reason") or "").strip()
        # Either field marks a failure. Reading success off a blank
        # `failure_reason` alone meant a row carrying a `failure_kind` with no
        # message counted as a success — unreachable today, since the one
        # writer sets both, but nothing enforces that and the error would be
        # invisible: the rate simply reads better. Reported on PR #118.
        if not reason and not kind:
            rate.succeeded += 1
            latency = row.get("latency_ms")
            if isinstance(latency, int):
                rate.latencies_ms.append(latency)
            continue
        label = kind or "unclassified"
        rate.by_kind[label] = rate.by_kind.get(label, 0) + 1
        if label in EXCUSED_KINDS:
            rate.excused += 1
            continue
        rate.failed += 1
        if label not in COUNTED_KINDS:
            # Still counted against the adapter — a failure is not excused by
            # being unrecognised. Flagged, though: "everything not excused is a
            # failure" left `COUNTED_KINDS` decorative, so a kind added to
            # `AiderAgent` and to neither list changed the rate with nothing to
            # show for it. Reported on PR #118.
            rate.unclassified += 1
    return rate
