"""M8-6: a run of unimproving experiments becomes a hypothesis that names a change.

`maybe_mint_improvement_hypothesis` already reacts when one execution loses to
its parent. The gap this closes is the campaign where every experiment looks
acceptable alone and the score still has not moved.

Exit criterion 1 for the multi-experiment case: the minted hypothesis must cite
*every* experiment in the window by id, not just the newest, so a reader
checking "did this cite its evidence" can resolve all of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    ScoreEvent,
    score_summary,
)
from labpilot.research_engine.conductor.stagnation import (
    mint_stagnation_hypothesis,
    stagnation_window,
    techniques_in,
)
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str = "demo") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _event(eid: str, value: float, technique: str | None = None, combo: list[str] | None = None):
    return ScoreEvent(
        experiment_id=eid,
        metric_name="cv_rmse",
        value=value,
        maximize=False,
        technique=technique,
        combo_techniques=combo or [],
    )


def _stagnant(*, techniques: list[str | None] | None = None) -> BudgetState:
    """Four experiments, each worse than the first — steps_since_improvement 3."""
    names = techniques or ["target_encoding", "mixup", "feature_interactions", "smoothing"]
    values = [194.8, 195.0, 196.0, 197.0]
    return BudgetState(
        score_events=[
            _event(f"E-{i:03d}", value, technique=name)
            for i, (value, name) in enumerate(zip(values, names, strict=True), start=1)
        ]
    )


def _with_vocabulary(ws: Workspace, *names: str, status: str = "candidate") -> Workspace:
    """Techniques the planner may propose.

    The ledger reads `list_techniques()` unfiltered, and its own
    `TechniqueRecord.status` is a different axis (worked/failed/untried), so a
    proposal is only safe if the M18 vocabulary status is checked too.
    """
    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore

    with KnowledgeStore(ws.knowledge_dir, ws.competition) as store:
        for name in names:
            store.merge_technique(name)
            store._conn.execute(  # noqa: SLF001 — setting the status is the point
                "UPDATE techniques SET status = ? WHERE lower(name) = lower(?)", (status, name)
            )
        store._conn.commit()  # noqa: SLF001
    return ws


# --- the window -----------------------------------------------------------


def test_a_campaign_below_the_window_has_no_stagnation_window():
    """Three experiments since the last record is not yet `plateau_window`."""
    state = BudgetState(
        score_events=[_event("E-001", 194.8), _event("E-002", 195.0), _event("E-003", 196.0)]
    )

    assert stagnation_window(state, BudgetConfig(plateau_window=3)) == []


def test_the_window_is_the_experiments_since_the_last_improvement():
    """Not the last N events — the ones that actually failed to improve. The
    experiment that set the record is not part of what went wrong."""
    window = stagnation_window(_stagnant(), BudgetConfig(plateau_window=3))

    assert [e.experiment_id for e in window] == ["E-002", "E-003", "E-004"]


def test_a_recovering_campaign_has_no_window():
    state = BudgetState(
        score_events=[
            _event("E-001", 197.0),
            _event("E-002", 196.0),
            _event("E-003", 195.0),
            _event("E-004", 194.0),
        ]
    )

    assert stagnation_window(state, BudgetConfig(plateau_window=3)) == []


# --- technique accounting -------------------------------------------------


def test_combo_members_count_as_spent():
    """A technique tried as half of a combination has been tried. Proposing it
    alone as though untested would re-run work the window already covers."""
    events = [
        _event("E-001", 1.0, technique="target_encoding"),
        _event("E-002", 2.0, combo=["mixup", "cutout"]),
    ]

    assert techniques_in(events) == ["target_encoding", "mixup", "cutout"]


def test_record_references_are_not_techniques():
    """`hyp:H-010` travels in these fields as provenance. It was the most
    common "technique" in the knowledge base before `shared.labels` existed."""
    events = [_event("E-001", 1.0, technique="hyp:H-010", combo=["fork:H-003", "mixup"])]

    assert techniques_in(events) == ["mixup"]


# --- the mint -------------------------------------------------------------


def test_the_mint_cites_every_experiment_in_the_window(tmp_path: Path):
    """Exit criterion 1 for the multi-experiment case."""
    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart")

    minted_id = mint_stagnation_hypothesis(ws, _stagnant(), BudgetConfig(plateau_window=3))

    assert minted_id is not None
    hypothesis = HypothesisStore(ws.knowledge_dir, ws.competition).get(minted_id)
    cited = {ref.ref for ref in hypothesis.evidence}
    assert cited == {"E-002", "E-003", "E-004"}
    for eid in ("E-002", "E-003", "E-004"):
        assert eid in hypothesis.reason
    assert hypothesis.status is HypothesisStatus.PROPOSED
    assert hypothesis.technique == "gradient_boosting_dart"


def test_a_long_window_still_names_every_shown_id_whole(tmp_path: Path):
    """`reason` used to be built by joining every citation and then hard-
    truncating the whole string to a length cap. For a window long enough to
    push the joined string past that cap, the cut landed mid-citation and
    silently dropped whichever ids came after it -- the prose no longer
    matched the exit criterion that a reader can resolve every cited id.

    Citing by count instead of by character keeps every printed id whole;
    the tail is named by number, and `evidence` (never truncated) is still
    where all of them are actually resolvable.
    """
    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart")
    # 40 events, each cited as "E-NNN (tNN)" -- long enough to have blown past
    # the old 1000-char cap mid-citation. Monotonically worsening, so the
    # window is every event after the first (the best-so-far): E-001..E-039.
    events = [_event(f"E-{i:03d}", 200.0 + i, technique=f"t{i}") for i in range(40)]
    state = BudgetState(score_events=events)

    minted_id = mint_stagnation_hypothesis(ws, state, BudgetConfig(plateau_window=3))

    assert minted_id is not None
    hypothesis = HypothesisStore(ws.knowledge_dir, ws.competition).get(minted_id)
    # Every experiment is still resolvable structurally, never truncated.
    assert {ref.ref for ref in hypothesis.evidence} == {e.experiment_id for e in events[1:]}
    # The first 12 of the window are cited whole and intact in the prose.
    for i in range(1, 13):
        assert f"E-{i:03d} (t{i})" in hypothesis.reason
    # The rest are named by count, not silently dropped mid-string.
    assert "and 27 more (see evidence)" in hypothesis.reason
    assert "E-039" not in hypothesis.reason


def test_long_combo_citations_do_not_overflow_the_prose_caps(tmp_path: Path):
    """The count cap alone (limit=12) isn't enough: a dozen two-technique
    combo citations run long enough to blow past `observation`'s 500-char
    cap on their own, and that hard slice used to land mid-word -- reproduced
    live with exactly this fixture before `_cite_list` grew a character
    budget too. `reason` (cap 1000) had the same exposure, just at a higher
    citation count.
    """
    ws = _with_vocabulary(_ws(tmp_path), "winner")
    events = [
        _event(
            f"E-{i:03d}",
            200.0 + i,
            combo=["mixup_augmentation", "cutout_regularization"],
        )
        for i in range(14)
    ]
    state = BudgetState(score_events=events)

    minted_id = mint_stagnation_hypothesis(ws, state, BudgetConfig(plateau_window=3))

    assert minted_id is not None
    hypothesis = HypothesisStore(ws.knowledge_dir, ws.competition).get(minted_id)
    assert len(hypothesis.observation) <= 500
    assert len(hypothesis.reason) <= 1000
    # No citation appears cut mid-word: a cut mid-citation leaves an opening
    # "(" with no matching ")", which balanced parens rules out directly.
    assert hypothesis.observation.count("(") == hypothesis.observation.count(")")
    assert hypothesis.reason.count("(") == hypothesis.reason.count(")")
    assert "more (see evidence)" in hypothesis.observation


def test_a_long_metric_name_does_not_overflow_the_prose_caps_either(tmp_path: Path):
    """The character budget round 4 gave `_cite_list` only bounded the
    citation portion -- the wrapper text ("N experiments since {metric} last
    improved: ...") sits outside it, and `metric` has no length bound.
    Reproduced live: 20 combo events with a 120-char metric name still cut
    `observation` mid-word at its 500-char cap, even with the round-4 fix in
    place -- `_cite_list`'s fixed 400-char budget left no room for a wrapper
    that size. The budget passed to `_cite_list` must be sized to what's left
    after each template's own wrapper length, not a fixed number.
    """
    ws = _with_vocabulary(_ws(tmp_path), "winner")
    long_metric = "x" * 120
    events = [
        ScoreEvent(
            experiment_id=f"E-{i:03d}",
            metric_name=long_metric,
            value=200.0 + i,
            maximize=False,
            combo_techniques=["mixup_augmentation", "cutout_regularization"],
        )
        for i in range(20)
    ]
    state = BudgetState(score_events=events)

    minted_id = mint_stagnation_hypothesis(ws, state, BudgetConfig(plateau_window=3))

    assert minted_id is not None
    hypothesis = HypothesisStore(ws.knowledge_dir, ws.competition).get(minted_id)
    assert len(hypothesis.observation) <= 500
    assert len(hypothesis.reason) <= 1000
    assert hypothesis.observation.count("(") == hypothesis.observation.count(")")
    assert hypothesis.reason.count("(") == hypothesis.reason.count(")")


def test_cite_list_output_never_exceeds_its_max_chars_budget():
    """The invariant that actually matters, exercised directly (not through
    the full mint pipeline, so it pressures the exact boundary instead of
    relying on luck in a fixture's citation lengths): whatever `_cite_list`
    returns, including the trailing "and N more" note, must fit within
    `max_chars` -- for any `max_chars` a caller might size from its own
    template's remaining room, not just the function's own default.
    """
    from labpilot.research_engine.conductor.stagnation import _cite_list

    events = [
        _event(f"E-{i:03d}", 200.0 + i, technique="a_moderately_named_technique") for i in range(30)
    ]
    for max_chars in (100, 200, 400):
        result = _cite_list(events, max_chars=max_chars)
        assert len(result) <= max_chars, f"max_chars={max_chars} gave {len(result)}: {result!r}"


def test_a_passed_through_window_and_summary_match_the_recomputed_ones(tmp_path: Path):
    """`window`/`summary` are optional so direct callers (this file, mostly)
    keep working unchanged -- but nothing had ever exercised the path a real
    caller actually takes: passing both in, pre-computed. Proves the two
    entry points agree rather than merely coexisting untested.
    """
    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart")
    state = _stagnant()
    config = BudgetConfig(plateau_window=3)

    recomputed_id = mint_stagnation_hypothesis(ws, state, config)

    ws2 = _with_vocabulary(_ws(tmp_path, slug="demo2"), "gradient_boosting_dart")
    state2 = _stagnant()
    summary = score_summary(state2, config)
    window = stagnation_window(state2, config, summary=summary)
    threaded_id = mint_stagnation_hypothesis(ws2, state2, config, window=window, summary=summary)

    a = HypothesisStore(ws.knowledge_dir, ws.competition).get(recomputed_id)
    b = HypothesisStore(ws2.knowledge_dir, ws2.competition).get(threaded_id)
    assert a.observation == b.observation
    assert a.reason == b.reason
    assert a.technique == b.technique
    assert {r.ref for r in a.evidence} == {r.ref for r in b.evidence}


def test_the_proposal_avoids_everything_the_window_spent(tmp_path: Path):
    """Re-proposing a technique the stagnant window already ran is the one
    thing this must not do.

    `mixup` is the *only* thing in the vocabulary, so the exclusion is the
    sole reason nothing is proposed. An earlier version of this test offered a
    second technique alongside it and passed whether or not the exclusion
    existed — `techniques_untried` is sorted, so the acceptable one came first
    and the filter was never reached.
    """
    ws = _with_vocabulary(_ws(tmp_path), "mixup")

    minted_id = mint_stagnation_hypothesis(
        ws, _stagnant(techniques=["a", "mixup", "c", "d"]), BudgetConfig(plateau_window=3)
    )

    assert minted_id is None
    assert HypothesisStore(ws.knowledge_dir, ws.competition).list() == []


def test_the_proposal_is_one_the_window_did_not_use(tmp_path: Path):
    """The other half: with an untried technique available it is chosen, so
    the exclusion does not simply suppress every proposal."""
    ws = _with_vocabulary(_ws(tmp_path), "mixup", "gradient_boosting_dart")

    minted_id = mint_stagnation_hypothesis(
        ws, _stagnant(techniques=["a", "mixup", "c", "d"]), BudgetConfig(plateau_window=3)
    )

    hypothesis = HypothesisStore(ws.knowledge_dir, ws.competition).get(minted_id)
    assert hypothesis.technique == "gradient_boosting_dart"


@pytest.mark.parametrize("status", ["dormant", "rejected"])
def test_a_technique_the_vocabulary_hides_is_never_proposed(tmp_path: Path, status):
    """M18's whole point: a live campaign asked the engineer to implement a
    technique called `the`. The ledger does not filter on vocabulary status,
    so this has to."""
    ws = _with_vocabulary(_ws(tmp_path), "the", status=status)

    assert mint_stagnation_hypothesis(ws, _stagnant(), BudgetConfig(plateau_window=3)) is None


def test_no_untried_technique_means_no_hypothesis(tmp_path: Path):
    """Naming nothing is honest when the inventory is exhausted. Inventing a
    technique would put a fabricated cause on the record."""
    ws = _ws(tmp_path)

    assert mint_stagnation_hypothesis(ws, _stagnant(), BudgetConfig(plateau_window=3)) is None
    assert HypothesisStore(ws.knowledge_dir, ws.competition).list() == []


def test_a_later_plateau_does_not_re_propose_what_an_earlier_one_queued(tmp_path: Path):
    """A PROPOSED hypothesis leaves its technique untried in the ledger — that
    axis is derived from CONFIRMED/REJECTED ones only. So without subtracting
    the open backlog, every plateau in a campaign proposes the same name again
    and grows the pile of stale proposed rows M21 exists because of."""
    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart")
    config = BudgetConfig(plateau_window=3)
    state = _stagnant()
    store = HypothesisStore(ws.knowledge_dir, ws.competition)

    first = mint_stagnation_hypothesis(ws, state, config)
    assert store.get(first).technique == "gradient_boosting_dart"
    assert store.get(first).status == HypothesisStatus.PROPOSED

    # A record, then a second plateau — stagnant again, and nobody has run the
    # proposal yet. The only thing that should stop a mint here is the backlog.
    state.score_events.extend(
        [
            _event("E-005", 100.0, "win"),
            _event("E-006", 101.0, "swa"),
            _event("E-007", 102.0, "cutmix"),
            _event("E-008", 103.0, "ema"),
        ]
    )
    assert len(stagnation_window(state, config)) == 3

    assert mint_stagnation_hypothesis(ws, state, config) is None
    assert [h.id for h in store.list()] == [first]


def test_a_campaign_that_is_not_stagnant_mints_nothing(tmp_path: Path):
    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart")
    improving = BudgetState(
        score_events=[_event("E-001", 197.0), _event("E-002", 196.0), _event("E-003", 195.0)]
    )

    assert mint_stagnation_hypothesis(ws, improving, BudgetConfig(plateau_window=3)) is None


# --- the edge trigger, through the loop hook ------------------------------


def _mint_hook(ws: Workspace, state: BudgetState, config: BudgetConfig) -> None:
    from labpilot.research_engine.conductor.loop import _maybe_mint_on_stagnation

    _maybe_mint_on_stagnation(ws, state, config)


def test_a_long_plateau_mints_once_not_every_step(tmp_path: Path):
    """`steps_since_improvement` only grows while a campaign is stuck, so a
    level-triggered mint would add a near-duplicate hypothesis on every
    remaining step. Content dedup does not prevent that; the latch does."""
    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart", "swa", "cutmix")
    state = _stagnant()
    config = BudgetConfig(plateau_window=3)
    store = HypothesisStore(ws.knowledge_dir, ws.competition)

    _mint_hook(ws, state, config)
    assert state.stagnation_mint_fired is True
    after_first = len(store.list())

    # The plateau continues: three more experiments, still no improvement.
    for i in range(5, 8):
        state.score_events.append(_event(f"E-{i:03d}", 197.0 + i, technique=f"t{i}"))
        _mint_hook(ws, state, config)

    assert len(store.list()) == after_first
    assert after_first == 1


def test_an_improvement_rearms_the_latch(tmp_path: Path):
    """A second plateau later in the same campaign must mint again rather than
    staying suppressed for good."""
    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart", "swa", "cutmix", "ema")
    state = _stagnant()
    config = BudgetConfig(plateau_window=3)
    store = HypothesisStore(ws.knowledge_dir, ws.competition)

    _mint_hook(ws, state, config)
    assert len(store.list()) == 1

    # A record, which clears the latch...
    state.score_events.append(_event("E-005", 100.0, technique="breakthrough"))
    _mint_hook(ws, state, config)
    assert state.stagnation_mint_fired is False

    # ...then a fresh plateau on top of it.
    for i in range(6, 10):
        state.score_events.append(_event(f"E-{i:03d}", 100.0 + i, technique=f"t{i}"))
        _mint_hook(ws, state, config)

    assert len(store.list()) == 2


def test_a_broken_mint_does_not_escape_the_hook(tmp_path: Path, monkeypatch):
    """`_maybe_mint_on_stagnation` runs inside `_record_experiment_outcome`,
    which runs inside the dispatch loop's outer try/except -- an escape here
    would not stay local, it would land as a dispatch error and re-record the
    just-succeeded experiment as failed. A broken mint must not do that."""
    import labpilot.research_engine.conductor.loop as loop_mod

    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart")
    state = _stagnant()
    config = BudgetConfig(plateau_window=3)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("ledger is unreadable")

    monkeypatch.setattr(loop_mod, "mint_stagnation_hypothesis", _boom)

    loop_mod._maybe_mint_on_stagnation(ws, state, config)  # noqa: SLF001 — testing the guard

    assert state.stagnation_mint_fired is False
    assert HypothesisStore(ws.knowledge_dir, ws.competition).list() == []


def test_a_plateau_that_had_nothing_to_propose_retries_when_something_appears(tmp_path: Path):
    """M8-5 opens `analyze_competition` *because* the campaign is stagnant, so
    the vocabulary can grow during the very plateau this reacts to.

    Latching on the attempt rather than the result would mean the technique
    the stagnation signal went and found could never be proposed for the
    plateau that prompted the search.
    """
    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore

    ws = _ws(tmp_path)  # vocabulary starts empty
    state = _stagnant()
    config = BudgetConfig(plateau_window=3)
    store = HypothesisStore(ws.knowledge_dir, ws.competition)

    _mint_hook(ws, state, config)
    assert store.list() == []
    assert state.stagnation_mint_fired is False, "an attempt that minted nothing must not latch"

    # analyze_competition runs and brings back something untried.
    with KnowledgeStore(ws.knowledge_dir, ws.competition) as knowledge:
        knowledge.merge_technique("gradient_boosting_dart")

    state.score_events.append(_event("E-005", 198.0, technique="another"))
    _mint_hook(ws, state, config)

    assert len(store.list()) == 1
    assert state.stagnation_mint_fired is True


def test_the_hook_does_nothing_while_the_campaign_improves(tmp_path: Path):
    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart")
    state = BudgetState(
        score_events=[_event("E-001", 197.0), _event("E-002", 196.0), _event("E-003", 195.0)]
    )

    _mint_hook(ws, state, BudgetConfig(plateau_window=3))

    assert state.stagnation_mint_fired is False
    assert HypothesisStore(ws.knowledge_dir, ws.competition).list() == []


def test_a_combo_in_the_window_is_cited_whole(tmp_path: Path):
    """Picking one member to blame for a delta produced by two together is the
    misattribution M19 §5 fixed for evidence cards."""
    ws = _with_vocabulary(_ws(tmp_path), "gradient_boosting_dart")
    state = BudgetState(
        score_events=[
            _event("E-001", 194.8, technique="target_encoding"),
            _event("E-002", 195.0, combo=["mixup", "cutout"]),
            _event("E-003", 196.0, technique="smoothing"),
            _event("E-004", 197.0, technique="scaling"),
        ]
    )

    minted_id = mint_stagnation_hypothesis(ws, state, BudgetConfig(plateau_window=3))

    reason = HypothesisStore(ws.knowledge_dir, ws.competition).get(minted_id).reason
    assert "mixup + cutout" in reason
