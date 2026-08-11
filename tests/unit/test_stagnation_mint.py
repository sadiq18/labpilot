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

from labpilot.research_engine.conductor.budgets import BudgetConfig, BudgetState, ScoreEvent
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
