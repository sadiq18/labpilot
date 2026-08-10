from pathlib import Path

import pytest

from labpilot.research_engine.shared.experiments.graph import ExperimentGraph, assemble_experiment, build_graph
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore, linked_experiments
from labpilot.research_engine.shared.experiments.models import Experiment, HypothesisStatus


def test_create_allocates_incrementing_ids(tmp_path: Path):
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    first = store.create(
        observation="Rare classes underperform",
        reason="Imbalance",
        prediction="Focal loss helps",
        confidence=0.7,
        tags=["loss"],
    )
    second = store.create(
        observation="Cabin is noisy",
        reason="Missingness",
        prediction="HasCabin flag helps",
        confidence=0.6,
    )
    assert first.id == "H-001"
    assert second.id == "H-002"
    assert (tmp_path / "knowledge/titanic/hypotheses/H-001.json").is_file()
    assert (tmp_path / "knowledge/titanic/hypotheses/H-002.json").is_file()


def test_ensure_baseline_uses_reserved_id(tmp_path: Path):
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    baseline = store.ensure_baseline()
    assert baseline.id == "H-BASELINE"
    again = store.ensure_baseline()
    assert again.id == "H-BASELINE"
    improvement = store.create(
        observation="Try Mixup",
        reason="Regularization",
        prediction="CV improves",
        confidence=0.6,
    )
    assert improvement.id == "H-001"


def test_list_filters_by_status(tmp_path: Path):
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    store.create(
        observation="a", reason="b", prediction="c", confidence=0.5
    )
    h2 = store.create(
        observation="d", reason="e", prediction="f", confidence=0.5
    )
    store.update_status(h2.id, HypothesisStatus.TESTING)

    proposed = store.list(status=HypothesisStatus.PROPOSED)
    testing = store.list(status=HypothesisStatus.TESTING)
    assert len(proposed) == 1
    assert proposed[0].id == "H-001"
    assert len(testing) == 1
    assert testing[0].id == "H-002"


def test_update_status_routes_evidence(tmp_path: Path):
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    hypothesis = store.create(
        observation="a", reason="b", prediction="c", confidence=0.5
    )

    confirmed = store.update_status(
        hypothesis.id,
        HypothesisStatus.CONFIRMED,
        evidence_run_id="run-for",
    )
    assert confirmed.status == HypothesisStatus.CONFIRMED
    assert confirmed.evidence_for == ["run-for"]
    assert confirmed.evidence_against == []

    # Dedupe on re-append
    confirmed_again = store.update_status(
        hypothesis.id,
        HypothesisStatus.CONFIRMED,
        evidence_run_id="run-for",
    )
    assert confirmed_again.evidence_for == ["run-for"]

    rejected = store.update_status(
        hypothesis.id,
        HypothesisStatus.REJECTED,
        evidence_run_id="run-against",
    )
    assert rejected.status == HypothesisStatus.REJECTED
    assert rejected.evidence_against == ["run-against"]
    # previously confirmed evidence remains
    assert rejected.evidence_for == ["run-for"]

    inconclusive = store.update_status(
        hypothesis.id,
        HypothesisStatus.INCONCLUSIVE,
        evidence_run_id="run-skip",
    )
    assert inconclusive.evidence_for == ["run-for"]
    assert inconclusive.evidence_against == ["run-against"]


def test_mark_testing_if_proposed_only_from_proposed(tmp_path: Path):
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    hypothesis = store.create(
        observation="a", reason="b", prediction="c", confidence=0.5
    )
    testing = store.mark_testing_if_proposed(hypothesis.id)
    assert testing.status == HypothesisStatus.TESTING

    store.update_status(hypothesis.id, HypothesisStatus.CONFIRMED)
    again = store.mark_testing_if_proposed(hypothesis.id)
    assert again.status == HypothesisStatus.CONFIRMED


def test_mark_testing_if_proposed_claim_race_is_exclusive(tmp_path: Path):
    """M11: concurrent claimers must not both observe `proposed`."""
    import threading

    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    hypothesis = store.create(observation="a", reason="b", prediction="c", confidence=0.5)

    winners: list[HypothesisStatus] = []
    lock = threading.Lock()

    def claim() -> None:
        result = store.mark_testing_if_proposed(hypothesis.id)
        with lock:
            winners.append(result.status)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every caller gets back TESTING (the winner transitioned it, everyone
    # else's read then sees the already-`testing` hypothesis) — the race this
    # guards against is a *file write* race, not the return value, so assert
    # on-disk state directly: exactly one hypothesis, status TESTING, and no
    # corrupted/partial JSON from an interleaved write.
    assert all(status == HypothesisStatus.TESTING for status in winners)
    reloaded = store.get(hypothesis.id)
    assert reloaded is not None
    assert reloaded.status == HypothesisStatus.TESTING


def test_release_claim_reverts_testing_to_proposed(tmp_path: Path):
    """M11: rollback path when worktree setup fails after a successful claim."""
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    hypothesis = store.create(observation="a", reason="b", prediction="c", confidence=0.5)

    store.mark_testing_if_proposed(hypothesis.id)
    released = store.release_claim(hypothesis.id)
    assert released.status == HypothesisStatus.PROPOSED
    assert store.get(hypothesis.id).status == HypothesisStatus.PROPOSED


def test_release_claim_leaves_other_statuses_untouched(tmp_path: Path):
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    hypothesis = store.create(observation="a", reason="b", prediction="c", confidence=0.5)

    store.update_status(hypothesis.id, HypothesisStatus.CONFIRMED)
    released = store.release_claim(hypothesis.id)
    assert released.status == HypothesisStatus.CONFIRMED


def test_create_mirrors_hypothesis_into_knowledge_db(tmp_path: Path):
    import json

    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore

    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    hyp = store.create(
        observation="Rare classes underperform",
        reason="Imbalance",
        prediction="Focal loss helps",
        confidence=0.7,
        expected_impact=0.015,
        tags=["focal_loss"],
    )
    assert hyp.expected_impact == pytest.approx(0.015)
    with KnowledgeStore(tmp_path / "knowledge", "titanic") as kstore:
        row = kstore.get_hypothesis(hyp.id)
        assert row is not None
        assert row["observation"] == "Rare classes underperform"
        assert row["prediction"] == "Focal loss helps"
        assert row["rationale"] == "Imbalance"
        assert row["status"] == "proposed"
        assert row["confidence"] == pytest.approx(0.7)
        assert row["expected_impact"] == pytest.approx(0.015)
        meta = json.loads(row["metadata"])
        assert meta["tags"] == ["focal_loss"]

    store.update_status(hyp.id, HypothesisStatus.TESTING)
    with KnowledgeStore(tmp_path / "knowledge", "titanic") as kstore:
        row = kstore.get_hypothesis(hyp.id)
        assert row is not None
        assert row["status"] == "testing"


def test_proposed_hypothesis_tags_are_not_tried(tmp_path: Path):
    from labpilot.research_engine.intelligence.hypothesis.persist import (
        load_existing_technique_tags,
    )
    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
    from labpilot.research_engine.intelligence.models import (
        ResearchArtifact,
        ResearchArtifactType,
    )

    knowledge_dir = tmp_path / "knowledge"
    store = HypothesisStore(knowledge_dir, "titanic")
    store.create(
        observation="a",
        reason="b",
        prediction="c",
        confidence=0.5,
        tags=["Focal Loss"],
    )
    testing = store.create(
        observation="d",
        reason="e",
        prediction="f",
        confidence=0.5,
        tags=["SpecAugment"],
    )
    store.update_status(testing.id, HypothesisStatus.TESTING)

    with KnowledgeStore(knowledge_dir, "titanic") as kstore:
        kstore.upsert_artifact(
            ResearchArtifact(
                id="exp:1",
                type=ResearchArtifactType.EXPERIMENT,
                source="m2",
                title="run",
                techniques=["Mixup"],
            )
        )

    tried = load_existing_technique_tags(knowledge_dir, "titanic")
    assert "focal loss" not in tried
    assert "specaugment" in tried
    assert "mixup" in tried


def test_mark_testing_missing_raises(tmp_path: Path):
    store = HypothesisStore(tmp_path / "knowledge", "titanic")
    with pytest.raises(FileNotFoundError, match="H-999"):
        store.mark_testing_if_proposed("H-999")


def test_linked_experiments_filters_graph(tmp_path: Path):
    from datetime import datetime

    graph = ExperimentGraph(
        "titanic",
        {
            "a": Experiment(
                id="a",
                competition="titanic",
                status="completed",
                progress="14/14 stages",
                description="x",
                hypothesis_id="H-001",
                created_at=datetime(2026, 1, 1),
            ),
            "b": Experiment(
                id="b",
                competition="titanic",
                status="completed",
                progress="14/14 stages",
                description="y",
                hypothesis_id="H-002",
                created_at=datetime(2026, 1, 2),
            ),
            "c": Experiment(
                id="c",
                competition="titanic",
                status="completed",
                progress="14/14 stages",
                description="z",
                created_at=datetime(2026, 1, 3),
            ),
        },
    )
    linked = linked_experiments("H-001", graph)
    assert [exp.id for exp in linked] == ["a"]


def _seed_run(runs_dir: Path, run_id: str, *, hypothesis_id: str | None = None) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    import json

    metadata = {}
    if hypothesis_id:
        metadata["hypothesis_id"] = hypothesis_id
    manifest = {
        "run_id": run_id,
        "competition": "titanic",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "status": "completed",
        "stages": [
            {
                "name": "parse_competition",
                "status": "completed",
                "started_at": "2026-01-01T00:00:00",
                "finished_at": "2026-01-01T00:00:01",
                "artifacts": [],
                "error": None,
            }
        ],
        "metadata": metadata,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "baseline_choice.json").write_text(
        json.dumps(
            {
                "problem_type": "tabular_classification",
                "template_name": "tabular_classification",
                "rationale": "test",
                "target_column": "Survived",
            }
        )
    )
    return run_dir


def test_description_prefers_hypothesis_prediction(tmp_path: Path):
    knowledge = tmp_path / "knowledge"
    store = HypothesisStore(knowledge, "titanic")
    hypothesis = store.create(
        observation="obs",
        reason="reason",
        prediction="Focal Loss will improve Macro F1",
        confidence=0.74,
    )
    runs_dir = tmp_path / "runs"
    run_dir = _seed_run(runs_dir, "run-1", hypothesis_id=hypothesis.id)

    experiment = assemble_experiment(run_dir, knowledge_dir=knowledge)
    assert experiment.description == "Focal Loss will improve Macro F1"
    assert experiment.hypothesis_id == hypothesis.id


def test_description_falls_back_when_hypothesis_missing(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_dir = _seed_run(runs_dir, "run-1", hypothesis_id="H-404")
    experiment = assemble_experiment(run_dir, knowledge_dir=tmp_path / "knowledge")
    assert experiment.description == "tabular_classification baseline for titanic"


def test_build_graph_passes_knowledge_dir(tmp_path: Path):
    knowledge = tmp_path / "knowledge"
    store = HypothesisStore(knowledge, "titanic")
    hypothesis = store.create(
        observation="o", reason="r", prediction="pred text", confidence=0.5
    )
    runs_dir = tmp_path / "runs"
    _seed_run(runs_dir, "run-1", hypothesis_id=hypothesis.id)
    graph = build_graph(runs_dir, "titanic", knowledge_dir=knowledge)
    assert graph.nodes["run-1"].description == "pred text"
