"""Integration: structured reflection side effects on improve + hypothesis."""

import json
from pathlib import Path

from labpilot.config import AppConfig
from labpilot.experiments.graph import assemble_experiment, build_graph
from labpilot.experiments.hypothesis import HypothesisStore
from labpilot.experiments.models import HypothesisStatus
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline
from helpers.kaggle import FakeKaggleGateway


class FakeReflectionLLM:
    """Returns a confirmed update for H-001 plus one new draft."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return json.dumps(
            {
                "observation": "Metric improved vs parent",
                "evidence": ["cv score up"],
                "likely_cause": "Tuned hyperparameters",
                "confidence": 0.8,
                "suggested_next": ["Try feature recipes"],
                "hypothesis_updates": [
                    {
                        "hypothesis_id": "H-001",
                        "new_status": "confirmed",
                        "note": "CV improved",
                    },
                    {
                        "hypothesis_id": "H-999",
                        "new_status": "rejected",
                        "note": "should be ignored",
                    },
                ],
                "new_hypotheses": [
                    {
                        "observation": "Title feature unused",
                        "reason": "Name encodes class",
                        "prediction": "Title extraction helps",
                        "confidence": 0.55,
                        "tags": ["features"],
                    }
                ],
            }
        )


def test_improve_with_hypothesis_applies_structured_reflection_side_effects(
    tmp_path: Path,
    titanic_data_dir: Path,
    competition_configs_dir: Path,
):
    gateway = FakeKaggleGateway(titanic_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"
    config.knowledge_dir = tmp_path / "knowledge"
    config.runs_dir.mkdir(parents=True)

    store = HypothesisStore(config.knowledge_dir, "titanic")
    hypothesis = store.create(
        observation="Params matter",
        reason="LR sensitive",
        prediction="Tuning helps",
        confidence=0.6,
    )
    assert hypothesis.id == "H-001"

    llm = FakeReflectionLLM()
    parent = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
        llm_client=llm,
    ).run("titanic")
    assert parent.status == StageStatus.COMPLETED

    # Root reflection may also create drafts; reset knowledge to just H-001 after parent.
    # Keep H-001; delete any drafts created by root run so the improve assertion is clean.
    for path in (config.knowledge_dir / "titanic" / "hypotheses").glob("H-*.json"):
        if path.stem != "H-001":
            path.unlink()

    child = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
        llm_client=llm,
    ).improve(parent.run_id, strategy="tune", hypothesis_id="H-001")
    assert child.status == StageStatus.COMPLETED

    child_dir = config.runs_dir / child.run_id
    assert (child_dir / "reflection.json").is_file()
    assert (child_dir / "reflection.md").is_file()
    assert (child_dir / "comparison.json").is_file()

    updated = store.get("H-001")
    assert updated is not None
    assert updated.status == HypothesisStatus.CONFIRMED
    assert child.run_id in updated.evidence_for

    # Mismatched H-999 must not be created / updated
    assert store.get("H-999") is None

    # One new draft from improve reflection
    all_hyp = store.list()
    assert any(h.source == "reflection" for h in all_hyp)

    assembled = assemble_experiment(child_dir, knowledge_dir=config.knowledge_dir)
    assert assembled.reflection is not None
    assert assembled.reflection.generated_by == "llm"
    assert assembled.reflection.observation == "Metric improved vs parent"

    graph = build_graph(config.runs_dir, "titanic", knowledge_dir=config.knowledge_dir)
    assert graph.nodes[child.run_id].reflection is not None
    assert (
        graph.nodes[child.run_id].reflection.model_dump()
        == assembled.reflection.model_dump()
    )
