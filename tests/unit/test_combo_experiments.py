"""Combination shortlist, LLM/rule picks, ranking, ablation / avoid_pairs."""

from __future__ import annotations

from pathlib import Path

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.execution.outcome import (
    ExecutionOutcomeSummary,
    maybe_mint_ablation_from_combo_win,
    record_combo_avoid_on_loss,
)
from labpilot.research_engine.intelligence.hypothesis.combo import (
    build_combo_shortlist,
    filter_picks_to_shortlist,
    picks_to_candidates,
    rule_engine_pick_combos,
    technique_category,
)
from labpilot.research_engine.intelligence.hypothesis.ledger import build_experiment_ledger
from labpilot.research_engine.intelligence.hypothesis.models import HypothesisCandidateKind
from labpilot.research_engine.intelligence.hypothesis.ranking import (
    rank_candidates,
    score_candidate,
)
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.micro_agents.combo_portfolio import (
    ComboPortfolioAgent,
)
from labpilot.research_engine.intelligence.models import ResearchArtifact, ResearchArtifactType
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus


def _seed_untried(tmp_path: Path, competition: str, techniques: list[str]) -> None:
    with KnowledgeStore(tmp_path, competition) as store:
        store.upsert_artifact(
            ResearchArtifact(
                id="repo:combo",
                type=ResearchArtifactType.REPOSITORY,
                source="github",
                title="combo seed",
                techniques=techniques,
                confidence=0.7,
            )
        )
        for name in techniques:
            store.merge_technique(name)


def test_technique_category_diverse() -> None:
    assert technique_category("target encoding") == "feature_engineering"
    assert technique_category("LightGBM") == "model"
    assert technique_category("mixup") == "augmentation"


def test_shortlist_excludes_failed_and_avoid_pairs(tmp_path: Path) -> None:
    competition = "combo-shortlist"
    techs = [
        "Target Encoding",
        "LightGBM",
        "Mixup",
        "SWA",
        "Frequency Encoding",
        "CatBoost",
    ]
    _seed_untried(tmp_path, competition, techs)

    hyps = HypothesisStore(tmp_path, competition)
    parent = hyps.create(
        observation="won",
        reason="r",
        prediction="p",
        confidence=0.7,
        expected_impact=0.02,
        tags=["Alpha"],
        technique="Alpha",
        technique_stack=["Alpha"],
    )
    hyps.update_outcome(
        parent.id,
        actual_outcome="gain",
        status=HypothesisStatus.CONFIRMED,
        evidence_run_id="E-1",
    )
    failed_combo = hyps.create(
        observation="combo lost",
        reason="r",
        prediction="p",
        confidence=0.6,
        tags=["Target Encoding", "LightGBM", "combination"],
        technique="Target Encoding+LightGBM",
        parent_hypothesis_id=parent.id,
        technique_stack=["Alpha", "Target Encoding", "LightGBM"],
        combo_techniques=["Target Encoding", "LightGBM"],
    )
    hyps.update_outcome(
        failed_combo.id,
        actual_outcome="loss",
        status=HypothesisStatus.REJECTED,
        evidence_run_id="E-2",
    )

    ledger = build_experiment_ledger(tmp_path, competition)
    shortlist = build_combo_shortlist(ledger)
    assert shortlist
    for portfolio in shortlist:
        members = {m.lower() for m in portfolio["techniques"]}
        assert not (
            "target encoding" in members and "lightgbm" in members
        ), "failed combo pair must be excluded from shortlist"


def test_filter_picks_drops_invented_techniques() -> None:
    shortlist = [
        {
            "id": "pair:a+b",
            "techniques": ["Target Encoding", "LightGBM"],
            "categories": ["feature_engineering", "model"],
            "size": 2,
            "diversity_score": 1.0,
        }
    ]
    cleaned = filter_picks_to_shortlist(
        [
            {
                "techniques": ["Target Encoding", "InventedMagic"],
                "rationale": "hallucination",
                "confidence": 0.9,
                "expected_impact": 0.05,
            },
            {
                "techniques": ["Target Encoding", "LightGBM"],
                "rationale": "ok",
                "confidence": 0.8,
                "expected_impact": 0.02,
            },
        ],
        shortlist,
    )
    assert len(cleaned) == 1
    assert cleaned[0]["techniques"] == ["Target Encoding", "LightGBM"]


def test_rule_engine_combo_fallback_without_llm(tmp_path: Path) -> None:
    competition = "combo-rule"
    _seed_untried(
        tmp_path,
        competition,
        ["Target Encoding", "LightGBM", "Mixup", "SWA"],
    )
    ledger = build_experiment_ledger(tmp_path, competition)
    shortlist = build_combo_shortlist(ledger)
    assert shortlist
    picks = rule_engine_pick_combos(shortlist, limit=2)
    assert picks
    assert all(len(p["techniques"]) >= 2 for p in picks)
    candidates = picks_to_candidates(picks, ledger)
    assert candidates
    assert all(c.kind == HypothesisCandidateKind.COMBINATION for c in candidates)
    assert all(c.metadata.get("combo_techniques") for c in candidates)

    agent = ComboPortfolioAgent(llm_client=None)
    draft = agent.run(
        StructuredContext(
            competition=competition,
            data={"shortlist": shortlist, "limit": 2},
        )
    )
    assert draft.picks
    assert not agent.last_used_llm


def test_combination_ranks_above_stacked_and_single() -> None:
    from labpilot.research_engine.intelligence.hypothesis.models import HypothesisCandidate
    from labpilot.research_engine.intelligence.repositories.models import (
        EffortEstimate,
        ExpectedGain,
    )

    combo = HypothesisCandidate(
        key="combination:te+lgbm",
        kind=HypothesisCandidateKind.COMBINATION,
        title="Combine TE + LGBM",
        technique="Target Encoding+LightGBM",
        expected_impact=ExpectedGain.MEDIUM,
        confidence=0.7,
        implementation_effort=EffortEstimate.HOURS_1,
        tags=["combination", "stacked"],
        parent_hypothesis_id="H-001",
        technique_stack=["Alpha", "Target Encoding", "LightGBM"],
        metadata={"combo_techniques": ["Target Encoding", "LightGBM"]},
        score_hint=0.8,
    )
    stacked = HypothesisCandidate(
        key="stacked:te",
        kind=HypothesisCandidateKind.STACKED,
        title="Stack TE",
        technique="Target Encoding",
        expected_impact=ExpectedGain.MEDIUM,
        confidence=0.7,
        implementation_effort=EffortEstimate.HOURS_1,
        tags=["stacked"],
        parent_hypothesis_id="H-001",
        technique_stack=["Alpha", "Target Encoding"],
        score_hint=0.7,
    )
    single = HypothesisCandidate(
        key="technique:te",
        kind=HypothesisCandidateKind.TECHNIQUE,
        title="Try TE",
        technique="Target Encoding",
        expected_impact=ExpectedGain.MEDIUM,
        confidence=0.7,
        implementation_effort=EffortEstimate.HOURS_1,
        tags=["technique"],
        score_hint=0.5,
    )
    assert score_candidate(combo) > score_candidate(stacked) > score_candidate(single)
    ranked = rank_candidates([single, stacked, combo], limit=3)
    assert ranked[0][1].kind == HypothesisCandidateKind.COMBINATION


def test_ablation_on_combo_gain_not_on_loss(tmp_path: Path) -> None:
    competition = "combo-ablate"
    hyps = HypothesisStore(tmp_path, competition)
    combo = hyps.create(
        observation="combo",
        reason="r",
        prediction="p",
        confidence=0.75,
        expected_impact=0.02,
        tags=["Target Encoding", "LightGBM", "combination", "stacked"],
        technique="Target Encoding+LightGBM",
        technique_stack=["Target Encoding", "LightGBM"],
        combo_techniques=["Target Encoding", "LightGBM"],
    )
    gain_summary = ExecutionOutcomeSummary(
        competition=competition,
        execution_id="E-10",
        plan_id="P-10",
        hypothesis_id=combo.id,
        learning_gain=0.03,
        learning_loss=None,
        hypothesis_outcome={"actual_outcome": "gain"},
    )
    ablations = maybe_mint_ablation_from_combo_win(
        knowledge_dir=tmp_path,
        competition=competition,
        summary=gain_summary,
    )
    assert len(ablations) == 2
    for hid in ablations:
        h = hyps.get(hid)
        assert h is not None
        assert "ablation" in {t.lower() for t in h.tags}
        assert len(h.combo_techniques) == 1

    loss_summary = ExecutionOutcomeSummary(
        competition=competition,
        execution_id="E-11",
        plan_id="P-11",
        hypothesis_id=combo.id,
        learning_gain=None,
        learning_loss=0.02,
        hypothesis_outcome={"actual_outcome": "loss"},
    )
    assert (
        maybe_mint_ablation_from_combo_win(
            knowledge_dir=tmp_path,
            competition=competition,
            summary=loss_summary,
        )
        == []
    )
    record_combo_avoid_on_loss(
        knowledge_dir=tmp_path,
        competition=competition,
        summary=loss_summary,
    )
    updated = hyps.get(combo.id)
    assert updated is not None
    assert updated.status == HypothesisStatus.REJECTED

    ledger = build_experiment_ledger(tmp_path, competition)
    avoid = {(a.lower(), b.lower()) for a, b in ledger.avoid_pairs} | {
        (b.lower(), a.lower()) for a, b in ledger.avoid_pairs
    }
    assert ("target encoding", "lightgbm") in avoid
