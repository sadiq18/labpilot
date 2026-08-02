"""FE recipes, experiment ledger, stacked ranking, overlays, WRITE_CODE override."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
    CodeEngineeringCapability,
)
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import ResearchExecution
from labpilot.research_engine.intelligence.feature_recipes import (
    FEATURE_ENGINEERING_CATEGORY,
    FeatureRecipe,
    heuristic_feature_recipes,
)
from labpilot.research_engine.intelligence.hypothesis.candidates import generate_candidates
from labpilot.research_engine.intelligence.hypothesis.ledger import build_experiment_ledger
from labpilot.research_engine.intelligence.hypothesis.models import HypothesisCandidateKind
from labpilot.research_engine.intelligence.hypothesis.ranking import score_candidate
from labpilot.research_engine.intelligence.knowledge.merger import _fe_category
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.micro_agents.forum_analyzer import ForumAnalyzerAgent
from labpilot.research_engine.intelligence.micro_agents.paper_analyzer import PaperAnalyzerAgent
from labpilot.research_engine.intelligence.micro_agents.repository_analyzer import (
    RepositoryAnalyzerAgent,
)
from labpilot.research_engine.intelligence.models import ResearchArtifact, ResearchArtifactType
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.intelligence.retrieval.models import (
    QueryType,
    ResearchContext,
    RetrievalIntent,
)
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import (
    HypothesisCreatedBy,
    HypothesisGenerator,
    HypothesisOrigin,
    HypothesisStatus,
)
from labpilot.research_engine.shared.skills import (
    load_skill_overlay,
    summarize_skill_text,
    upsert_skill_overlay,
)


def test_heuristic_feature_recipes_extract_target_encoding() -> None:
    recipes = heuristic_feature_recipes(
        "We apply target encoding on categorical columns to create new features."
    )
    assert recipes
    assert any("target" in r.name.lower() or "encod" in r.name.lower() for r in recipes)


def test_repo_and_paper_analyzers_emit_feature_recipes() -> None:
    text = (
        "Pipeline uses target encoding and frequency encoding to create "
        "new categorical features before LightGBM."
    )
    repo = RepositoryAnalyzerAgent().run(
        StructuredContext(
            competition="demo",
            text=text,
            data={"repo_id": "r1", "full_name": "org/demo", "has_readme": True},
        )
    )
    assert repo.feature_recipes or any(
        "encod" in t.lower() or "feature" in t.lower() for t in repo.techniques
    )

    paper = PaperAnalyzerAgent().run(
        StructuredContext(
            competition="demo",
            text=text,
            data={"paper_id": "p1", "title": "FE paper"},
        )
    )
    assert paper.feature_recipes or any("encod" in t.lower() for t in paper.techniques)


def test_forum_analyzer_attaches_fe_techniques() -> None:
    extract = ForumAnalyzerAgent().run(
        StructuredContext(
            competition="demo",
            text=(
                "Found that target encoding improved CV a lot. "
                "Key insight: frequency encoding also helped."
            ),
        )
    )
    assert extract.techniques or extract.feature_recipes


def test_fe_category_helper() -> None:
    assert _fe_category("target_encoding") == FEATURE_ENGINEERING_CATEGORY
    assert _fe_category("SpecAugment") == ""


def test_experiment_ledger_marks_worked_failed_untried(tmp_path: Path) -> None:
    competition = "ledger-demo"
    with KnowledgeStore(tmp_path, competition) as store:
        store.upsert_artifact(
            ResearchArtifact(
                id="paper:fe",
                type=ResearchArtifactType.PAPER,
                source="test",
                title="FE",
                techniques=["Target Encoding", "UnusedBoost"],
                claims=["claim about unused boost"],
                confidence=0.8,
                metadata={
                    "feature_recipes": [
                        FeatureRecipe(
                            name="Target Encoding",
                            transform="mean target by category",
                        ).model_dump()
                    ]
                },
            )
        )
        store.merge_technique("Target Encoding", category=FEATURE_ENGINEERING_CATEGORY)
        store.merge_technique("UnusedBoost", category="model")
        store.merge_technique("FailedAug", category="augmentation")

    hyps = HypothesisStore(tmp_path, competition)
    now = datetime.now()
    won = hyps.create(
        observation="TE helped",
        reason="artifact paper:fe; technique Target Encoding",
        prediction="TE improves metric",
        confidence=0.7,
        expected_impact=0.02,
        tags=["Target Encoding", "stacked"],
        technique="Target Encoding",
        technique_stack=["Target Encoding"],
        created_by=HypothesisCreatedBy.HYPOTHESIZE,
        generator=HypothesisGenerator.RULE_ENGINE,
        origin=HypothesisOrigin.PAPER,
    )
    hyps.update_outcome(
        won.id,
        actual_outcome="gain +0.02",
        status=HypothesisStatus.CONFIRMED,
        evidence_run_id="E-001",
    )
    failed = hyps.create(
        observation="aug failed",
        reason="bad aug",
        prediction="aug helps",
        confidence=0.5,
        tags=["FailedAug"],
        technique="FailedAug",
    )
    hyps.update_outcome(
        failed.id,
        actual_outcome="loss",
        status=HypothesisStatus.REJECTED,
        evidence_run_id="E-002",
    )

    ledger = build_experiment_ledger(tmp_path, competition)
    assert any("Target" in t or "target" in t.lower() for t in ledger.techniques_worked)
    assert any("Failed" in t or "failed" in t.lower() for t in ledger.techniques_failed)
    assert any("Unused" in t or "unused" in t.lower() for t in ledger.techniques_untried)
    assert ledger.winning_hypothesis_id == won.id


def test_stacked_candidates_rank_above_fresh(tmp_path: Path) -> None:
    competition = "stack-demo"
    with KnowledgeStore(tmp_path, competition) as store:
        store.upsert_artifact(
            ResearchArtifact(
                id="repo:x",
                type=ResearchArtifactType.REPOSITORY,
                source="github",
                title="x",
                techniques=["Alpha", "Beta"],
                confidence=0.7,
            )
        )
        store.merge_technique("Alpha")
        store.merge_technique("Beta")

    hyps = HypothesisStore(tmp_path, competition)
    parent = hyps.create(
        observation="Alpha worked",
        reason="technique Alpha",
        prediction="Alpha improves",
        confidence=0.7,
        expected_impact=0.03,
        tags=["Alpha"],
        technique="Alpha",
        technique_stack=["Alpha"],
    )
    hyps.update_outcome(
        parent.id,
        actual_outcome="gain",
        status=HypothesisStatus.CONFIRMED,
        evidence_run_id="E-010",
    )

    ledger = build_experiment_ledger(tmp_path, competition)
    ctx = ResearchContext(
        intent=RetrievalIntent(
            question="next",
            query_type=QueryType.HYPOTHESIS_GENERATION,
        ),
        techniques=[{"name": "Beta", "confidence": 0.6}],
    )
    candidates = generate_candidates(ctx, ledger=ledger)
    stacked = [c for c in candidates if c.kind == HypothesisCandidateKind.STACKED]
    fresh = [
        c
        for c in candidates
        if c.kind == HypothesisCandidateKind.TECHNIQUE and c.technique == "Beta"
    ]
    assert stacked
    assert stacked[0].parent_hypothesis_id == parent.id
    if fresh:
        assert score_candidate(stacked[0]) > score_candidate(fresh[0])


def test_skill_overlay_upsert_summarize_and_inject(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    path = upsert_skill_overlay(
        ws,
        "code_engineer",
        lesson_id="E-001",
        keep=["target encoding"],
        avoid=["leaky CV"],
        try_next=["stack frequency encoding"],
        note="first win",
    )
    assert path.is_file()
    # Idempotent
    upsert_skill_overlay(
        ws,
        "code_engineer",
        lesson_id="E-001",
        keep=["duplicate"],
    )
    text = path.read_text(encoding="utf-8")
    assert text.count("<!-- lesson:E-001 -->") == 1

    huge = "# Competition skill overlay\n" + ("- Keep: x\n" * 400)
    summarized = summarize_skill_text(huge, max_chars=500)
    assert len(summarized) <= 500

    # Force on-disk summarize via low budget
    for i in range(20):
        upsert_skill_overlay(
            ws,
            "code_engineer",
            lesson_id=f"E-{i+2:03d}",
            keep=[f"keep-{i}" * 20],
            avoid=[f"avoid-{i}" * 20],
            try_next=[f"try-{i}" * 20],
            note="n" * 200,
            on_disk_budget=800,
        )
    overlay = load_skill_overlay(ws, "code_engineer", max_chars=400)
    assert overlay
    assert len(overlay) <= 400


def test_write_code_overrides_existing_train_py(tmp_path: Path) -> None:
    competition = "override-demo"
    knowledge = tmp_path / "knowledge"
    paths = ResearchPaths(knowledge, competition).ensure()
    ws = tmp_path / "workspace"
    (ws / "pipeline").mkdir(parents=True)
    (ws / "artifacts").mkdir(parents=True)
    train = ws / "pipeline" / "train.py"
    train.write_text("print('prior')\n", encoding="utf-8")
    (ws / "profile.json").write_text(
        '{"competition":"override-demo","problem_type":"tabular_regression"}',
        encoding="utf-8",
    )
    # The baseline selector reads the problem type from the competition
    # contract; without it the type is "unknown", no template matches, and a
    # non-dry run now refuses to continue rather than leaving a stub behind.
    (ws / "competition.json").write_text(
        '{"slug":"override-demo","title":"Override Demo",'
        '"problem_type":"tabular_regression","tags":["tabular"]}',
        encoding="utf-8",
    )

    now = datetime.now()
    plan = ResearchPlan(
        id="P-002",
        competition=competition,
        hypothesis_id="H-001",
        goal="Improve prior",
        status=PlanStatus.READY,
        metadata={
            "parent_hypothesis_id": "H-BASELINE",
            "technique": "target_encoding",
            "force_rewrite": True,
        },
        tasks=[
            ResearchTask(
                id="P-002-T01",
                plan_id="P-002",
                type=TaskType.WRITE_CODE,
                description="Override train.py with TE",
            )
        ],
        created_at=now,
        updated_at=now,
    )
    execution = ResearchExecution(
        id="E-100",
        plan_id=plan.id,
        competition=competition,
    )
    ctx = TaskContext(
        plan=plan,
        task=plan.tasks[0],
        execution=execution,
        paths=paths,
        workspace_root=ws,
        competition=competition,
    )
    evidence = CodeEngineeringCapability(llm_client=None).execute(ctx)
    assert evidence.passed
    assert evidence.metadata.get("overrode_existing") is True
    backups = list((ws / "artifacts" / "code_backups").glob("train_E-100.py"))
    assert backups
    assert "prior" in backups[0].read_text(encoding="utf-8")
