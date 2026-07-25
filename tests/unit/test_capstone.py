"""Plan 11 — Capstone: terminal mockup, analyze.json contract, north-star Q1–Q5."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from labpilot.research_engine.intelligence.hypothesis import HypothesisAssistant
from labpilot.research_engine.intelligence.knowledge import KnowledgeStore
from labpilot.research_engine.intelligence.models import AnalysisReport
from labpilot.research_engine.intelligence.renderers.json import (
    PUBLIC_TOP_LEVEL_KEYS,
    assert_public_contract,
    validate_json,
    write_report,
)
from labpilot.research_engine.intelligence.renderers.terminal import render_terminal_text
from labpilot.research_engine.intelligence.retrieval import ContextBuilder, QueryType
from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label
from helpers.capstone_fixture import (
    CAPSTONE_SLUG,
    run_capstone_analyze,
    seed_capstone_store,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "capstone"
GOLDEN_ANALYZE = FIXTURES / "analyze.golden.json"
ANALYZERS_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "labpilot"
    / "research_engine"
    / "intelligence"
    / "analyzers"
)


def _normalize_report(payload: dict) -> dict:
    """Strip volatile fields for golden comparison."""
    data = json.loads(json.dumps(payload))
    data["generated_at"] = "<stable>"
    for card in data.get("hypothesis_recommendations", []):
        if card.get("hypothesis_id"):
            card["hypothesis_id"] = "H-XXX"
    for card in data.get("suggested_experiments", []):
        if card.get("hypothesis_id"):
            card["hypothesis_id"] = "H-XXX"
    for hyp in data.get("hypotheses", []):
        if hyp.get("id"):
            hyp["id"] = "H-XXX"
    # Hypothesis notes may include counts that stay stable; keep as-is.
    return data


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# --- contract + terminal ----------------------------------------------------


def test_analyze_json_validates_public_contract(tmp_path: Path) -> None:
    report, path = run_capstone_analyze(tmp_path)
    loaded = validate_json(path.read_text())
    assert_public_contract(loaded)
    assert loaded.schema_version == 1
    assert set(loaded.model_dump(mode="json")) >= PUBLIC_TOP_LEVEL_KEYS
    assert loaded.competition["slug"] == CAPSTONE_SLUG
    assert loaded.techniques.locally_validated  # Mixup promoted in fixture
    assert "Mixup" in loaded.techniques.locally_validated
    # External-only must never appear under locally_validated as Established.
    assert "Established" not in json.dumps(loaded.techniques.model_dump())


def test_golden_analyze_json_snapshot(tmp_path: Path) -> None:
    report, _path = run_capstone_analyze(tmp_path)
    actual = _normalize_report(report.model_dump(mode="json"))
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if not GOLDEN_ANALYZE.is_file():
        GOLDEN_ANALYZE.write_text(json.dumps(actual, indent=2) + "\n")
    expected = json.loads(GOLDEN_ANALYZE.read_text())
    # Compare stable contract sections (hypotheses ranks may reorder slightly —
    # assert structural keys + technique buckets + winning solutions honesty).
    assert actual["competition"]["slug"] == expected["competition"]["slug"]
    assert actual["techniques"]["locally_validated"] == expected["techniques"][
        "locally_validated"
    ]
    assert (
        actual["competition"]["winning_solutions"]["status"]
        == expected["competition"]["winning_solutions"]["status"]
        == "unavailable"
    )
    assert set(actual) >= PUBLIC_TOP_LEVEL_KEYS
    assert len(actual["hypothesis_recommendations"]) == len(
        expected["hypothesis_recommendations"]
    )


def test_terminal_mockup_parity_and_no_established_external(tmp_path: Path) -> None:
    report, _ = run_capstone_analyze(tmp_path)
    text = _plain(render_terminal_text(report))
    for heading in (
        "Competition Summary",
        "Related Competitions",
        "Relevant Papers",
        "Relevant Experiments",
        "Relevant Repositories",
        "Relevant Failures",
        "Winning Solutions",
        "External Recommendations",
        "Locally Validated",
        "Suggested Next Experiments",
    ):
        assert heading in text
    assert "Unavailable" in text
    assert "Suggested" in text
    assert "#1" in text or "# 1" in text or "impact=" in text
    # Never claim Established for external-only techniques.
    assert "Established" not in text
    assert "Mixup" in text


def test_analyzers_do_not_import_renderers() -> None:
    offenders: list[str] = []
    for path in ANALYZERS_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "renderers" in alias.name:
                        offenders.append(f"{path}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if "renderers" in node.module:
                    offenders.append(f"{path}:{node.module}")
    assert offenders == []


# --- north-star success criteria (README §1) --------------------------------


def test_q1_techniques_improve_macro_f1(tmp_path: Path) -> None:
    """Q1: techniques that improve Macro F1 — grounded in experiment evidence."""
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, CAPSTONE_SLUG) as store:
        seed_capstone_store(store)
        context = ContextBuilder(store).build(
            "What techniques consistently improve Macro F1 on imbalanced "
            "audio-classification tasks?",
            query_type=QueryType.STRUCTURED_QUERY,
            profile={
                "slug": CAPSTONE_SLUG,
                "domain": "bioacoustics",
                "metric": {"name": "macro_f1"},
                "task": "Audio Classification",
            },
            progressive=False,
        )
    names = {normalize_label(str(t.get("name") or "")) for t in context.techniques}
    assert "mixup" in names
    mixup = next(
        t for t in context.techniques if normalize_label(str(t.get("name"))) == "mixup"
    )
    evidence_ids = set(mixup.get("experiment_ids") or []) | set(
        mixup.get("paper_ids") or []
    )
    assert "exp:12" in evidence_ids or "paper:mixup-macro-f1" in set(
        mixup.get("paper_ids") or []
    )
    assert evidence_ids


def test_q2_winning_solutions_ema_unavailable(tmp_path: Path) -> None:
    """Q2: winning solutions for EMA — honest Unavailable (Null provider)."""
    report, _ = run_capstone_analyze(tmp_path)
    winning = report.competition.get("winning_solutions") or {}
    assert winning.get("status") == "unavailable"
    assert winning.get("available") is False
    assert "provider" in str(winning.get("reason") or "").lower()
    # Must not fabricate a list of BirdCLEF winners that used EMA.
    assert not (winning.get("items") or [])


def test_q3_focal_loss_hurt_experiments(tmp_path: Path) -> None:
    """Q3: experiments where Focal Loss hurt — exact local failure evidence."""
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, CAPSTONE_SLUG) as store:
        seed_capstone_store(store)
        context = ContextBuilder(store).build(
            "Show experiments where Focal Loss hurt performance",
            query_type=QueryType.STRUCTURED_QUERY,
            progressive=False,
        )
    failure_ids = {str(item.get("document_id") or "") for item in context.failures}
    exp_ids = {str(item.get("document_id") or "") for item in context.experiments}
    # exp:19 is the deterministic negative Focal Loss run.
    assert "exp:19" in failure_ids or "exp:19" in exp_ids
    text = json.dumps(context.model_dump(mode="json")).lower()
    assert "focal" in text
    assert any(marker in text for marker in ("hurt", "delta", "rare"))


def test_q4_compatible_github_transfers(tmp_path: Path) -> None:
    """Q4: GitHub implementations compatible with pipeline → transfer cards."""
    report, _ = run_capstone_analyze(tmp_path)
    assert report.transfer_opportunities
    top = report.transfer_opportunities[0]
    assert "Focal" in str(top.get("remote_choice") or top.get("summary") or "")
    assert top.get("effort")
    assert top.get("expected_gain")
    assert any(
        a.id.startswith("repo:") for a in report.artifacts
    )


def test_q5_suggest_untried_with_literature(tmp_path: Path) -> None:
    """Q5: top-N with literature support; exclude already-tried (Mixup)."""
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, CAPSTONE_SLUG) as store:
        seed_capstone_store(store)

    # Micro Agents off — rule_engine path.
    result = HypothesisAssistant(llm_client=None).recommend(
        knowledge_dir=knowledge_dir,
        competition=CAPSTONE_SLUG,
        question="Suggest five experiments with strong literature support",
        pipeline=["CrossEntropy"],
        transfers=[
            {
                "repo_id": "github:owner/audio-pipeline",
                "summary": "Focal Loss vs your CE",
                "remote_choice": "Focal Loss",
                "effort": "20m",
                "expected_gain": "medium",
                "hypothesis_hint": "Swap CE → Focal Loss",
                "deltas": [],
                "interesting_files": [],
                "local_baseline": "CrossEntropy",
            }
        ],
        limit=5,
        persist=False,
        write_report=False,
        progressive=True,
    )
    assert 1 <= len(result.recommendations) <= 5
    titles = " ".join(card.title for card in result.recommendations).lower()
    tags = {
        normalize_label(tag)
        for card in result.recommendations
        for tag in card.tags
    }
    # Mixup already tried in exp:12 — must not be suggested as a new technique.
    assert "mixup" not in tags
    assert "try mixup" not in titles
    # Every card must carry grounded evidence refs.
    for card in result.recommendations:
        assert card.supporting_evidence or card.evidence
        assert card.generator.value == "rule_engine" or str(card.generator) == "rule_engine"


def test_format_text_and_json_still_write_analyze_json(tmp_path: Path) -> None:
    """`--format` only changes stdout; analyze.json always persists."""
    report, path = run_capstone_analyze(tmp_path)
    assert path.is_file()
    validate_json(path.read_text())
    # Re-write via renderer API (same contract).
    write_report(report, path)
    assert AnalysisReport.model_validate_json(path.read_text())
