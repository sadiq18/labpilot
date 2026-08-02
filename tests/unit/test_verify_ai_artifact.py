"""Post-AI verify_ai_artifact gates (analyze + reflection soft reject)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from labpilot.research_engine.artifacts.analysis import read_analysis
from labpilot.research_engine.intelligence.analyzers.repositories import (
    _ground_terms,
    _grounding_corpus,
)
from labpilot.research_engine.intelligence.models import AnalysisReport
from labpilot.research_engine.intelligence.repositories.models import Repository
from labpilot.research_engine.shared.verify_artifact import (
    VerifyResult,
    auto_approve_artifact,
    verify_ai_artifact,
)
from labpilot.research_engine.tools.handlers.analyze import analyze_competition
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def test_verify_auto_approve_default() -> None:
    result = verify_ai_artifact("analysis_report", {"n": 1})
    assert result.decision == "approve"
    assert result.kind == "analysis_report"


def test_verify_custom_prompt_reject_and_spot_check() -> None:
    reject = verify_ai_artifact(
        "analysis_report",
        {},
        auto=False,
        prompt=lambda kind, payload: VerifyResult(decision="reject", kind=kind, comment="nope"),
    )
    assert reject.decision == "reject"
    assert reject.comment == "nope"

    spot = verify_ai_artifact(
        "analysis_report",
        {},
        prompt=lambda kind, _p: VerifyResult(decision="spot_check", kind=kind),
    )
    assert spot.decision == "spot_check"
    assert auto_approve_artifact("x").decision == "approve"


def test_analyze_reject_skips_write(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    client = scaffold_workspace(tmp_path / "demo", "demo")
    ws = Workspace.from_client(client).ensure_roots()
    report = AnalysisReport(competition={"slug": "demo"}, analyzers=["competition"])

    class _Orch:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            del args, kwargs

        def analyze(self, context, **kwargs):  # noqa: ANN001, ANN003
            del context, kwargs
            return report

    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.analyze.AnalyzeOrchestrator",
        _Orch,
    )
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.analyze.build_default_registry",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.analyze.build_context",
        lambda *a, **k: MagicMock(
            report_path=ws.knowledge_dir / "analyze.json",
            paths=MagicMock(brief_path=ws.knowledge_dir / "brief.md"),
        ),
    )

    result = analyze_competition(
        ws,
        ingest_knowledge=False,
        hypothesize=False,
        brief=False,
        verify=lambda kind, _p: VerifyResult(decision="reject", kind=kind),
    )
    assert result.data["written"] is False
    assert result.refs == []
    assert read_analysis(ws.knowledge_dir, ws.competition) is None


def test_analyze_spot_check_marks_needs_review(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    client = scaffold_workspace(tmp_path / "demo", "demo")
    ws = Workspace.from_client(client).ensure_roots()
    report = AnalysisReport(competition={"slug": "demo"}, analyzers=["competition"])

    class _Orch:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            del args, kwargs

        def analyze(self, context, **kwargs):  # noqa: ANN001, ANN003
            del context, kwargs
            return report

    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.analyze.AnalyzeOrchestrator",
        _Orch,
    )
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.analyze.build_default_registry",
        lambda: MagicMock(),
    )
    report_path = ws.research_paths.ensure().report_path
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.analyze.build_context",
        lambda *a, **k: MagicMock(
            report_path=report_path,
            paths=MagicMock(brief_path=ws.knowledge_dir / "brief.md"),
        ),
    )

    result = analyze_competition(
        ws,
        ingest_knowledge=False,
        hypothesize=False,
        brief=False,
        verify=lambda kind, _p: VerifyResult(decision="spot_check", kind=kind, comment="look"),
    )
    assert result.data["written"] is True
    assert result.data["needs_review"] is True
    loaded = read_analysis(ws.knowledge_dir, ws.competition)
    assert loaded is not None
    assert loaded.summary.get("needs_review") is True
    assert any("needs_review" in n for n in loaded.notes)


def test_reflection_reject_skips_belief_and_claims(tmp_path: Path) -> None:
    from labpilot.research_engine.reflection.pipeline import run_reflection

    knowledge = tmp_path / "knowledge"
    competition = "demo"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "metrics.json").write_text(
        json.dumps({"cv_accuracy": 0.8}),
        encoding="utf-8",
    )
    (workspace / "baseline_choice.json").write_text(
        json.dumps(
            {
                "template_name": "tabular_classification",
                "problem_type": "tabular_classification",
            }
        ),
        encoding="utf-8",
    )
    (workspace / "artifacts").mkdir()
    (workspace / "artifacts" / "comparison.json").write_text(
        json.dumps({"delta": 0.02, "verdict": "worth_keeping", "maximize": True}),
        encoding="utf-8",
    )

    result = run_reflection(
        knowledge,
        competition,
        workspace_path=workspace,
        execution_id="E-verify",
        persist=True,
        verify=lambda kind, _p: VerifyResult(decision="reject", kind=kind),
    )
    assert result["verification"]["decision"] == "reject"
    assert result["belief"].get("skipped") is True
    assert result["claims"] == []
    assert result["lesson"]  # lessons still generated
    assert "assessment" in result


def test_ground_terms_drops_ungrounded_architecture() -> None:
    repo = Repository(
        id="github:o/r",
        full_name="o/r",
        url="https://github.com/o/r",
        readme_excerpt="Uses EfficientNet and mixup for audio.",
        key_files=["train.py"],
        dependencies=["torch", "timm"],
        file_texts={"train.py": "model = efficientnet_b0()\n# mixup batch"},
    )
    corpus = _grounding_corpus(repo)
    kept = _ground_terms(
        ["EfficientNet", "ResNet50", "mixup", "quantum-gan"],
        corpus,
    )
    assert "EfficientNet" in kept
    assert "mixup" in kept
    assert "ResNet50" not in kept
    assert "quantum-gan" not in kept
