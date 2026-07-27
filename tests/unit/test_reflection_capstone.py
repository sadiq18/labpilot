"""Capstone: dry-run style reflect → journal green (Plan 10)."""

from __future__ import annotations

import json
from pathlib import Path

from labpilot.research_engine.reflection.journal import JournalProjector
from labpilot.research_engine.reflection.pipeline import run_reflection


def test_capstone_reflect_to_journal(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "titanic-fixture"
    workspace = tmp_path / "competitions" / competition
    workspace.mkdir(parents=True)
    (workspace / "metrics.json").write_text(
        json.dumps({"cv_accuracy": 0.79, "runtime_seconds": 8.0}),
        encoding="utf-8",
    )
    (workspace / "baseline_choice.json").write_text(
        json.dumps(
            {
                "template_name": "tabular_classification",
                "problem_type": "tabular_classification",
                "metric_name": "accuracy",
            }
        ),
        encoding="utf-8",
    )
    (workspace / "artifacts").mkdir()
    (workspace / "artifacts" / "comparison.json").write_text(
        json.dumps(
            {
                "compare_to": "P-001",
                "delta": 0.012,
                "verdict": "worth_keeping",
                "maximize": True,
                "outcome": "improved",
            }
        ),
        encoding="utf-8",
    )

    result = run_reflection(
        knowledge,
        competition,
        execution_id="E-001",
        workspace_path=workspace,
        plan_id="P-002",
        persist=True,
    )
    assert result["evidence"]["id"].startswith("EE-")
    assert result["evidence"]["strength"] in {"strong", "moderate"}
    assert result["belief"]["belief_update_id"] >= 1
    assert result["lesson"]["id"].startswith("L-")

    journal = JournalProjector(knowledge, competition)
    try:
        text = journal.render_markdown()
        assert "Research Journal" in text
        assert "Evidence" in text
        assert "Recommended next experiment" in text
        data = journal.build()
        assert data["recommended_next"]["action"]
    finally:
        journal.close()
