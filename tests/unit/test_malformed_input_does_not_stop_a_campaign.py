"""Two ways a campaign could stall on input it should have shrugged off.

Both are the same shape as the defects this branch already fixes: a check whose
answer is wrong for a case nobody pictured, and the campaign pays in steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.research_engine.conductor.policy import offline_next_action

# --- a feedback row that names no tool ---------------------------------------


def _observe(feedback: list[dict]) -> dict:
    return {"completed_tools": [], "operator_feedback": feedback}


def test_a_rejection_without_a_tool_name_does_not_raise():
    """`decision` was read with `.get` and `gated_tool` by subscript, so a row
    carrying the first and not the second passed the filter and then raised."""
    action = offline_next_action(
        _observe([{"decision": "reject"}]),
        {"analyze_competition", "submit"},
    )

    assert action.tool


def test_a_named_rejection_is_still_honoured():
    """The carve-out must not cost the behaviour it guards."""
    action = offline_next_action(
        _observe([{"decision": "reject", "gated_tool": "submit"}]),
        {"submit"},
    )

    assert action.tool != "submit"


def test_a_missing_name_does_not_reject_an_unnamed_tool():
    """`None` in the rejected set must not match anything real."""
    action = offline_next_action(
        _observe([{"decision": "reject"}, {"decision": "reject", "gated_tool": "submit"}]),
        {"analyze_competition"},
    )

    assert action.tool == "analyze_competition"


# --- a training script that is not there -------------------------------------


def _engineer(tmp_path: Path):
    from labpilot.research_engine.execution.engineer import ResearchEngineer

    engineer = ResearchEngineer.__new__(ResearchEngineer)
    engineer.knowledge_dir = tmp_path / "knowledge"  # type: ignore[attr-defined]
    engineer.competition = "rogii-wellbore-geology-prediction"  # type: ignore[attr-defined]
    return engineer


def test_a_missing_train_script_counts_as_unrunnable(tmp_path, monkeypatch):
    """Returning False here answered "yes, it runs" for a file that is not
    there: `write_code` stayed `done`, the retry skipped it, and the plan
    re-ran nothing — the rebuild-never-happens loop by another door."""
    monkeypatch.setattr(
        "labpilot.research_engine.execution.engineer.competition_workspace_path",
        lambda *_: tmp_path,
    )
    assert _engineer(tmp_path)._train_script_is_unrunnable() is True


def test_a_present_and_valid_script_is_runnable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "labpilot.research_engine.execution.engineer.competition_workspace_path",
        lambda *_: tmp_path,
    )
    script = tmp_path / "pipeline" / "train.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        '"""B."""\n\n\ndef main():\n    pass\n\n\nif __name__ == "__main__":\n    main()\n',
        encoding="utf-8",
    )

    assert _engineer(tmp_path)._train_script_is_unrunnable() is False


@pytest.mark.parametrize(
    "body",
    [
        '"""B."""\n\n# /// script\n# requires-python = \\\n',  # truncated
        '"""B."""\n\n\ndef main():\n    pass\n',  # no entry point
    ],
)
def test_a_broken_script_on_disk_is_unrunnable(tmp_path, monkeypatch, body):
    monkeypatch.setattr(
        "labpilot.research_engine.execution.engineer.competition_workspace_path",
        lambda *_: tmp_path,
    )
    script = tmp_path / "pipeline" / "train.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")

    assert _engineer(tmp_path)._train_script_is_unrunnable() is True
