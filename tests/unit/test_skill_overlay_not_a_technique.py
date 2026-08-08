"""A record reference must not reach an agent's system prompt.

`shared/labels.py` documents this rule and two prior failed attempts at it. This
is the fifth write site, and the one the existing guards cannot reach: the
knowledge-store guards protect `techniques.name`, and the four reader-side
filters run when tags are read as techniques. Neither touches a markdown file
that is concatenated into a system prompt.

Measured on the rogii workspace 2026-08-08: all six overlays under
`.labpilot/skills/` were byte-identical and carried

    - Keep: vit
    - Keep: hyp:H-010

so every agent was told to keep a hypothesis id on every run.
"""

from __future__ import annotations

from labpilot.research_engine.execution.outcome import (
    ExecutionOutcomeSummary,
    update_competition_skill_overlays,
)

_AGENT_KEYS = (
    "code_engineer",
    "hypothesis_generator",
    "research_planner",
    "planning_engine",
    "experiment_reviewer",
    "research_brief",
)


def _overlay_text(root, agent_key: str = "code_engineer") -> str:
    path = root / ".labpilot" / "skills" / f"{agent_key}.md"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _run(root, *, techniques, gain=1.0, loss=0.0):
    update_competition_skill_overlays(
        workspace_root=root,
        summary=ExecutionOutcomeSummary(
            competition="rogii-wellbore-geology-prediction",
            plan_id="P-001",
            execution_id="E-001",
            hypothesis_id="H-010",
            learning_gain=gain,
            learning_loss=loss,
        ),
        reflection={},
        techniques=techniques,
    )


def test_a_hypothesis_id_is_not_written_as_a_technique_to_keep(tmp_path):
    _run(tmp_path, techniques=["SWA", "hyp:H-010"])
    text = _overlay_text(tmp_path)

    assert "Keep: SWA" in text
    assert "hyp:H-010" not in text


def test_a_fork_reference_is_filtered_too(tmp_path):
    """`is_record_reference` covers `fork:` as well — the guard is one rule."""
    _run(tmp_path, techniques=["fork:H-003", "EMA"])
    text = _overlay_text(tmp_path)

    assert "Keep: EMA" in text
    assert "fork:H-003" not in text


def test_the_avoid_list_is_filtered_on_the_same_rule(tmp_path):
    """A fabricated *failure* is the worse half: it teaches agents to avoid
    something that never existed. rogii recorded exactly that — a belief with
    `effect='negative'` for `hyp:H-010`."""
    _run(tmp_path, techniques=["hyp:H-010", "target_encoding"], gain=0.0, loss=1.0)
    text = _overlay_text(tmp_path)

    assert "Avoid: target_encoding" in text
    assert "hyp:H-010" not in text


def test_no_agent_receives_the_reference(tmp_path):
    """One lesson is broadcast to six agents, so one leak is six prompts."""
    _run(tmp_path, techniques=["hyp:H-010", "SWA"])

    for key in _AGENT_KEYS:
        assert "hyp:H-010" not in _overlay_text(tmp_path, key), key


def test_the_hypothesis_provenance_line_is_still_written(tmp_path):
    """The carve-out must not cost the behaviour it guards.

    "parent stack from H-010" is prose about lineage, not a technique name
    offered for reuse — filtering it would remove real advice.
    """
    _run(tmp_path, techniques=["SWA"])

    assert "parent stack from H-010" in _overlay_text(tmp_path)


def test_blank_technique_names_do_not_become_empty_bullets(tmp_path):
    _run(tmp_path, techniques=["", "   ", "SWA"])
    text = _overlay_text(tmp_path)

    assert "Keep: SWA" in text
    assert "- Keep: \n" not in text
