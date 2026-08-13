"""Plan 6 — Research Planner capstone: offline hypothesis → DAG → DB → projections."""

from __future__ import annotations

import json
import re
from pathlib import Path

from helpers.cli import cli_runner

from labpilot.cli.main import app
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.planner.schemas.task_types import TaskType
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.planner.validator import topological_levels, validate_plan

runner = cli_runner()
_HELP_ENV = {"COLUMNS": "200", "NO_COLOR": "1"}
CAPSTONE_SLUG = "planner-capstone"


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _seed_specaugment_hypothesis(knowledge: Path) -> str:
    """Fixture hypothesis that exercises the augmentation template offline."""
    hyp = HypothesisStore(knowledge, CAPSTONE_SLUG).create(
        observation="Baseline pipeline has no SpecAugment",
        reason="SpecAugment is a common win for audio classification",
        prediction="Adding SpecAugment will improve CV vs baseline",
        confidence=0.72,
        expected_impact=0.012,
        tags=["augmentation", "specaugment"],
    )
    return hyp.id


def test_planner_capstone_offline_create_persists_dag(tmp_path: Path) -> None:
    """Given hypothesis H, emit an inspectable durable ResearchPlan without an LLM."""
    knowledge = tmp_path / "knowledge"
    hyp_id = _seed_specaugment_hypothesis(knowledge)

    create = runner.invoke(
        app,
        [
            "plan",
            "create",
            CAPSTONE_SLUG,
            "--hypothesis",
            hyp_id,
            "--format",
            "json",
            "--knowledge-dir",
            str(knowledge),
        ],
        env=_HELP_ENV,
    )
    assert create.exit_code == 0, create.output
    payload = json.loads(create.stdout)
    plan_id = payload["id"]
    assert plan_id.startswith("P-")
    assert payload["generated_by"] == "rule_engine"
    assert payload["hypothesis_id"] == hyp_id

    # Durable DB round-trip
    store = PlanStore(knowledge, CAPSTONE_SLUG)
    try:
        plan = store.get_plan(plan_id)
    finally:
        store.close()
    assert plan is not None
    validate_plan(plan)
    levels = topological_levels(plan)
    assert levels  # non-empty DAG
    assert all(level for level in levels)

    types = {task.type for task in plan.tasks}
    assert {
        TaskType.READ_CODE,
        TaskType.WRITE_CODE,
        TaskType.RUN_TRAINING,
        TaskType.EVALUATE,
        TaskType.COMPARE,
    } <= types

    # Derived projections on disk
    plans_dir = knowledge / CAPSTONE_SLUG / "research" / "plans"
    assert (plans_dir / f"{plan_id}.json").is_file()
    md = (plans_dir / f"{plan_id}.md").read_text()
    assert plan.goal in md
    assert plan.tasks[0].id in md

    # Non-goals: no execution artifacts
    assert not (knowledge / CAPSTONE_SLUG / "research" / "runs").exists()
    assert not (tmp_path / "runs").exists()


def test_planner_capstone_no_execute_flag_in_cli() -> None:
    for args in (["plan", "--help"], ["plan", "create", "--help"]):
        result = runner.invoke(app, args, env=_HELP_ENV)
        assert result.exit_code == 0, result.output
        assert "--execute" not in _plain(result.stdout)
