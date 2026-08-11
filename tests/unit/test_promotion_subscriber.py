"""The promotion subscriber, driven through the real bus (M11).

Everything here goes through `install_default_subscribers` and `bus.publish`
rather than calling the module directly, because the wiring is half the
feature: a promotion module nobody installed promotes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from labpilot.research_engine.agents.events import (
    EXPERIMENT_COMPLETED,
    MODEL_FAILED,
    EventBus,
)
from labpilot.research_engine.agents.git_evolution import write_experiment_git_record
from labpilot.research_engine.agents.promotion import cohort_path
from labpilot.research_engine.agents.subscribers import install_default_subscribers
from labpilot.workspace import scaffold_workspace


def _workspace(tmp_path: Path):
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    return client, Path(client.root), Path(client.root) / "runs"


def _competition_profile(runs_dir: Path, execution_id: str) -> None:
    """What the metric resolver reads to answer key + direction."""
    run_dir = runs_dir / execution_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "competition.json").write_text(
        json.dumps(
            {
                "slug": "titanic",
                "title": "titanic",
                "evaluation_metric": {"name": "rmse", "key": "rmse", "direction": "minimize"},
            }
        ),
        encoding="utf-8",
    )


def _payload(client, runs_dir: Path, execution_id: str, value: float, **overrides) -> dict:
    base = {
        "competition": "titanic",
        "knowledge_dir": str(client.knowledge_dir),
        "workspace_root": str(client.root),
        "runs_dir": str(runs_dir),
        "execution_id": execution_id,
        "experiment_id": f"exp_titanic_{execution_id}",
        "plan_id": "P-001",
        "status": "succeeded",
        "metrics": {"cv_rmse": value},
        "cohort_id": "C-1",
        "completed_at": "2026-08-11T10:00:00+00:00",
    }
    base.update(overrides)
    return base


def _land(bus, client, runs_dir: Path, execution_id: str, value: float, **overrides) -> None:
    write_experiment_git_record(
        runs_dir,
        {
            "experiment_id": f"exp_titanic_{execution_id}",
            "execution_id": execution_id,
            "competition": "titanic",
            "status": "succeeded",
            "metrics": {"cv_rmse": value},
            "aliases": [],
        },
    )
    _competition_profile(runs_dir, execution_id)
    bus.publish(EXPERIMENT_COMPLETED, _payload(client, runs_dir, execution_id, value, **overrides))


def test_three_branches_produce_one_promoted_winner(tmp_path: Path) -> None:
    client, _, runs = _workspace(tmp_path)
    bus = EventBus()
    install_default_subscribers(bus)

    _land(bus, client, runs, "E-1", 0.50)
    _land(bus, client, runs, "E-2", 0.20)
    _land(bus, client, runs, "E-3", 0.90)

    state = json.loads(cohort_path(runs, "C-1").read_text(encoding="utf-8"))
    assert state["promoted"] == "E-2"
    assert sorted(state["demoted"]) == ["E-1", "E-3"]
    assert state["metric_key"] == "cv_rmse"
    assert state["maximize"] is False


def test_the_direction_comes_from_the_competition_not_a_guess(tmp_path: Path) -> None:
    """The profile says minimize. Defaulting to "higher is better" would
    promote the worst branch of every error-metric competition."""
    client, _, runs = _workspace(tmp_path)
    bus = EventBus()
    install_default_subscribers(bus)

    _land(bus, client, runs, "E-1", 0.90)
    _land(bus, client, runs, "E-2", 0.20)

    state = json.loads(cohort_path(runs, "C-1").read_text(encoding="utf-8"))
    assert state["maximize"] is False
    assert state["promoted"] == "E-2"


def test_a_step_without_a_cohort_writes_nothing(tmp_path: Path) -> None:
    """The ordinary K=1 campaign step has nothing to rank against."""
    client, _, runs = _workspace(tmp_path)
    bus = EventBus()
    install_default_subscribers(bus)

    _land(bus, client, runs, "E-1", 0.5, cohort_id=None)

    assert not (runs / "experiment" / "cohorts").exists()


def test_a_failed_branch_is_not_a_candidate(tmp_path: Path) -> None:
    client, _, runs = _workspace(tmp_path)
    bus = EventBus()
    install_default_subscribers(bus)

    _land(bus, client, runs, "E-1", 0.50)
    _land(bus, client, runs, "E-2", 0.10, status="failed", error="boom")

    state = json.loads(cohort_path(runs, "C-1").read_text(encoding="utf-8"))
    assert [m["id"] for m in state["members"]] == ["E-1"]
    assert state["promoted"] == "E-1"


def test_model_failed_never_reaches_promotion(tmp_path: Path) -> None:
    """The failure event carries a cohort_id too; promotion must ignore it."""
    client, _, runs = _workspace(tmp_path)
    bus = EventBus()
    install_default_subscribers(bus)

    bus.publish(MODEL_FAILED, _payload(client, runs, "E-1", 0.1, status="failed"))

    assert not (runs / "experiment" / "cohorts").exists()


def test_a_broken_payload_does_not_fail_the_experiment(tmp_path: Path) -> None:
    """The subscriber runs on the branch's own thread — an escape here would
    surface as the experiment failing."""
    bus = EventBus()
    install_default_subscribers(bus)

    bus.publish(EXPERIMENT_COMPLETED, {"cohort_id": "../escape", "runs_dir": "/nope"})
    bus.publish(EXPERIMENT_COMPLETED, {"cohort_id": "C-1"})


def test_the_verdict_updates_as_later_branches_land(tmp_path: Path) -> None:
    client, _, runs = _workspace(tmp_path)
    bus = EventBus()
    install_default_subscribers(bus)

    _land(bus, client, runs, "E-1", 0.50)
    first = json.loads(cohort_path(runs, "C-1").read_text(encoding="utf-8"))
    assert first["promoted"] == "E-1"

    _land(bus, client, runs, "E-2", 0.20)
    second = json.loads(cohort_path(runs, "C-1").read_text(encoding="utf-8"))
    assert second["promoted"] == "E-2"
    assert second["demoted"] == ["E-1"]


def test_the_cohort_survives_its_branch_worktree_being_deleted(tmp_path: Path) -> None:
    """The verdict outlives the branches it is about."""
    import shutil

    client, ws_root, runs = _workspace(tmp_path)
    bus = EventBus()
    install_default_subscribers(bus)

    branch_root = tmp_path / "worktrees" / "b1"
    branch_root.mkdir(parents=True)
    _land(bus, client, runs, "E-1", 0.50, workspace_root=str(branch_root))
    shutil.rmtree(branch_root)

    state = json.loads(cohort_path(runs, "C-1").read_text(encoding="utf-8"))
    assert state["promoted"] == "E-1"
