import json
from pathlib import Path

from labpilot.experiments.graph import assemble_experiment, build_graph
from labpilot.experiments.store import ExperimentRecord, ExperimentStore

# Mirrors experiments/graph.py's `_ALL_PIPELINE_STAGES` — kept as a literal
# copy here rather than importing the private constant, so the fixture
# doesn't silently track internal renames.
_ALL_STAGES = [
    "parse_competition",
    "download_data",
    "profile_dataset",
    "generate_brief",
    "select_baseline",
    "generate_code",
    "train_model",
    "evaluate_cv",
    "generate_submission",
    "export_kernel",
    "upload_submission",
    "log_experiment",
    "write_reflection",
    "write_report",
]


def _stage(name: str, status: str = "completed") -> dict:
    return {
        "name": name,
        "status": status,
        "started_at": "2026-01-01T00:00:00",
        "finished_at": "2026-01-01T00:05:00",
        "error": None,
        "artifacts": [],
    }


def _seed_run(
    runs_dir: Path,
    run_id: str,
    *,
    competition: str = "titanic",
    parent_id: str | None = None,
    iteration: int = 0,
    metrics: dict[str, float] | None = None,
    created_at: str = "2026-01-01T00:00:00",
    rationale: str = "",
    strategy: str = "auto",
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)

    metadata: dict = {"iteration": iteration}
    if parent_id is not None:
        metadata["parent_run_id"] = parent_id
        metadata["improvement_strategy"] = strategy

    manifest = {
        "run_id": run_id,
        "competition": competition,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "completed",
        "stages": [_stage(name) for name in _ALL_STAGES],
        "metadata": metadata,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "baseline_choice.json").write_text(
        json.dumps(
            {
                "problem_type": "tabular_classification",
                "template_name": "lgbm_tabular",
                "rationale": "baseline",
            }
        )
    )
    if parent_id is not None:
        (run_dir / "improvement_plan.json").write_text(
            json.dumps(
                {
                    "parent_run_id": parent_id,
                    "strategy": strategy,
                    "rationale": rationale,
                }
            )
        )
    ExperimentStore(run_dir).save(
        ExperimentRecord(run_id=run_id, competition=competition, metrics=metrics or {})
    )
    return run_dir


def _build_fixture(tmp_path: Path) -> Path:
    """root-1 -> {child-a, child-b}; child-a -> grandchild-1.

    Simulates two `improve` runs off the baseline, then a further `improve`
    off the better of the two children.
    """
    runs_dir = tmp_path / "runs"
    _seed_run(runs_dir, "root-1", metrics={"cv_accuracy": 0.80}, created_at="2026-01-01T00:00:00")
    _seed_run(
        runs_dir,
        "child-a",
        parent_id="root-1",
        iteration=1,
        metrics={"cv_accuracy": 0.85},
        rationale="Add mixup augmentation",
        created_at="2026-01-01T01:00:00",
    )
    _seed_run(
        runs_dir,
        "child-b",
        parent_id="root-1",
        iteration=1,
        metrics={"cv_accuracy": 0.82},
        rationale="Add focal loss",
        created_at="2026-01-01T02:00:00",
    )
    _seed_run(
        runs_dir,
        "grandchild-1",
        parent_id="child-a",
        iteration=2,
        metrics={"cv_accuracy": 0.90},
        rationale="Add EMA",
        created_at="2026-01-01T03:00:00",
    )
    return runs_dir


def test_roots_children_ancestors_descendants(tmp_path):
    runs_dir = _build_fixture(tmp_path)
    graph = build_graph(runs_dir, "titanic")

    assert {exp.id for exp in graph.roots} == {"root-1"}
    assert {exp.id for exp in graph.children("root-1")} == {"child-a", "child-b"}
    assert [exp.id for exp in graph.ancestors("grandchild-1")] == ["child-a", "root-1"]
    assert {exp.id for exp in graph.descendants("root-1")} == {
        "child-a",
        "child-b",
        "grandchild-1",
    }
    assert graph.children("grandchild-1") == []
    assert graph.ancestors("root-1") == []


def test_best_path_picks_higher_scoring_branch(tmp_path):
    runs_dir = _build_fixture(tmp_path)
    graph = build_graph(runs_dir, "titanic")

    path = graph.best_path("cv_accuracy")
    assert [exp.id for exp in path] == ["root-1", "child-a", "grandchild-1"]

    # A metric no experiment reports yields an empty path, not a crash.
    assert graph.best_path("no_such_metric") == []


def test_experiment_artifacts_includes_late_stage_files(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _seed_run(runs_dir, "root-1", metrics={"cv_accuracy": 0.8})
    (run_dir / "reflection.md").write_text("# Reflection")
    (run_dir / "report.html").write_text("<html></html>")

    experiment = assemble_experiment(run_dir)
    assert str(run_dir / "reflection.md") in experiment.artifacts
    assert str(run_dir / "report.html") in experiment.artifacts
    assert experiment.reflection_path == str(run_dir / "reflection.md")
    assert experiment.report_path == str(run_dir / "report.html")


def test_progress_and_description_for_root_and_child(tmp_path):
    runs_dir = _build_fixture(tmp_path)
    root = assemble_experiment(runs_dir / "root-1")
    child = assemble_experiment(runs_dir / "child-a")

    assert root.progress == "14/14 stages"
    assert "lgbm_tabular" in root.description
    assert "titanic" in root.description

    assert child.progress == "14/14 stages"
    assert child.description == "Add mixup augmentation"


def test_config_snapshot_roundtrips_and_defaults_empty(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _seed_run(runs_dir, "root-1", metrics={"cv_accuracy": 0.8})

    no_config = assemble_experiment(run_dir)
    assert no_config.config_snapshot == {}

    (run_dir / "config.json").write_text(json.dumps({"llm": {"provider": "openai"}}))
    with_config = assemble_experiment(run_dir)
    assert with_config.config_snapshot == {"llm": {"provider": "openai"}}


def test_no_backfill_for_pre_milestone_2_runs(tmp_path):
    """Resolves Plan 1's open question 1 ('optional everywhere, no migration'):
    a run whose manifest predates git_commit tracking has no such key at all
    (not even `null`) — reading it must report `None` and must never write
    anything back to `manifest.json` on disk."""
    runs_dir = tmp_path / "runs"
    run_dir = _seed_run(runs_dir, "root-1", metrics={"cv_accuracy": 0.8})
    manifest_path = run_dir / "manifest.json"

    manifest_data = json.loads(manifest_path.read_text())
    assert "git_commit" not in manifest_data["metadata"]
    before_bytes = manifest_path.read_bytes()
    before_mtime = manifest_path.stat().st_mtime_ns

    experiment = assemble_experiment(run_dir)
    assert experiment.git_commit is None

    graph = build_graph(runs_dir, "titanic")
    assert graph.nodes["root-1"].git_commit is None

    assert manifest_path.read_bytes() == before_bytes
    assert manifest_path.stat().st_mtime_ns == before_mtime


def test_progress_for_partial_manifest_with_running_stage(tmp_path):
    """A run that's still in flight (some stages completed/skipped, one
    `running`, and the rest never reached) must report a partial progress
    string instead of erroring — this is the "mid-run" case `progress` is
    explicitly designed to answer, distinct from `status`."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "in-flight"
    run_dir.mkdir(parents=True)

    stages = [
        _stage("parse_competition", status="completed"),
        _stage("download_data", status="completed"),
        _stage("profile_dataset", status="skipped"),
        _stage("generate_brief", status="running"),
        # Every later stage is simply absent from `manifest.stages` — never
        # reached yet, not "pending" records.
    ]
    manifest = {
        "run_id": "in-flight",
        "competition": "titanic",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "status": "running",
        "stages": stages,
        "metadata": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    experiment = assemble_experiment(run_dir)
    # 2 completed + 1 skipped = 3 "done"; total stays the full 14-stage
    # pipeline (no config.json override), not len(manifest.stages).
    assert experiment.progress == "3/14 stages"
    assert experiment.status == "running"
