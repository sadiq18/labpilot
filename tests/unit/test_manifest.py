from pathlib import Path

from labpilot.orchestrator.manifest import (
    RunManifest,
    StageStatus,
    generate_run_id,
    load_manifest,
    save_manifest,
)


def test_generate_run_id():
    run_id = generate_run_id("titanic")
    assert "titanic" in run_id


def test_manifest_lifecycle(tmp_path: Path):
    run_dir = tmp_path / "test-run"
    manifest = RunManifest(run_id="test-run", competition="titanic")

    manifest.mark_running("parse_competition")
    manifest.mark_completed("parse_competition", ["competition.json"])

    save_manifest(run_dir, manifest)
    loaded = load_manifest(run_dir)

    assert loaded.run_id == "test-run"
    assert loaded.stage("parse_competition").status == StageStatus.COMPLETED
    assert loaded.stage("parse_competition").artifacts == ["competition.json"]


def test_manifest_failure(tmp_path: Path):
    run_dir = tmp_path / "failed-run"
    manifest = RunManifest(run_id="failed-run", competition="titanic")

    manifest.mark_running("train_model")
    manifest.mark_failed("train_model", "CUDA OOM")

    save_manifest(run_dir, manifest)
    loaded = load_manifest(run_dir)

    assert loaded.status == StageStatus.FAILED
    assert loaded.stage("train_model").error == "CUDA OOM"
