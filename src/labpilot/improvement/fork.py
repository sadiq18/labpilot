import shutil
from pathlib import Path

from labpilot.config import AppConfig
from labpilot.experiments.graph import capture_git_commit
from labpilot.orchestrator.manifest import (
    RunManifest,
    StageStatus,
    generate_run_id,
    load_manifest,
    save_manifest,
)

# Init artifacts copied from the parent run. Downstream stages re-execute.
COPY_FILES = (
    "competition.json",
    "profile.json",
    "profile.md",
    "brief.md",
    "baseline_choice.json",
)

# Stages whose artifacts were copied or inherited — marked completed in the child manifest.
PRECOMPLETED_STAGES = (
    "parse_competition",
    "download_data",
    "profile_dataset",
    "generate_brief",
    "select_baseline",
)


def fork_run(
    parent_run_dir: Path,
    runs_dir: Path,
    *,
    parent_run_id: str | None = None,
    improvement_strategy: str = "auto",
    config: AppConfig | None = None,
) -> tuple[str, Path]:
    """Fork a parent run directory into a new child run with lineage metadata."""
    parent_run_dir = parent_run_dir.resolve()
    parent_manifest = load_manifest(parent_run_dir)
    if parent_manifest.status != StageStatus.COMPLETED:
        raise ValueError(
            f"Parent run '{parent_manifest.run_id}' must be completed before improving "
            f"(status: {parent_manifest.status.value})."
        )

    resolved_parent_id = parent_run_id or parent_manifest.run_id
    parent_iteration = int(parent_manifest.metadata.get("iteration", 0))
    child_iteration = parent_iteration + 1

    child_run_id = generate_run_id(parent_manifest.competition)
    child_run_dir = (runs_dir / child_run_id).resolve()
    if child_run_dir.exists():
        raise FileExistsError(f"Child run directory already exists: {child_run_dir}")
    child_run_dir.mkdir(parents=True)

    for name in COPY_FILES:
        source = parent_run_dir / name
        if source.is_file():
            shutil.copy2(source, child_run_dir / name)

    parent_data = parent_run_dir / "data"
    if parent_data.is_dir():
        shutil.copytree(parent_data, child_run_dir / "data")

    # The child's own resolved config, not copied from the parent — a fork
    # can be re-planned under a different config (e.g. a later `improve()`
    # call after `configs/default.yaml` changed).
    if config is not None:
        try:
            (child_run_dir / "config.json").write_text(config.model_dump_json(indent=2))
        except OSError:
            pass

    child_manifest = RunManifest(
        run_id=child_run_id,
        competition=parent_manifest.competition,
        status=StageStatus.RUNNING,
        stages=[],
        metadata={
            "parent_run_id": resolved_parent_id,
            "iteration": child_iteration,
            "improvement_strategy": improvement_strategy,
            "git_commit": capture_git_commit(),
        },
    )

    for stage_name in PRECOMPLETED_STAGES:
        artifacts: list[str] = []
        if stage_name == "parse_competition":
            artifacts = [str(child_run_dir / "competition.json")]
        elif stage_name == "download_data":
            artifacts = [str(child_run_dir / "data")]
        elif stage_name == "profile_dataset":
            artifacts = [str(child_run_dir / "profile.json"), str(child_run_dir / "profile.md")]
        elif stage_name == "generate_brief":
            artifacts = [str(child_run_dir / "brief.md")]
        elif stage_name == "select_baseline":
            artifacts = [str(child_run_dir / "baseline_choice.json")]
        child_manifest.mark_completed(stage_name, artifacts)

    save_manifest(child_run_dir, child_manifest)
    return child_run_id, child_run_dir
