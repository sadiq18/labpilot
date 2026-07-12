"""Discover project.yaml in the working tree."""

from pathlib import Path

import yaml

from labpilot.workspace.models import ProjectConfig, ResolvedProject

PROJECT_FILENAME = "project.yaml"


def find_project_root(start: Path | None = None, project_dir: Path | None = None) -> Path | None:
    """Return the directory containing project.yaml, or None if not found."""
    if project_dir is not None:
        candidate = project_dir.resolve()
        if (candidate / PROJECT_FILENAME).is_file():
            return candidate
        return None

    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        if (directory / PROJECT_FILENAME).is_file():
            return directory
    return None


def load_project(
    start: Path | None = None,
    project_dir: Path | None = None,
) -> ResolvedProject | None:
    """Load and resolve project.yaml if present."""
    root = find_project_root(start=start, project_dir=project_dir)
    if root is None:
        return None

    raw = yaml.safe_load((root / PROJECT_FILENAME).read_text()) or {}
    project = ProjectConfig.model_validate(raw)
    return project.resolve(root)


def init_project(
    directory: Path,
    name: str,
    *,
    copy_default_config: bool = True,
) -> ResolvedProject:
    """Create project.yaml and standard directories."""
    root = directory.resolve()
    root.mkdir(parents=True, exist_ok=True)

    config_rel = Path("configs/default.yaml")
    competitions_dir = root / "competitions"
    runs_dir = root / "runs"
    cache_dir = root / ".cache" / "kaggle"
    runtimes_dir = root / "configs" / "runtimes"

    for path in (competitions_dir, runs_dir, cache_dir, runtimes_dir):
        path.mkdir(parents=True, exist_ok=True)

    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    project_config_path = configs_dir / "default.yaml"
    if copy_default_config and not project_config_path.is_file():
        repo_default = Path("configs/default.yaml")
        if repo_default.is_file():
            project_config_path.write_text(repo_default.read_text())
        else:
            project_config_path.write_text("runs_dir: runs\n")

    project = ProjectConfig(
        name=name,
        config=config_rel,
        competitions_dir=Path("competitions"),
        runs_dir=Path("runs"),
        cache_dir=Path(".cache/kaggle"),
        runtimes_dir=Path("configs/runtimes"),
    )
    (root / PROJECT_FILENAME).write_text(
        yaml.safe_dump(
            {
                "name": project.name,
                "config": str(project.config),
                "competitions_dir": str(project.competitions_dir),
                "runs_dir": str(project.runs_dir),
                "cache_dir": str(project.cache_dir),
                "runtimes_dir": str(project.runtimes_dir),
                "default_runtime": project.default_runtime,
                "competitions": [],
            },
            sort_keys=False,
        )
    )
    return project.resolve(root)
