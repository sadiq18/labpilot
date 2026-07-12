from pathlib import Path

import yaml

from labpilot.config import load_config
from labpilot.workspace.discover import find_project_root, init_project, load_project


def test_init_project_creates_layout(tmp_path: Path):
    project = init_project(tmp_path / "my-project", "kaggle-2026", copy_default_config=False)
    assert (project.root / "project.yaml").is_file()
    assert project.runs_dir.is_dir()
    assert project.competitions_dir.is_dir()
    assert project.runtimes_dir.is_dir()


def test_find_project_root_walks_up(tmp_path: Path, monkeypatch):
    project = init_project(tmp_path / "proj", "demo", copy_default_config=False)
    nested = project.root / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert find_project_root() == project.root


def test_load_config_merges_project_overrides(tmp_path: Path, monkeypatch):
    project = init_project(tmp_path / "proj", "demo", copy_default_config=True)
    project_config = project.config_path
    data = yaml.safe_load(project_config.read_text())
    data["training"] = {"cv_folds": 3}
    project_config.write_text(yaml.safe_dump(data))

    monkeypatch.chdir(project.root)
    config = load_config(project_dir=project.root)
    assert config.training.cv_folds == 3
    assert config.runs_dir == project.runs_dir

    loaded = load_project(project_dir=project.root)
    assert loaded is not None
    assert loaded.name == "demo"
