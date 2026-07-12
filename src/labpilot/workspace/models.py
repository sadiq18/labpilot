"""Project workspace models for multi-competition LabPilot projects."""

from pathlib import Path

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Configuration loaded from project.yaml at a project root."""

    name: str
    config: Path = Path("configs/default.yaml")
    competitions_dir: Path = Path("competitions")
    runs_dir: Path = Path("runs")
    cache_dir: Path = Path(".cache/kaggle")
    runtimes_dir: Path = Path("configs/runtimes")
    default_runtime: str = "local-default"
    competitions: list[str] = Field(default_factory=list)

    def resolve(self, project_root: Path) -> "ResolvedProject":
        """Resolve relative paths against the project root."""
        return ResolvedProject(
            root=project_root,
            name=self.name,
            config_path=_resolve_path(project_root, self.config),
            competitions_dir=_resolve_path(project_root, self.competitions_dir),
            runs_dir=_resolve_path(project_root, self.runs_dir),
            cache_dir=_resolve_path(project_root, self.cache_dir),
            runtimes_dir=_resolve_path(project_root, self.runtimes_dir),
            default_runtime=self.default_runtime,
            competitions=list(self.competitions),
        )


class ResolvedProject(BaseModel):
    """Fully resolved project paths."""

    root: Path
    name: str
    config_path: Path
    competitions_dir: Path
    runs_dir: Path
    cache_dir: Path
    runtimes_dir: Path
    default_runtime: str
    competitions: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


def _resolve_path(root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return (root / value).resolve()
