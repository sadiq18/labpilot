"""Competition workspace discovery and scaffolding (``labpilot.yaml``).

A workspace is a client-owned folder ``<root>/<slug>/`` that holds knowledge,
pipeline code, data, and caches for one competition. CLI commands walk up from
CWD looking for ``labpilot.yaml``; when found, paths resolve relative to that
root instead of the LabPilot package tree.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from labpilot.config import AppConfig, load_config

MARKER_NAME = "labpilot.yaml"
SCHEMA_VERSION = 1

_WORKSPACE_DIRS = (
    "configs",
    "knowledge",
    "pipeline",
    "data",
    "data/raw",
    "data/processed",
    "artifacts",
    "logs",
    "models",
    ".cache",
    ".cache/kaggle",
)

_GITIGNORE = """\
# Competition data (often huge)
data/
.cache/

# Secrets
.env
.env.*

# Local DB journals / models
**/knowledge.db-journal
models/

# Python / OS
__pycache__/
*.py[cod]
.DS_Store
.venv/
venv/
"""

_README_TEMPLATE = """\
# {slug}

LabPilot competition workspace.

## Credentials (required for Kaggle)

Create a **workspace-local** ``.env`` (gitignored) — do not rely on the LabPilot repo:

```bash
cp .env.example .env
# Edit .env and set KAGGLE_API_TOKEN from https://www.kaggle.com/settings
```

## Commands (from this directory)

```bash
# Run LabPilot from your clone (v1) — use --project so CWD stays here:
uv run --project {labpilot_hint} research analyze
uv run --project {labpilot_hint} research plan create --baseline
uv run --project {labpilot_hint} research run --plan P-001
```

Or set an alias:

```bash
alias research='uv run --project {labpilot_hint} research'
```

Slug and paths are read from ``labpilot.yaml`` when your shell CWD is this folder.
"""

_ENV_EXAMPLE = """\
# Competition workspace credentials (gitignored when copied to .env)
# Create token: https://www.kaggle.com/settings → API → Create New Token
KAGGLE_API_TOKEN=

# Optional LLM (analyze / codegen / narrative)
# GEMINI_API_KEY=
# LABPILOT_LLM_PROVIDER=gemini
# OPENAI_API_KEY=
"""


class WorkspacePaths(BaseModel):
    knowledge: str = "knowledge"
    data: str = "data"
    pipeline: str = "pipeline"
    artifacts: str = "artifacts"
    cache: str = ".cache"
    config: str = "configs/default.yaml"


class CompetitionWorkspace(BaseModel):
    """Resolved competition workspace rooted at ``labpilot.yaml``."""

    root: Path
    competition: str
    paths: WorkspacePaths = Field(default_factory=WorkspacePaths)
    schema_version: int = SCHEMA_VERSION
    created_at: str | None = None

    @property
    def knowledge_dir(self) -> Path:
        return (self.root / self.paths.knowledge).resolve()

    @property
    def data_dir(self) -> Path:
        return (self.root / self.paths.data).resolve()

    @property
    def pipeline_dir(self) -> Path:
        return (self.root / self.paths.pipeline).resolve()

    @property
    def artifacts_dir(self) -> Path:
        return (self.root / self.paths.artifacts).resolve()

    @property
    def cache_dir(self) -> Path:
        return (self.root / self.paths.cache).resolve()

    @property
    def config_path(self) -> Path:
        return (self.root / self.paths.config).resolve()

    @property
    def marker_path(self) -> Path:
        return self.root / MARKER_NAME

    def code_workspace_root(self) -> Path:
        """Engineer code/data root — the slug folder itself (not competitions/)."""
        return self.root.resolve()


def _discovery_roots(start: Path | None) -> list[Path]:
    """Roots to walk for ``labpilot.yaml``.

    Prefer an explicit start. Otherwise try ``Path.cwd()`` and shell ``PWD``
    (``uv run --directory`` chdirs into the LabPilot clone while ``PWD`` often
    remains the competition folder).
    """
    if start is not None:
        return [Path(start).expanduser().resolve()]
    roots: list[Path] = []
    for candidate in (Path.cwd(), Path(os.environ["PWD"]) if os.environ.get("PWD") else None):
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def discover_workspace(
    start: Path | None = None,
    *,
    explicit: Path | None = None,
) -> CompetitionWorkspace | None:
    """Walk parents from ``start`` (default CWD / shell PWD) for ``labpilot.yaml``.

    If ``explicit`` is set, load that path (file or directory containing the marker).
    """
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        marker = path if path.is_file() else path / MARKER_NAME
        if not marker.is_file():
            raise FileNotFoundError(f"No {MARKER_NAME} at {path}")
        return load_workspace(marker)

    seen: set[Path] = set()
    for root in _discovery_roots(start):
        for candidate in [root, *root.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            marker = candidate / MARKER_NAME
            if marker.is_file():
                return load_workspace(marker)
    return None


def load_workspace(marker: Path) -> CompetitionWorkspace:
    raw = yaml.safe_load(marker.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Empty workspace marker: {marker}")
    competition = str(raw.get("competition") or "").strip()
    if not competition:
        raise ValueError(f"Workspace marker missing competition: {marker}")
    paths_raw = raw.get("paths") or {}
    paths = WorkspacePaths.model_validate(paths_raw if isinstance(paths_raw, dict) else {})
    return CompetitionWorkspace(
        root=marker.parent.resolve(),
        competition=competition,
        paths=paths,
        schema_version=int(raw.get("schema_version") or SCHEMA_VERSION),
        created_at=raw.get("created_at"),
    )


def apply_workspace_to_config(
    config: AppConfig,
    workspace: CompetitionWorkspace,
) -> AppConfig:
    """Rewrite artifact roots onto the competition workspace."""
    config.knowledge_dir = workspace.knowledge_dir
    config.runs_dir = workspace.root / "runs"
    config.kaggle.cache_dir = workspace.cache_dir / "kaggle"
    config.llm.cache.path = workspace.cache_dir / "llm.sqlite"
    return config


def load_config_for_cwd(
    *,
    config_path: Path | None = None,
    knowledge_dir: Path | None = None,
    runs_dir: Path | None = None,
    workspace_path: Path | None = None,
    start: Path | None = None,
) -> tuple[AppConfig, CompetitionWorkspace | None]:
    """Load AppConfig with workspace discovery.

    Precedence: package/repo defaults → workspace/explicit ``--config`` →
    workspace path roots → CLI knowledge/runs overrides → env (via ``load_config``).
    """
    workspace = discover_workspace(start=start, explicit=workspace_path)

    explicit = config_path
    defaultish = explicit is None or str(explicit) in {"configs/default.yaml", "."}
    if workspace is not None and defaultish and workspace.config_path.is_file():
        explicit = workspace.config_path

    config = load_config(explicit)
    if workspace is not None:
        config = apply_workspace_to_config(config, workspace)

    if knowledge_dir is not None:
        config.knowledge_dir = Path(knowledge_dir)
    if runs_dir is not None:
        config.runs_dir = Path(runs_dir)
    return config, workspace


def resolve_competition_arg(
    competition: str | None,
    workspace: CompetitionWorkspace | None,
    *,
    required: bool = True,
) -> str:
    """Default competition slug from workspace when CLI omits it."""
    if competition and competition.strip():
        slug = competition.strip()
        if workspace is not None and slug != workspace.competition:
            raise ValueError(
                f"Competition {slug!r} does not match workspace "
                f"{workspace.competition!r} at {workspace.root}"
            )
        return slug
    if workspace is not None:
        return workspace.competition
    if required:
        raise ValueError(
            "Competition slug required (pass the slug/URL, or run from a "
            f"directory containing {MARKER_NAME})."
        )
    return ""


def write_workspace_marker(workspace: CompetitionWorkspace) -> Path:
    payload: dict[str, Any] = {
        "schema_version": workspace.schema_version,
        "competition": workspace.competition,
        "created_at": workspace.created_at or datetime.now(UTC).isoformat(),
        "paths": workspace.paths.model_dump(mode="json"),
    }
    path = workspace.marker_path
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def scaffold_workspace(
    root: Path,
    competition: str,
    *,
    force: bool = False,
    labpilot_hint: str = "/path/to/labpilot",
) -> CompetitionWorkspace:
    """Create ``<root>/<slug>/`` layout + marker. Does not download data."""
    root = Path(root).expanduser().resolve()
    if root.exists():
        nonempty = any(root.iterdir())
        if nonempty and not force:
            raise FileExistsError(
                f"Workspace directory is not empty: {root}. Pass --force to reuse."
            )
    root.mkdir(parents=True, exist_ok=True)

    for rel in _WORKSPACE_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)

    workspace = CompetitionWorkspace(
        root=root,
        competition=competition,
        created_at=datetime.now(UTC).isoformat(),
    )
    write_workspace_marker(workspace)

    gitignore = root / ".gitignore"
    if not gitignore.is_file() or force:
        gitignore.write_text(_GITIGNORE, encoding="utf-8")

    readme = root / "README.md"
    if not readme.is_file() or force:
        readme.write_text(
            _README_TEMPLATE.format(slug=competition, labpilot_hint=labpilot_hint),
            encoding="utf-8",
        )

    env_example = root / ".env.example"
    if not env_example.is_file() or force:
        env_example.write_text(_ENV_EXAMPLE, encoding="utf-8")

    overlay = root / "configs" / "default.yaml"
    if not overlay.is_file() or force:
        overlay.write_text(
            "# Workspace overlay — paths are relative to this competition folder.\n"
            "knowledge_dir: knowledge\n"
            "runs_dir: runs\n"
            "kaggle:\n"
            "  cache_dir: .cache/kaggle\n"
            "llm:\n"
            "  cache:\n"
            "    path: .cache/llm.sqlite\n",
            encoding="utf-8",
        )

    return workspace


def init_git_repo(root: Path) -> None:
    """``git init`` + initial commit of scaffold (no data)."""
    root = Path(root).resolve()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=labpilot@localhost",
            "-c",
            "user.name=LabPilot",
            "commit",
            "-m",
            "Initial LabPilot competition workspace",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def competition_workspace_path(knowledge_dir: Path, competition: str) -> Path:
    """Resolve Engineer code root.

    Prefer an active workspace (slug folder). Legacy: sibling
    ``competitions/<slug>/`` next to the knowledge root.
    """
    knowledge_dir = Path(knowledge_dir).resolve()
    # Workspace layout: <slug>/knowledge/… with labpilot.yaml beside knowledge/
    marker_at_parent = knowledge_dir.parent / MARKER_NAME
    if marker_at_parent.is_file():
        ws = load_workspace(marker_at_parent)
        if ws.competition == competition:
            return ws.code_workspace_root()
    # Discover from CWD as last resort (CLI often runs with CWD = workspace)
    ws = discover_workspace()
    if ws is not None and ws.competition == competition:
        return ws.code_workspace_root()
    return knowledge_dir.parent / "competitions" / competition
