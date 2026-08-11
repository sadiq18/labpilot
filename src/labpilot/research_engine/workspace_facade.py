"""Research OS Workspace facade over competition + knowledge layouts.

Wraps client-owned ``labpilot.yaml`` workspaces and the legacy
``knowledge/`` + ``competitions/<slug>/`` layout behind one handle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, PrivateAttr

from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.workspace import (
    MARKER_NAME,
    CompetitionWorkspace,
    WorkspacePaths,
    competition_workspace_path,
    discover_workspace,
    ensure_required_ignores,
    load_workspace,
)

LayoutKind = Literal["client", "legacy"]

# Legacy layout has no marker to read relative names from, and its hardcoded
# ones have always matched these defaults.
_DEFAULT_PATHS = WorkspacePaths()


class Workspace(BaseModel):
    """Stable path handle for tools and (later) orchestrators.

    Prefer constructing via :meth:`from_competition`, :meth:`from_cwd`, or
    :meth:`from_client`. Slug identifies the competition *inside* the workspace.
    """

    competition: str
    knowledge_dir: Path
    root: Path
    layout: LayoutKind = "legacy"
    goal: str | None = None
    runs_dir: Path | None = None
    data_dir_override: Path | None = None
    cache_dir_override: Path | None = None

    _client: CompetitionWorkspace | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_client(
        cls,
        client: CompetitionWorkspace,
        *,
        goal: str | None = None,
        runs_dir: Path | None = None,
    ) -> Workspace:
        """Build from a discovered / loaded :class:`CompetitionWorkspace`."""
        ws = cls(
            competition=client.competition,
            knowledge_dir=client.knowledge_dir,
            root=client.code_workspace_root(),
            layout="client",
            goal=goal,
            runs_dir=runs_dir or (client.root / "runs"),
        )
        ws._client = client
        return ws

    @classmethod
    def from_competition(
        cls,
        knowledge_dir: Path,
        competition: str,
        *,
        goal: str | None = None,
        runs_dir: Path | None = None,
        code_root: Path | None = None,
    ) -> Workspace:
        """Resolve paths for a competition under a knowledge root.

        Detects client-owned layout when ``labpilot.yaml`` sits beside
        ``knowledge_dir``; otherwise uses legacy ``competitions/<slug>/``.
        """
        knowledge_dir = Path(knowledge_dir).resolve()
        competition = competition.strip()
        marker = knowledge_dir.parent / MARKER_NAME
        if marker.is_file():
            client = load_workspace(marker)
            if client.competition == competition:
                return cls.from_client(client, goal=goal, runs_dir=runs_dir)

        root = Path(code_root) if code_root is not None else competition_workspace_path(
            knowledge_dir, competition
        )
        # competition_workspace_path may still find a CWD client workspace.
        discovered = discover_workspace()
        if (
            discovered is not None
            and discovered.competition == competition
            and discovered.code_workspace_root() == Path(root).resolve()
        ):
            return cls.from_client(discovered, goal=goal, runs_dir=runs_dir)

        root = Path(root).resolve()
        layout: LayoutKind = "legacy"
        if (root / MARKER_NAME).is_file():
            try:
                client = load_workspace(root / MARKER_NAME)
                if client.competition == competition:
                    return cls.from_client(client, goal=goal, runs_dir=runs_dir)
            except ValueError:
                pass
            layout = "client"

        return cls(
            competition=competition,
            knowledge_dir=knowledge_dir,
            root=root,
            layout=layout,
            goal=goal,
            runs_dir=runs_dir or (knowledge_dir.parent / "runs"),
        )

    @classmethod
    def from_cwd(
        cls,
        start: Path | None = None,
        *,
        knowledge_dir: Path | None = None,
        competition: str | None = None,
        goal: str | None = None,
        runs_dir: Path | None = None,
    ) -> Workspace:
        """Build from CWD / ``PWD`` discovery, or legacy paths when no marker.

        When ``labpilot.yaml`` is found, ``competition`` must match if provided.
        Without a marker, ``competition`` is required and ``knowledge_dir``
        defaults to ``<start or CWD>/knowledge``.
        """
        client = discover_workspace(start=start)
        if client is not None:
            if competition is not None and competition != client.competition:
                raise ValueError(
                    f"competition {competition!r} does not match workspace "
                    f"{client.competition!r} at {client.root}"
                )
            kd = Path(knowledge_dir).resolve() if knowledge_dir else client.knowledge_dir
            ws = cls.from_client(client, goal=goal, runs_dir=runs_dir)
            if knowledge_dir is not None:
                ws = ws.model_copy(update={"knowledge_dir": kd})
                ws._client = client
            return ws

        if not competition:
            raise ValueError(
                "No labpilot.yaml found; pass competition= for legacy layout"
            )
        base = Path(start or Path.cwd()).resolve()
        kd = Path(knowledge_dir).resolve() if knowledge_dir else (base / "knowledge")
        return cls.from_competition(kd, competition, goal=goal, runs_dir=runs_dir)

    def for_branch(self, code_root: Path) -> Workspace:
        """Copy whose *code* lives in a per-branch worktree (M11).

        Code paths follow ``code_root``: ``root``, ``pipeline_dir`` and
        ``artifacts_dir`` become branch-private, which is the isolation a
        worktree exists to give. Everything shared stays pinned where the
        campaign put it — ``data_dir`` and ``cache_dir`` so K branches don't
        each re-download a competition into a worktree that is about to be
        deleted, ``effective_runs_dir`` so a branch's experiment record
        outlives the branch (see `agents/experiment.py`).

        Pins resolve before the copy, so branching a branch keeps the original
        shared locations rather than compounding.
        """
        return self.model_copy(
            update={
                "root": Path(code_root).resolve(),
                "data_dir_override": self.data_dir,
                "cache_dir_override": self.cache_dir,
                "runs_dir": self.effective_runs_dir,
            }
        )

    @property
    def _paths(self) -> WorkspacePaths:
        """Relative directory names this workspace's layout uses."""
        return self._client.paths if self._client is not None else _DEFAULT_PATHS

    def _under_root(self, relative: str) -> Path:
        """Resolve a layout-relative name against the current code root.

        Against ``self.root`` rather than ``self._client.root``: the two are the
        same until :meth:`for_branch` moves the code into a worktree, and a path
        that kept pointing at the client root would send every branch's writes
        back into the shared workspace the worktree exists to protect.
        """
        return (self.root / relative).resolve()

    @property
    def data_dir(self) -> Path:
        """Competition data root (``data/``).

        Follows ``data_dir_override`` when pinned — see :meth:`for_branch`.
        """
        if self.data_dir_override is not None:
            return Path(self.data_dir_override).resolve()
        return self._under_root(self._paths.data)

    @property
    def raw_data_dir(self) -> Path:
        """Downloaded competition data (``data/raw``)."""
        return self.data_dir / "raw"

    @property
    def pipeline_dir(self) -> Path:
        """Pipeline / code package root."""
        return self._under_root(self._paths.pipeline)

    @property
    def artifacts_dir(self) -> Path:
        """Workspace artifacts root (submissions, copies)."""
        return self._under_root(self._paths.artifacts)

    @property
    def cache_dir(self) -> Path:
        """Local cache root.

        Follows ``cache_dir_override`` when pinned — see :meth:`for_branch`.
        """
        if self.cache_dir_override is not None:
            return Path(self.cache_dir_override).resolve()
        return self._under_root(self._paths.cache)

    @property
    def research_paths(self) -> ResearchPaths:
        """Canonical research tree under ``knowledge_dir`` / competition."""
        return ResearchPaths(self.knowledge_dir, self.competition).ensure()

    @property
    def effective_runs_dir(self) -> Path:
        """Runs directory used by analyze context builders."""
        if self.runs_dir is not None:
            return Path(self.runs_dir).resolve()
        if self._client is not None:
            return (self.root / "runs").resolve()
        return (self.knowledge_dir.parent / "runs").resolve()

    def ensure_roots(self) -> Workspace:
        """Create common workspace directories if missing; return self.

        Also reconciles the machine-local ignore patterns (M11) — this is the
        only path that runs against an *existing* workspace, so it is where a
        newly-added lock/temp pattern can actually reach the workspaces that
        generate those files. `scaffold_workspace` only covers fresh ones.
        """
        self.research_paths.ensure()
        for path in (
            self.root,
            self.data_dir,
            self.pipeline_dir,
            self.artifacts_dir,
            self.cache_dir,
            self.effective_runs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        ensure_required_ignores(self.root)
        return self
