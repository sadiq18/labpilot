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
    competition_workspace_path,
    discover_workspace,
    load_workspace,
)

LayoutKind = Literal["client", "legacy"]


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

    @property
    def data_dir(self) -> Path:
        """Competition data root (``data/``)."""
        if self._client is not None:
            return self._client.data_dir
        return (self.root / "data").resolve()

    @property
    def raw_data_dir(self) -> Path:
        """Downloaded competition data (``data/raw``)."""
        return self.data_dir / "raw"

    @property
    def pipeline_dir(self) -> Path:
        """Pipeline / code package root."""
        if self._client is not None:
            return self._client.pipeline_dir
        return (self.root / "pipeline").resolve()

    @property
    def artifacts_dir(self) -> Path:
        """Workspace artifacts root (submissions, copies)."""
        if self._client is not None:
            return self._client.artifacts_dir
        return (self.root / "artifacts").resolve()

    @property
    def cache_dir(self) -> Path:
        """Local cache root."""
        if self._client is not None:
            return self._client.cache_dir
        return (self.root / ".cache").resolve()

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
        """Create common workspace directories if missing; return self."""
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
        return self
