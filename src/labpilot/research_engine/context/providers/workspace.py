"""Workspace reports / brief text as context candidates."""

from __future__ import annotations

from pathlib import Path

import anyio

from labpilot.accessor.common.derived import read_derived
from labpilot.research_engine.context.models import ContextItem, ContextRequest

_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}


class WorkspaceProvider:
    """Read competition research reports and brief files from disk."""

    name = "workspace"

    async def fetch(self, request: ContextRequest) -> list[ContextItem]:
        if request.knowledge_dir is None:
            return []
        return await anyio.to_thread.run_sync(self._fetch_sync, request)

    def _fetch_sync(self, request: ContextRequest) -> list[ContextItem]:
        from labpilot.research_engine.workspace_facade import Workspace

        ws = Workspace.from_competition(
            Path(request.knowledge_dir),
            request.competition,
            goal=request.goal or None,
        )
        paths = ws.research_paths
        items: list[ContextItem] = []
        reports_dir = paths.reports_dir
        if reports_dir.is_dir():
            for path in sorted(reports_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
                    continue
                try:
                    text = read_derived(path, errors="ignore").strip()
                except OSError:
                    continue
                if not text:
                    continue
                rel = path.relative_to(reports_dir).as_posix()
                items.append(
                    ContextItem(
                        id=f"workspace:report:{rel}",
                        source=self.name,
                        kind="note",
                        text=text[:4000],
                        score=0.4,
                        reason=f"workspace report {rel}",
                        metadata={
                            "competition": request.competition,
                            "path": str(path),
                            "status": "available",
                        },
                    )
                )
        brief = paths.brief_path
        if brief.is_file():
            try:
                text = read_derived(brief, errors="ignore").strip()
            except OSError:
                text = ""
            if text:
                items.append(
                    ContextItem(
                        id="workspace:brief",
                        source=self.name,
                        kind="brief",
                        text=text[:4000],
                        score=0.6,
                        reason="research brief",
                        metadata={
                            "competition": request.competition,
                            "path": str(brief),
                            "status": "available",
                        },
                    )
                )
        return items
