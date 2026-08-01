"""Conductor decision / feedback snippets (episodic session memory)."""

from __future__ import annotations

from pathlib import Path

import anyio

from labpilot.research_engine.context.models import ContextItem, ContextRequest


class EpisodicProvider:
    """Read Conductor decisions and operator feedback for a session."""

    name = "episodic"

    async def fetch(self, request: ContextRequest) -> list[ContextItem]:
        if request.knowledge_dir is None:
            return []
        return await anyio.to_thread.run_sync(self._fetch_sync, request)

    def _fetch_sync(self, request: ContextRequest) -> list[ContextItem]:
        from labpilot.research_engine.conductor.store import ConductorStore

        store = ConductorStore(Path(request.knowledge_dir), request.competition)
        try:
            session_id = request.session_id
            if not session_id:
                sessions = store.list_sessions()
                if not sessions:
                    return []
                sessions = sorted(sessions, key=lambda s: s.updated_at, reverse=True)
                session_id = sessions[0].id
            items: list[ContextItem] = []
            for d in store.list_decisions(session_id):
                text = f"{d.tool_name or '—'}: {d.rationale}".strip()
                if not text or text == "—:":
                    continue
                items.append(
                    ContextItem(
                        id=f"episodic:decision:{d.id}",
                        source=self.name,
                        kind="decision",
                        text=text[:1500],
                        score=0.55,
                        reason="conductor decision",
                        metadata={
                            "competition": request.competition,
                            "session_id": session_id,
                            "status": "stop" if d.stop else "active",
                            "tool": d.tool_name,
                        },
                    )
                )
            for fb in store.list_feedback(session_id, limit=20):
                text = f"{fb.gated_tool} {fb.decision}: {fb.comment}".strip()
                items.append(
                    ContextItem(
                        id=f"episodic:feedback:{fb.id}",
                        source=self.name,
                        kind="feedback",
                        text=text[:1500],
                        score=0.5,
                        reason="operator feedback",
                        metadata={
                            "competition": request.competition,
                            "session_id": session_id,
                            "status": fb.decision,
                            "gated_tool": fb.gated_tool,
                        },
                    )
                )
            return items
        finally:
            store.close()
