"""Experiment / evidence summaries as context candidates."""

from __future__ import annotations

from pathlib import Path

import anyio

from labpilot.research_engine.context.models import ContextItem, ContextRequest


class ExperimentProvider:
    """Evidence cards and knowledge artifacts typed as experiments."""

    name = "experiments"

    async def fetch(self, request: ContextRequest) -> list[ContextItem]:
        if request.knowledge_dir is None:
            return []
        return await anyio.to_thread.run_sync(self._fetch_sync, request)

    def _fetch_sync(self, request: ContextRequest) -> list[ContextItem]:
        knowledge_dir = Path(request.knowledge_dir)
        items: list[ContextItem] = []
        items.extend(self._from_evidence(knowledge_dir, request.competition))
        items.extend(self._from_knowledge(knowledge_dir, request.competition))
        return items

    def _from_evidence(self, knowledge_dir: Path, competition: str) -> list[ContextItem]:
        try:
            from labpilot.research_engine.artifacts.evidence import EvidenceArtifacts
        except ImportError:
            return []
        try:
            cards = EvidenceArtifacts(knowledge_dir, competition).list()
        except Exception:  # noqa: BLE001
            return []
        out: list[ContextItem] = []
        for card in cards:
            decision = getattr(card.decision, "value", str(card.decision))
            # The summary, not the raw field: a card whose delta touched the
            # validation region has to say so wherever its verdict is read.
            reason = getattr(card, "decision_summary", "") or ""
            text = f"{card.id}: {decision} — {reason}".strip(" —")
            out.append(
                ContextItem(
                    id=f"experiments:evidence:{card.id}",
                    source=self.name,
                    kind="experiment",
                    text=text[:2000],
                    score=0.5,
                    reason="evidence card",
                    metadata={
                        "competition": competition,
                        "status": decision,
                        "card_id": card.id,
                        "node_id": card.id,
                        "created_at": getattr(card, "created_at", None) or "",
                    },
                )
            )
        return out

    def _from_knowledge(self, knowledge_dir: Path, competition: str) -> list[ContextItem]:
        try:
            from labpilot.research_engine.intelligence.knowledge.store import (
                KnowledgeStore,
            )
        except ImportError:
            return []
        out: list[ContextItem] = []
        try:
            with KnowledgeStore(knowledge_dir, competition) as store:
                for art in store.list_artifacts(type="experiment"):
                    text = (art.summary or art.title or art.id or "").strip()
                    if not text:
                        continue
                    out.append(
                        ContextItem(
                            id=f"experiments:artifact:{art.id}",
                            source=self.name,
                            kind="experiment",
                            text=text[:2000],
                            score=0.45,
                            reason="knowledge experiment artifact",
                            metadata={
                                "competition": competition,
                                "status": getattr(art, "status", None) or "known",
                                "artifact_id": art.id,
                                "node_id": art.id,
                                "created_at": getattr(art, "created_at", None) or "",
                                "updated_at": getattr(art, "updated_at", None) or "",
                            },
                        )
                    )
        except Exception:  # noqa: BLE001
            return []
        return out
