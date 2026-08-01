"""Cross-competition Experience Records for Context Engine retrieve."""

from __future__ import annotations

from pathlib import Path

import anyio

from labpilot.research_engine.context.models import ContextItem, ContextRequest


class ExperienceProvider:
    """Fetch Experience Records from the shared experience store.

    Uses ``metadata.source_competition`` (not ``competition``) so
    ``apply_filters`` keeps cross-competition items for transfer memory.
    Operator-seeded IDs for the request competition get a modest score boost.
    """

    name = "experience"

    async def fetch(self, request: ContextRequest) -> list[ContextItem]:
        if request.knowledge_dir is None:
            return []
        return await anyio.to_thread.run_sync(self._fetch_sync, request)

    def _fetch_sync(self, request: ContextRequest) -> list[ContextItem]:
        from labpilot.research_engine.memory import ExperienceStore
        from labpilot.research_engine.memory.seed import load_seeded_experience_ids

        knowledge_dir = Path(request.knowledge_dir)
        seeded = load_seeded_experience_ids(knowledge_dir, request.competition)
        meta = request.metadata or {}
        outcome = meta.get("experience_outcome")
        facet = meta.get("experience_facet")

        store = ExperienceStore(knowledge_dir)
        try:
            records = store.list(
                outcome=str(outcome) if outcome else None,
                facet=str(facet) if facet else None,
                limit=int(meta.get("experience_limit") or 200),
            )
            items: list[ContextItem] = []
            for record in records:
                # Prefer other competitions for transfer; still include same-slug.
                facet_bits = " ".join(record.facet_names())
                text = " | ".join(
                    part
                    for part in (
                        record.goal,
                        record.hypothesis,
                        record.action,
                        record.result,
                        facet_bits,
                        record.source_competition,
                    )
                    if part
                )
                if not text.strip():
                    continue
                base = 0.58 if record.outcome == "success" else 0.48
                if record.id in seeded:
                    base = min(0.75, base + 0.12)
                items.append(
                    ContextItem(
                        id=f"experience:{record.id}",
                        source=self.name,
                        kind="experience",
                        text=text[:2000],
                        score=base,
                        reason=(
                            f"experience {record.outcome} from "
                            f"{record.source_competition}"
                            + (" (seeded)" if record.id in seeded else "")
                        ),
                        metadata={
                            # Do NOT set competition — filters would drop cross-comp.
                            "source_competition": record.source_competition,
                            "status": record.outcome,
                            "experience_id": record.id,
                            "facets": record.facet_names(),
                            "outcome": record.outcome,
                            "seeded": record.id in seeded,
                            "artifacts": record.artifacts.model_dump(mode="json"),
                            "created_at": record.created_at.isoformat(),
                            "updated_at": record.updated_at.isoformat(),
                            "node_id": record.id,
                        },
                    )
                )
            return items
        finally:
            store.close()
