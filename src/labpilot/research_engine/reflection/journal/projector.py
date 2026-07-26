"""JournalProjector — human-readable research memory projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.reflection.recommendation.next_experiment import (
    recommend_next_experiment,
)
from labpilot.research_engine.reflection.store import ReflectionStore
from labpilot.research_engine.reflection.synthesis.synthesizer import KnowledgeSynthesizer


class JournalProjector:
    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.competition = competition
        self._reflection = ReflectionStore(Path(knowledge_dir), competition)
        self._knowledge = KnowledgeStore(Path(knowledge_dir), competition)
        self._synth = KnowledgeSynthesizer(Path(knowledge_dir), competition)

    def close(self) -> None:
        self._reflection.close()
        self._knowledge.close()
        self._synth.close()

    def build(self, *, assessment_recommendation: str = "") -> dict[str, Any]:
        understanding = self._synth.current_understanding()
        evidence = self._reflection.list_evidence()
        claims = self._list_claims()
        recommendation = recommend_next_experiment(
            understanding, assessment_recommendation=assessment_recommendation
        )
        buckets = {
            "strong": [],
            "moderate": [],
            "weak": [],
            "rejected": [],
        }
        for row in evidence:
            buckets.setdefault(str(row.get("strength") or "moderate"), []).append(
                {
                    "id": row["id"],
                    "execution_id": row.get("execution_id"),
                    "metrics": row.get("metrics"),
                }
            )
        return {
            "competition": self.competition,
            "evidence": buckets,
            "open_questions": understanding.get("open_hypotheses") or [],
            "beliefs": understanding.get("beliefs") or [],
            "claims": claims,
            "understanding": understanding.get("summary"),
            "recommended_next": recommendation,
        }

    def render_markdown(self, journal: dict[str, Any] | None = None) -> str:
        data = journal or self.build()
        lines = [
            f"# Research Journal — {data['competition']}",
            "",
            data.get("understanding") or "",
            "",
            "## Evidence",
            "",
        ]
        for strength in ("strong", "moderate", "weak", "rejected"):
            items = data["evidence"].get(strength) or []
            lines.append(f"### {strength.title()} ({len(items)})")
            if not items:
                lines.append("- —")
            else:
                for item in items:
                    lines.append(
                        f"- `{item['id']}` exec={item.get('execution_id') or '—'}"
                    )
            lines.append("")
        lines.extend(["## Open questions", ""])
        for hyp in data.get("open_questions") or []:
            lines.append(
                f"- `{hyp['id']}` [{hyp['status']}]: {hyp.get('prediction', '')}"
            )
        if not data.get("open_questions"):
            lines.append("- —")
        lines.extend(["", "## Beliefs", ""])
        for belief in data.get("beliefs") or []:
            lines.append(
                f"- `{belief['id']}` {belief.get('technique')} "
                f"({belief.get('effect')}, conf={belief.get('confidence')})"
            )
        if not data.get("beliefs"):
            lines.append("- —")
        lines.extend(["", "## Claims", ""])
        for claim in data.get("claims") or []:
            lines.append(
                f"- `{claim['id']}` [{claim.get('status')}]: {claim.get('statement')}"
            )
        if not data.get("claims"):
            lines.append("- —")
        nxt = data.get("recommended_next") or {}
        lines.extend(
            [
                "",
                "## Recommended next experiment",
                "",
                nxt.get("rationale") or "—",
                "",
                f"Suggested: `{nxt.get('command') or '—'}`",
                "",
            ]
        )
        return "\n".join(lines)

    def render_json(self, journal: dict[str, Any] | None = None) -> str:
        return json.dumps(journal or self.build(), indent=2)

    def _list_claims(self) -> list[dict[str, Any]]:
        return [
            {
                "id": row["id"],
                "statement": row["statement"],
                "status": row["status"],
                "confidence": row["confidence"],
                "technique": row["technique"],
            }
            for row in self._reflection.list_claims()
        ]
