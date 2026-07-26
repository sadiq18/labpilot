"""KnowledgeSynthesizer — rollup + EvidenceSynthesis Micro Agent narrative."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.experiments.hypothesis import HypothesisStore
from labpilot.experiments.models import HypothesisStatus
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.reflection.synthesis.micro_agent import (
    EvidenceSynthesisAgent,
    EvidenceSynthesisDraft,
)
from labpilot.research_engine.reflection.store import ReflectionStore


class KnowledgeSynthesizer:
    def __init__(
        self, knowledge_dir: Path, competition: str, *, llm_client: Any | None = None
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self._reflection = ReflectionStore(self.knowledge_dir, competition)
        self._knowledge = KnowledgeStore(self.knowledge_dir, competition)
        self._hypotheses = HypothesisStore(self.knowledge_dir, competition)
        self._agent = EvidenceSynthesisAgent(llm_client=llm_client)

    def close(self) -> None:
        self._reflection.close()
        self._knowledge.close()

    def current_understanding(self) -> dict[str, Any]:
        evidence = self._reflection.list_evidence()
        beliefs = self._knowledge.list_beliefs()
        open_hyps = [
            h
            for h in self._hypotheses.list()
            if h.status
            in {
                HypothesisStatus.PROPOSED,
                HypothesisStatus.TESTING,
                HypothesisStatus.INCONCLUSIVE,
            }
        ]
        by_strength: dict[str, int] = {
            "strong": 0,
            "moderate": 0,
            "weak": 0,
            "rejected": 0,
        }
        ids_by_strength: dict[str, list[str]] = {
            "strong": [],
            "moderate": [],
            "weak": [],
            "rejected": [],
        }
        for row in evidence:
            strength = str(row.get("strength") or "moderate")
            by_strength[strength] = by_strength.get(strength, 0) + 1
            ids_by_strength.setdefault(strength, []).append(row["id"])

        top_beliefs = sorted(
            beliefs, key=lambda b: float(b.get("confidence") or 0), reverse=True
        )[:10]
        belief_lines = [
            f"{b.get('technique')} ({b.get('effect')}, conf={b.get('confidence')})"
            for b in top_beliefs
        ]
        open_lines = [f"{h.id}: {h.prediction}" for h in open_hyps]

        draft = self._agent.run(
            StructuredContext(
                competition=self.competition,
                data={
                    "evidence_by_strength": by_strength,
                    "belief_lines": belief_lines,
                    "open_hypothesis_lines": open_lines,
                },
            )
        )
        assert isinstance(draft, EvidenceSynthesisDraft)

        return {
            "competition": self.competition,
            "evidence_by_strength": by_strength,
            "evidence_ids_by_strength": ids_by_strength,
            "beliefs": [
                {
                    "id": b["id"],
                    "technique": b.get("technique"),
                    "effect": b.get("effect"),
                    "confidence": b.get("confidence"),
                    "status": b.get("status"),
                }
                for b in top_beliefs
            ],
            "open_hypotheses": [
                {"id": h.id, "prediction": h.prediction, "status": h.status.value}
                for h in open_hyps
            ],
            "summary": draft.summary,
            "open_questions_text": draft.open_questions,
            "key_takeaways": draft.key_takeaways,
            "generated_by": "llm" if self._agent.last_used_llm else "rule_engine",
        }
