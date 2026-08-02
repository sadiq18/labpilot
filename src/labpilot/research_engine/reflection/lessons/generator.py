"""Lessons generator — Micro Agent prose + ReflectionStore persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.reflection.critic.critic import CriticAssessment
from labpilot.research_engine.reflection.lessons.micro_agent import (
    LessonDraft,
    LessonGeneratorAgent,
)
from labpilot.research_engine.reflection.store import ReflectionStore


class LessonGenerator:
    def __init__(
        self, knowledge_dir: Path, competition: str, *, llm_client: Any | None = None
    ) -> None:
        self.competition = competition
        self._store = ReflectionStore(Path(knowledge_dir), competition)
        self._agent = LessonGeneratorAgent(llm_client=llm_client)

    def close(self) -> None:
        self._store.close()

    def generate(
        self,
        assessment: CriticAssessment,
        evidence: dict[str, Any],
        *,
        cross_competition: bool = False,
        needs_review: bool = False,
    ) -> dict[str, Any]:
        draft = self._agent.run(
            StructuredContext(
                competition=self.competition,
                data={
                    "strength": evidence.get("strength") or "moderate",
                    "likely_cause": assessment.likely_cause,
                    "summary": assessment.summary,
                    "belief_effect": assessment.belief_effect,
                },
            )
        )
        assert isinstance(draft, LessonDraft)
        slug = None if cross_competition else self.competition
        meta: dict[str, Any] = {
            "strength": evidence.get("strength"),
            "belief_effect": assessment.belief_effect,
            "generated_by": "llm" if self._agent.last_used_llm else "rule_engine",
        }
        if needs_review:
            meta["needs_review"] = True
        return self._store.create_lesson(
            draft.summary,
            category=draft.category,
            confidence=draft.confidence,
            source_execution=evidence.get("execution_id"),
            competition_slug=slug,
            metadata=meta,
        )
