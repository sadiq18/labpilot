"""``ConceptNormalizerAgent`` — collapse alias strings into one canonical concept.

Flagship "Yes" pattern (design §2.4 / §7): rules alone cannot reliably unify
"SpecAugment", "Time Masking", "Frequency Masking" into one technique. Emits a
:class:`ConceptNormalization` (canonical + aliases + category).
"""

from __future__ import annotations

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.intelligence.feature_recipes import (
    FEATURE_ENGINEERING_CATEGORY,
    looks_like_feature_engineering,
)
from labpilot.research_engine.intelligence.micro_agents.artifacts import ConceptNormalization


class ConceptNormalizerAgent(BaseMicroAgent):
    name = "ConceptNormalizerAgent"
    output_model = ConceptNormalization

    def system_prompt(self) -> str:
        return (
            "You normalize a set of technique/concept strings into one "
            'canonical concept. Respond ONLY with JSON: {"canonical": str, '
            '"aliases": [str], "category": str}. Pick the most standard name '
            "as canonical and list the rest as aliases. "
            "Use category 'feature_engineering' for feature-creation techniques."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        candidates = "\n".join(context.items)
        return f"Candidate concepts:\n{candidates}"

    def _run_rule_engine(self, context: StructuredContext) -> ConceptNormalization:
        seen: list[str] = []
        for raw in context.items:
            name = raw.strip()
            if name and name not in seen:
                seen.append(name)
        canonical = seen[0] if seen else ""
        category = str(context.data.get("category", "")).strip()
        if not category and looks_like_feature_engineering(" ".join(seen)):
            category = FEATURE_ENGINEERING_CATEGORY
        return ConceptNormalization(
            canonical=canonical,
            aliases=seen[1:],
            category=category,
        )
