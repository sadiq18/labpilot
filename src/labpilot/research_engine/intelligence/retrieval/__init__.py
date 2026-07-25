"""Multi-stage Retrieval + Context Builder (Plan 9).

Intent → Symbolic → Expansion → Compression → typed ``ResearchContext``.
No embeddings Stage 3. No analyze wiring — Plan 10 consumes this library.
"""

from labpilot.research_engine.intelligence.retrieval.context_builder import (
    ContextBuilder,
    build_research_context,
)
from labpilot.research_engine.intelligence.retrieval.fetchers import SymbolicFetcher
from labpilot.research_engine.intelligence.retrieval.intent import (
    classify_intent,
    classify_intent_rules,
)
from labpilot.research_engine.intelligence.retrieval.models import (
    L1_CHAR_BUDGET,
    L2_CHAR_BUDGET,
    L3_CHAR_BUDGET,
    TOTAL_CHAR_BUDGET,
    QueryPlan,
    QueryType,
    ResearchContext,
    RetrievalHit,
    RetrievalIntent,
    SymbolicBundle,
    TechniqueCard,
)
from labpilot.research_engine.intelligence.retrieval.plans import plan_for

__all__ = [
    "L1_CHAR_BUDGET",
    "L2_CHAR_BUDGET",
    "L3_CHAR_BUDGET",
    "TOTAL_CHAR_BUDGET",
    "ContextBuilder",
    "QueryPlan",
    "QueryType",
    "ResearchContext",
    "RetrievalHit",
    "RetrievalIntent",
    "SymbolicBundle",
    "SymbolicFetcher",
    "TechniqueCard",
    "build_research_context",
    "classify_intent",
    "classify_intent_rules",
    "plan_for",
]
