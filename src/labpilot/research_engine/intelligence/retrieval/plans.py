"""Fixed QueryPlan stubs per ``query_type`` (Phase 1 — no adaptive planner)."""

from __future__ import annotations

from labpilot.research_engine.intelligence.retrieval.models import QueryPlan, QueryType

_PLANS: dict[QueryType, QueryPlan] = {
    QueryType.HYPOTHESIS_GENERATION: QueryPlan(
        query_type=QueryType.HYPOTHESIS_GENERATION,
        tables=["techniques", "beliefs", "research_artifacts", "artifact_techniques"],
        traversals=["technique→papers", "technique→experiments", "technique→repositories"],
        limits={"techniques": 12, "papers": 8, "experiments": 8, "repositories": 5, "failures": 6},
        reasoning_agent="HypothesisGeneratorAgent",
        rounds=["relevant_techniques", "expand_evidence", "compress"],
    ),
    QueryType.STRUCTURED_QUERY: QueryPlan(
        query_type=QueryType.STRUCTURED_QUERY,
        tables=["techniques", "research_artifacts", "artifact_techniques"],
        traversals=["technique→papers", "technique→experiments"],
        limits={
            "techniques": 20,
            "papers": 12,
            "experiments": 12,
            "repositories": 4,
            "failures": 4,
        },
        rounds=["symbolic_filter", "expand_evidence", "compress"],
    ),
    QueryType.EXPLAIN: QueryPlan(
        query_type=QueryType.EXPLAIN,
        tables=["techniques", "beliefs", "research_artifacts"],
        traversals=["technique→papers", "technique→experiments"],
        limits={"techniques": 8, "papers": 6, "experiments": 6, "repositories": 3, "failures": 4},
        rounds=["relevant_techniques", "expand_evidence", "compress"],
    ),
    QueryType.COMPARE: QueryPlan(
        query_type=QueryType.COMPARE,
        tables=["techniques", "research_artifacts", "artifact_techniques"],
        traversals=["technique→papers", "technique→experiments", "technique→repositories"],
        limits={"techniques": 10, "papers": 8, "experiments": 8, "repositories": 4, "failures": 4},
        rounds=["relevant_techniques", "expand_evidence", "compress"],
    ),
}


def plan_for(query_type: QueryType | str) -> QueryPlan:
    """Return the fixed Phase-1 plan for a query type (deep-copied via model_copy)."""
    key = QueryType(str(query_type))
    return _PLANS[key].model_copy(deep=True)
