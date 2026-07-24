# RepoQueryPlannerAgent

Builds a typed, category-aware GitHub search plan. It does not search GitHub.

## Input

- `competition`: normalized competition slug.
- `text`: title, tags, modality, keywords, and core topic.
- `data.seed_queries`: deterministic `{category, query}` objects.

## Output (`RepoSearchPlan`)

```json
{
  "queries": [
    {"category": "winning_solution", "query": "birdclef-2026 solution"},
    {"category": "baseline", "query": "birdclef starter language:Python"}
  ]
}
```

Keep at most eight queries. Prefer short queries (about 3–6 tokens). Use at most
one quoted phrase per query. Prefer recall over stacked exact-phrase filters.

## Fallback

The `rule_engine` validates and returns the deterministic seed unchanged. Invalid
or unavailable LLM output therefore never prevents repository discovery.

## Hard rules

- Never call GitHub or any other network service.
- Never summarize the competition.
- Never invent categories outside `RepoCategory`.
- Do not require many simultaneous keywords (that often yields zero hits).
