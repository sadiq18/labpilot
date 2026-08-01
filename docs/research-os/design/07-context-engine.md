# Design — Context engine

Back to [../README.md](../README.md) · Milestone: [../milestones/04-context/](../milestones/04-context/).

**Milestone:** M4 · **Impl branch:** `research-os-m4-context`

---

## Goal

Probably the highest-leverage subsystem after the Conductor. **Never dump everything**
into the model.

```text
User Goal + Task
    → Context Builder
    → retrieve → rank → compress
    → ContextBundle
    → LLM / Conductor policy / agent
```

---

## Pipeline

| Step | Intent |
|------|--------|
| Retrieve | Candidates from memory hierarchy + workspace |
| Rank | Relevance to goal/task (scores, recency, graph distance) |
| Compress | Summaries for long logs/notebooks/papers before prompt use |

Port: ``build_context(request) -> ContextBundle`` in
``labpilot.research_engine.context`` (sync facade; AnyIO gather inside).
Existing RI retrieval is a **provider**, not rewritten.

---

## Retrieval signals (combine; don’t pick one religion)

| Signal | Use |
|--------|-----|
| Structured filters | Competition, metric, technique, status |
| **BM25** / lexical | Fast file and note search |
| Embeddings + ANN | Semantic similarity (**Qdrant** when needed; defer until lexical+graph insufficient) |
| **Graph traversal** | Research Graph / technique↔claim↔experiment paths (already started) |

Reuse V1 retrieval where possible; Context Engine is the *orchestration* of these
signals, not a second knowledge store.

---

## Compression

Summarize long logs, notebooks, experiment dumps, and papers. Persist reusable
summaries into working/long-term memory ([10-memory-os](10-memory-os.md)).

---

## Non-goals

- Fine-tuning models
- Replacing Evidence Card / graph stores
- Shipping Qdrant on day one of M4
- Agent registry (M5)
