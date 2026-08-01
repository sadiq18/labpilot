# M4 — Memory & Context Engine

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Stub.  
**Branch:** `research-os-m4-context`  
**Depends on:** M3  
**Design:** [07-context-engine](../../design/07-context-engine.md) · [10-memory-os](../../design/10-memory-os.md)

## Mission

Intelligence layer: memory hierarchy ports + retrieve→rank→compress. Keep Evidence
Card / Research Graph as semantic seed.

## Usable outcome

Task-local context bundles; better prompts; foundation for `explain`.

## Tech that ships with M4

| Area | Technology |
|------|------------|
| Metadata | SQLite |
| Retrieval | Hybrid: filters + BM25 + graph walk |
| Graph | Logical SQL graph → **Kuzu** only if needed |
| Vectors | **Defer** → Qdrant when ANN required |
| Embeddings | Chosen at impl via LLM router / provider |

## Non-goals

- Agent registry (M5)
- Experience transfer DB (M6)
- Shipping Qdrant on day one by default
