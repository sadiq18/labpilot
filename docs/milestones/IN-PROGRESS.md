# In Progress

Back to [MILESTONES.md](../MILESTONES.md).

---

**Milestone 3 — Research Intelligence** is in **design** — design README + Knowledge System
storage doc; sequenced `plan-N-*.md` files come after review. See
[milestone-3/README.md](milestone-3/README.md) and
[milestone-3/knowledge-system.md](milestone-3/knowledge-system.md).

**Package layout (design):** `cli/` · `common/` · `research_engine/` (deployable service) with
`execution/` and `intelligence/` (future: separate package or service each). On disk:
`knowledge/<slug>/research/{raw,extracted,knowledge,experiments,reports}/` + `knowledge.db`
— **local / gitignored**. See [knowledge-system.md](milestone-3/knowledge-system.md).

**Milestone 2 — Experiment Scientist** — Plans 1–8 shipped (experiment graph through
dashboard). See [milestone-2/README.md](milestone-2/README.md).

**P4 — v1.0 (Production Quality)** shipped — see [COMPLETED.md](COMPLETED.md).

### Milestone 3 at a glance

Architecture center: shared **Knowledge Extraction Pipeline** (Raw → Normalizer → Extractor →
Validator → Store → Retrieval → Reasoning) over pluggable Analyzers. Everything is a
**ResearchArtifact** (id, type, source, metadata, summary, techniques, models, datasets,
claims, references, confidence). Prefer **official APIs**. Literature / GitHub / Forum
engines extract typed cards — not full summaries. **Knowledge System:** merge into
techniques / datasets / architectures / tasks in SQLite — **not** a vector DB.
**Multi-stage retrieval** + **ContextBuilder** (typed `ResearchContext`; LLM never sees DB);
hierarchical L1–L3 memory; Progressive Context; Query Planner direction — **Knowledge Engine
is the center**, LLM is an attached reasoner. **Hypothesis Assistant:** top-10 —
**no autonomous planner.** Optional **Micro Agents**; system works without them.

```bash
research analyze birdclef-2026
research analyze papers birdclef-2026
research analyze birdclef-2026 --include papers,repositories
# → knowledge/<slug>/research/reports/analyze.json + terminal summary
```

**Phase 1 (ship):** Competition / Paper / Repository / Experiment / Dataset analyzers +
synthesis. **Spike (non-blocking):** Kaggle discussion access + ToS. **Future:** Forum
Intelligence providers (`DiscussionAnalyzer` + `ForumKnowledgeExtractor`: Kaggle, GitHub
Issues, Reddit, …) — GitHub Issues may land without waiting on Kaggle.

| # | Plan | Depends on |
|---|------|------------|
| 1 | Models + registry + orchestrator + fetch-cache | M2 |
| 2 | ExperimentAnalyzer + DatasetAnalyzer | 1 |
| 3 | CompetitionAnalyzer (API; winning solutions → unavailable if needed) | 1 |
| 4 | PaperAnalyzer + LiteratureProvider + PaperKnowledgeExtractor | 1 |
| 5 | RepositoryAnalyzer + extract + RepoDiffer (vs local) | 1 |
| 6 | KB + Retrieval + Hypothesis Assistant (top-10 recs) | 1 (+ any of 2–5) |
| 7 | Capstone report | 6 |
| — | Spike: Kaggle discussions | — |
| F | Forum Intelligence (extract + providers) | Spike go or GitHub Issues + 1 |

Also queued, unrelated to Milestone 3: **P2 execution** and **Packaging & PyPI** — see
[TODO.md](TODO.md) and [backlog.md](backlog.md).
