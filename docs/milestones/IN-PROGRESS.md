# In Progress

Back to [MILESTONES.md](../MILESTONES.md).

---

**Milestone 3 — Research Intelligence** is in **design** — single design README first;
sequenced `plan-N-*.md` files come after review. See
[milestone-3/README.md](milestone-3/README.md).

**Package layout (design):** `cli/` · `common/` · `research_engine/` (deployable service) with
`execution/` and `intelligence/` (future: separate package or service each). Explored
intelligence on disk: `knowledge/<slug>/intelligence/{papers,experiments,repositories,discussions,techniques,models,datasets}/`
— **local / gitignored**. See [milestone-3/README.md](milestone-3/README.md) §11.

**Milestone 2 — Experiment Scientist** — Plans 1–8 shipped (experiment graph through
dashboard). See [milestone-2/README.md](milestone-2/README.md).

**P4 — v1.0 (Production Quality)** shipped — see [COMPLETED.md](COMPLETED.md).

### Milestone 3 at a glance

Architecture center: Research Assistant stack (Readers → Extractor → structured knowledge
store → retrieval → Hypothesis Assistant) implemented as pluggable Analyzers.
`Analyzer.analyze(context) → ResearchArtifacts` (batch of **ResearchArtifact**: id, type,
source, title, summary, concepts, techniques, evidence, references, confidence). Prefer
**official APIs**. Fetch → cache → normalize → analyze. Literature is a **Paper Research Engine**: collect via
`LiteratureProvider` (Semantic Scholar → OpenAlex → arXiv → Papers with Code), then extract
**PaperKnowledge** (contributions / methods / limitations / ideas) — not full summaries.
Repositories are a **GitHub Intelligence** engine: collect via `RepositoryProvider`, extract
**RepoKnowledge** (architecture / loss / aug / tricks / files / deps), then **diff vs local**
(`TransferOpportunity`: effort + expected gain) — not README dumps.
**Forum Intelligence** (design now, providers gated): extract **ForumKnowledge** (common
mistakes / discoveries / dataset bugs / LB shakeups / OOD) from Kaggle / GitHub Issues /
Reddit / blogs — practical signal often absent from papers. Kaggle *access* is a
non-blocking spike. **Knowledge Extraction hub:** all sources normalize into
**KnowledgeUnit** (technique / task / problem / benefit / evidence / limitations /
references / confidence) and accumulate reusable cards in a **layered Research Knowledge
Base** (Documents → Knowledge → Evidence → Beliefs) — **not** a vector database.
**Research Retrieval:** given the competition, retrieve papers / experiments / repos /
discussions / **failures** by **task · metric · dataset · domain · architecture ·
technique** — not keywords alone. **Hypothesis Assistant:** connects everything → top-10
recommendations (impact, confidence, evidence, effort) — **no autonomous planner.**

```bash
research analyze birdclef-2026
research analyze papers birdclef-2026
research analyze birdclef-2026 --include papers,repositories
# → knowledge/<slug>/intelligence/analyze.json + terminal summary
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
