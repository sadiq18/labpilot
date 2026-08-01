# Design — Memory OS

Back to [../README.md](../README.md) · Milestone: [../milestones/06-transfer-memory/](../milestones/06-transfer-memory/).

**Milestone:** M6 · **Impl branch:** `research-os-m6-transfer-memory`

Most agent projects fail by equating memory with “a vector database.” Treat memory
as an **OS hierarchy**; the Context Engine ([07-context-engine](07-context-engine.md))
retrieves across tiers.

**M6 principle:** Build **research memory**, not a research wiki. System of record is
structured **Experience Records**. Retrieval and usage decide which patterns matter —
do not hardcode first-class stores for prompts, HPs, architectures, or papers.

---

## Hierarchy

| Tier | Holds | TTL |
|------|-------|-----|
| **Short-term** | Current task, open files, recent logs/errors | Minutes |
| **Working** | Session: experiments, artifacts, notebook, current plan, models | Hours (campaign) |
| **Long-term** | Papers, techniques, reflections, failures, winning ideas, prompt/code templates | Forever |
| **Episodic** | Full runs: session → task history → decisions → failures → timeline | Forever (replay) |
| **Semantic** | Knowledge graph (technique ↔ claim ↔ experiment ↔ paper …) | Forever |

You already seed semantic + experiment memory (Evidence Cards, Research Graph,
beliefs/claims). Keep and extend them — do not replace with generic LangChain/Mem0/Zep
as system of record. Those products are commodity chat memory; ours stores *why we
tried X, why it failed, what worked in similar competitions*.

### Experiment tracking (hybrid)

Reuse **MLflow** (or similar) for runs/metrics/artifacts. **Build** the research layer
above it: Hypothesis, Reason, Decision, Failure, Next action, Reflection links.
Do not pretend MLflow alone is research memory.

Content lenses (orthogonal to TTL tiers):

| Lens | Seed today |
|------|------------|
| Research | RI knowledge + hypotheses |
| Experiment | Experiments + Evidence Cards |
| Code | Templates / prompts |
| Reasoning | Conductor decision log (M2+) |
| Experience | Cross-comp Experience Records (M6) |

M1–M2 add clean **query/store ports**. M4 owns retrieve → rank → compress. M6 adds
the **experience** lens and cross-competition transfer — not a greenfield database
rewrite.

---

## Storage map (co-evolves with roadmap)

Do not stand up the full polyglot stack in M1. Introduce stores when a milestone
needs them — see [README roadmap](../README.md).

| Store | Holds | When |
|-------|-------|------|
| SQLite / Postgres | Artifacts, tasks, experiments, metadata, **experience records** | Now → Postgres if multi-user |
| Graph | Semantic edges | Logical SQL now → **Kuzu** when signals justify ([backlog](../backlog/kuzu-graph-backend.md)) |
| Object / FS | Models, checkpoints, logs | Workspace FS → S3/MinIO with remote runtimes |
| Vectors | ANN for Context Engine | Only if BM25+graph insufficient → Qdrant ([backlog](../backlog/hybrid-semantic-retrieval.md)) |
| Analytics | Offline experience analysis | **Defer** DuckDB ([pattern extraction backlog](../backlog/experience-pattern-extraction.md)) |

Shared tables across competitions: start with SQLite experience store; multi-tenant
scale is [shared-multi-tenant-store](../backlog/shared-multi-tenant-store.md).

---

## Experience Records (M6 SoR)

M6 builds the memory foundation. Hardcoding category tables (prompts, HPs,
architectures, papers, features) becomes rigid quickly. Instead, persist structured
episodes; let the Context Engine decide what to surface.

### Record shape

| Field | Intent |
|-------|--------|
| `goal` | What we were trying to improve |
| `hypothesis` | What we believed |
| `action` | What we changed |
| `result` | Measured outcome (metrics / delta) |
| `outcome` | `success` \| `fail` (or equivalent coarse label) |
| `artifacts` | Links: experiment, metrics, reflection, code `git_commit` (when M5 present) |
| `tags` | Lightweight facets (modality, technique) for filter/retrieve — not a taxonomy wiki |
| `source_competition` | Origin slug |
| `ids` | Stable experience id; idempotency key from experiment/execution |

Example:

```text
Experience Record
  Goal:        Improve BirdCLEF score
  Hypothesis:  SpecAugment helps minority classes
  Action:      Added SpecAugment + EMA
  Result:      +0.006 LB score
  Outcome:     Success
  Artifacts:   code commit, experiment, metrics, reflection
  Tags:        audio, augmentation, imbalance
```

### Write path

```text
Experiment completed (+ reflection)
    → ExperienceExtractor (deterministic)
    → ExperienceStore (SQLite, cross-competition)
```

Prefer M5 event subscriber when available; Reflection / Engineer completion hooks
are acceptable fallbacks. Upserts are idempotent on experiment/execution id.

### Read path (influence, not control)

```text
New Competition
        |
        v
Context Engine
        |
        +-- retrieve similar experiences (experience provider)
        |
        +-- build priors into ContextBundle
        v
Conductor
        v
Research Campaign
```

**Principle:** Memory influences the Conductor via `ContextBundle`. It never silently
controls strategy. No automatic campaign-start seeding in M6.

### Human-visible warm start (CLI)

```text
research memory seed --from <slug>          # explicit priors into target workspace
research memory inspect --similar-to <slug> # debug / trust
research memory list|show …                 # as needed
```

Retrieve-always keeps architecture clean; CLI gives researchers control to inspect,
debug, and warm-start when desired.

### What emerges later (not M6 stores)

```text
Experience Memory → extracts (via retrieval/usage) →
  Prompt patterns | Model patterns | Feature patterns | Paper patterns
```

That layer is backlog: [experience-pattern-extraction](../backlog/experience-pattern-extraction.md).
Automatic transfer with confidence scoring:
[automatic-transfer-confidence](../backlog/automatic-transfer-confidence.md) (M7+).

---

## Non-goals

- AutoML search over all history
- “Memory = Qdrant only”
- Replacing research memory with generic agent-memory SaaS as SoR
- Graph DB migration before SQL becomes a bottleneck
- Replacing in-competition Evidence Card flow
- First-class prompt / HP / architecture / paper / feature wiki tables in M6
- DuckDB analytics in M6
- Silent automatic warm-start at campaign start
- Pattern-library productization (emergent extraction is post-M6)
