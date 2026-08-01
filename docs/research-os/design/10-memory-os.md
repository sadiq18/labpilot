# Design — Memory OS

Back to [../README.md](../README.md) · Milestone: [../milestones/06-transfer-memory/](../milestones/06-transfer-memory/).

**Milestone:** M6 · **Impl branch:** `research-os-m6-transfer-memory`

Most agent projects fail by equating memory with “a vector database.” Treat memory
as an **OS hierarchy**; the Context Engine ([07-context-engine](07-context-engine.md))
retrieves across tiers.

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

M1–M2 add clean **query/store ports**. M6 deepens **cross-competition transfer**
and reasoning/episodic replay — not a greenfield database rewrite.

---

## Storage map (co-evolves with roadmap)

Do not stand up the full polyglot stack in M1. Introduce stores when a milestone
needs them — see [README roadmap](../README.md).

| Store | Holds | When |
|-------|-------|------|
| SQLite / Postgres | Artifacts, tasks, experiments, metadata | Now → Postgres if multi-user |
| Graph | Semantic edges | Logical SQL now → **Kuzu** in/after M4 if needed |
| Object / FS | Models, checkpoints, logs | Workspace FS → S3/MinIO with remote runtimes |
| Vectors | ANN for Context Engine | **M4+ only if** BM25+graph insufficient → Qdrant |
| Analytics | Experience analysis | **M6** optional DuckDB |

---

## Transfer learning across competitions

Store and reuse:

- Successful / failed prompt patterns
- Good hyperparameters and architectures that transferred
- Useful papers / features / ensembles by modality

Warm-start a new competition above blank-slate priors (ties to pipeline backlog
“cross-competition shared knowledge”).

---

## Non-goals

- AutoML search over all history
- “Memory = Qdrant only”
- Replacing research memory with generic agent-memory SaaS as SoR
- Graph DB migration before SQL becomes a bottleneck
- Replacing in-competition Evidence Card flow
