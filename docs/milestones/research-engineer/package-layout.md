# Research Engineer — Package layout

Back to [README.md](README.md) · Planner layout:
[../research-planner/package-layout.md](../research-planner/package-layout.md).

**Status:** Design Phase A. No `src/` moves until Phase B.

---

## 1. Pillar placement

```
research_engine/
  intelligence/   # analyze → knowledge → hypothesize   (shipped)
  planner/        # Hypothesis / baseline → ResearchPlan (shipped)
  execution/      # Research Engineer + capability executors  ← this milestone
```

`execution/` already exists (near-empty; reflection micro-agent only). It becomes the home
for the **Research Engineer** controller and all **capability** packages. Product name is
Research Engineer; directory name stays `execution/` for OS metaphor
(analyze → plan → **execute/implement**).

Shared infra remains `labpilot.accessor` (SQLite, LLM client, commons).

---

## 2. Target tree (sketch)

```
src/labpilot/research_engine/execution/
  __init__.py
  engineer.py              # Research Engineer controller (queue, dispatch, resume)
  context.py               # TaskContext assembly (bounded)
  registry.py              # CapabilityRegistry
  verification.py          # Verification Engine
  recovery.py              # Recovery Controller
  evidence.py              # Evidence writers
  artifacts.py             # Experiment Artifact Generator
  store.py                 # ExecutionStore (research_executions + task status updates)
  schemas/                 # ExecutionAttempt, TaskEvidence, … Pydantic
  capabilities/
    workspace/
    code_engineering/      # + micro_agents for patch proposal
    research_review/       # + micro_agent for research correctness
    dependency/
    verification/          # unit / smoke / integration runners
    runtime/               # select, dispatch, poll, pull
    training/
    evaluation/            # infer / evaluate / compare
    submission/
    reporting/             # report / belief / hypothesis / reflect
  micro_agents/            # only specialists used by capabilities (existing reflection stays)
```

Exact file names may tighten in Phase B; capability folders are the stable unit.

---

## 3. Migration map (into `execution/` or thin adapters)

| Today | Becomes |
|-------|---------|
| `orchestrator/pipeline.py` stage DAG | **Deleted** as SoR after capstone; logic absorbed into Engineer + capabilities |
| `orchestrator/manifest.py` | Informs execution/workspace records; may slim or merge |
| `codegen/` (Jinja SoR) | Superseded as SoR by **Code Engineering** capability (LLM-bounded patches + deterministic apply). Jinja may remain as `rule_engine` fallback for baseline scaffolds |
| `training/` | Training capability adapter |
| `evaluation/` | Evaluation capability |
| `submission/` + `kaggle/` upload bits | Submission capability |
| `baseline/` selector | Used by baseline **planner** template + Workspace/Code prep |
| `runtimes/` | Runtime capability (dispatch/poll added here or under `capabilities/runtime/`) |
| `reflection/` + `execution/micro_agents/reflection_generator` | Reporting & Memory capability |
| `improvement/` fork/plan | Replaced or reduced to “hypothesis plan → Engineer”; no parallel improve SoR |
| `tracking/` | Evidence / experiment logging hooks |

Import hygiene:

- `execution` → `accessor`, `common`, planner **schemas/store APIs** (read plans; update statuses)
- `execution` ✖ `intelligence` internals (consume Analyze artifacts via paths/DB, not deep imports)
- `intelligence` / `planner` ✖ `execution` (planner never calls Engineer)

---

## 4. Code Engineering (important)

Do **not** allow arbitrary edits.

Controlled operations only, e.g.:

```text
task: WRITE_CODE | MODIFY_CONFIG | fix
target: allowed path set
change: typed intent
validation: unit / compile / review
rollback: snapshot restore
```

LLM proposes a patch; deterministic code applies, Research Review gates, Verification
re-runs. This replaces “Jinja-only codegen” as the primary implementation path while
keeping offline fallbacks.

---

## 5. CLI wiring

| Command | Module |
|---------|--------|
| `research run --plan` | CLI → `execution.engineer.run_plan` |
| `research resume --execution` | CLI → `execution.engineer.resume` |
| `research plan create --baseline` | Planner (Phase B extension) |

Thin CLI — no orchestration logic in `cli/`.

---

## 6. Non-goals for layout

- New top-level package outside `research_engine/` for the Engineer
- Copy-paste of Pipeline into execution without deleting the old SoR
- Capability packages importing `cli`
- Intelligence importing execution
