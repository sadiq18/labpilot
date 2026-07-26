# Research Engineer — Package layout

Back to [README.md](README.md) · Planner layout:
[../research-planner/package-layout.md](../research-planner/package-layout.md).

**Status:** Phase B — execution SoR consolidated under `research_engine/execution/`.

---

## 1. Pillar placement

```
research_engine/
  intelligence/   # analyze → knowledge → hypothesize   (shipped)
  planner/        # Hypothesis / baseline → ResearchPlan (shipped)
  execution/      # Research Engineer + all implement/run code  ← SoR
```

Top-level `baseline/`, `codegen/`, `training/`, `evaluation/`, `kaggle/`,
`competition/`, `runtimes/`, `submission/`, `tracking/`, `workspace/`, and
repo-root `templates/` were **removed** — logic lives under accessor /
intelligence / execution / experiments / project as below.

Shared infra: `labpilot.accessor` (SQLite, LLM, **Kaggle client**, commons).

---

## 2. Tree

```
src/labpilot/
  accessor/kaggle/           # KaggleClient + CompetitionMetadata (no pillar imports)
  project/                   # project.yaml multi-comp root (≠ WorkspaceCapability)
  experiments/               # graph + logger/store/index (was tracking/)
  research_engine/
    intelligence/competition/  # Parser + CompetitionSpec (fetch/normalize)
    execution/
      engineer.py / context.py / registry.py / store.py / evidence.py / …
      baseline/ codegen/ templates/ training/ metrics.py
      runtimes/                # RuntimeConfig registry (config-only; ≠ Runtime capability)
      submission/              # Formatter + validator library
      capabilities/
        workspace/ code_engineering/ research_review/ dependency/
        verification/ runtime/ training/ evaluation/ submission/ reporting/
      micro_agents/
```

---

## 3. Migration map (done)

| Former top-level | Now |
|------------------|-----|
| `kaggle/` | `accessor/kaggle/` |
| `competition/` | `intelligence/competition/` |
| `runtimes/` | `execution/runtimes/` |
| `submission/` | `execution/submission/` |
| `tracking/` | `experiments/` (logger/store/index) |
| `workspace/` | `project/` (avoids WorkspaceCapability name clash) |
| `baseline/` | `execution/baseline/` |
| `codegen/` | `execution/codegen/` |
| repo-root `templates/` | `execution/templates/` |
| `training/` | `execution/training/` |
| `evaluation/metrics.py` | `execution/metrics.py` |
| `evaluation/cv.py` | **deleted** (unused) |
| `orchestrator/pipeline.py` | quarantined legacy (init/build/improve only) |

Import hygiene:

- `accessor` ✖ pillars; raw `CompetitionMetadata` may live in accessor
- `execution` → `accessor`, `common`, planner store/schema APIs
- `execution` ✖ deep `intelligence` internals (except shared competition types via public imports)
- `intelligence` / `planner` ✖ `execution`

---

## 4. Code Engineering

LLM proposes a typed `CodeProposal`; deterministic apply under allow-list.
Jinja templates under `execution/templates/` are the offline `rule_engine` full
scaffold — not a separate product surface.

---

## 5. CLI wiring

| Command | Module |
|---------|--------|
| `research run --plan` | CLI → `execution.engineer.run_plan` |
| `research resume --execution` | CLI → `execution.engineer.resume` |
| `research plan create --baseline` | Planner |
| `research templates` | `execution.baseline.list_templates` |
