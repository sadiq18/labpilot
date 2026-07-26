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
`competition/`, `runtimes/`, `submission/`, `tracking/`, `workspace/`/`project/`, and
repo-root `templates/` were **removed** — logic lives under accessor /
intelligence / execution / experiments / project as below.

Shared infra: `labpilot.accessor` (SQLite, LLM, **Kaggle client**, `accessor.common`).

---

## 2. Tree

```
src/labpilot/
  accessor/kaggle/           # KaggleClient + CompetitionMetadata (no pillar imports)
  accessor/data/             # dataset download + DataLayout
  accessor/profiler/         # TabularProfiler + DatasetProfile
  accessor/common/           # ids, JSON helpers, Micro Agent contract (sole common/)
  experiments/               # graph + logger/store/index (was tracking/)
  research_engine/
    intelligence/competition/  # Parser + CompetitionSpec (fetch/normalize)
    execution/
      engineer.py / context.py / registry.py / store.py / evidence.py / …
      baseline/ training/ metrics.py
      capabilities/code_engineering/  # apply + offline_codegen + templates/
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
| `data/` | `accessor/data/` |
| `profiler/` | `accessor/profiler/` |
| `brief/` | `intelligence/brief/` (merged; ResearchBrief + legacy BriefGenerator) |
| `competition/` | `intelligence/competition/` |
| `common/` + `accessor/commons/` | `accessor/common/` |
| `runtimes/` | `execution/runtimes/` |
| `submission/` | `execution/submission/` |
| `tracking/` | `experiments/` (logger/store/index) |
| `workspace/` (project.yaml overlay) | **removed** — WorkspaceCapability under `execution/capabilities/workspace/` |
| `project/` | **removed** (same overlay; superseded by Engineer workspace) |
| `baseline/` | `execution/baseline/` |
| `codegen/` | `execution/capabilities/code_engineering/offline_codegen/` |
| repo-root `templates/` | `execution/capabilities/code_engineering/templates/` |
| `training/` | `execution/training/` |
| `evaluation/metrics.py` | `execution/metrics.py` |
| `evaluation/cv.py` | **deleted** (unused) |
| `orchestrator/pipeline.py` | **deleted** — manifests remain in `experiments/manifest.py` |

Import hygiene:

- `accessor` ✖ pillars; raw `CompetitionMetadata` may live in accessor
- `execution` → `accessor`, `common`, planner store/schema APIs
- `execution` ✖ deep `intelligence` internals (except shared competition types via public imports)
- `intelligence` / `planner` ✖ `execution`

---

## 4. Code Engineering

LLM proposes a typed `CodeProposal`; deterministic apply under allow-list.
Jinja templates under `execution/capabilities/code_engineering/templates/` are the offline `rule_engine` full
scaffold — not a separate product surface.

---

## 5. CLI wiring

| Command | Module |
|---------|--------|
| `research run --plan` | CLI → `execution.engineer.run_plan` |
| `research resume --execution` | CLI → `execution.engineer.resume` |
| `research plan create --baseline` | Planner |
| `research templates` | `execution.baseline.list_templates` |
