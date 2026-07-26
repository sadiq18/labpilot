# Research Reflection — Package layout

Back to [README](README.md).

---

## 1. Target tree (Reflection pillar)

```text
src/labpilot/research_engine/reflection/
  __init__.py
  evidence/                 # deterministic EvidenceExtractor (no LLM)
  critic/                   # RootCauseAgent (skill.md + micro_agent.py) + ExperimentCritic facade
  contradiction/            # ContradictionDetectorAgent
  confidence/               # ConfidenceEstimatorAgent (qualitative)
  synthesis/                # EvidenceSynthesisAgent + KnowledgeSynthesizer
  lessons/                  # LessonGeneratorAgent + LessonGenerator
  hypotheses/               # HypothesisRevisionAgent + HypothesisEvaluator
  recommendation/           # RecommendationAgent + recommend_next_experiment
  beliefs/                  # BeliefUpdater — deterministic confidence math
  claims/                   # ClaimPromoter (rules)
  journal/                  # JournalProjector (assembly only)
  store.py pipeline.py
```

Each LLM slice keeps `skill.md` + `micro_agent.py` in its domain folder (same
pattern as `critic/`). There is no central `reflection/micro_agents/` package.

---

## 2. Shared experiments (design — not nested under Reflection)

Hypothesis / comparator / knowledge / graph / ranking are **cross-pillar SoR**,
used by Intelligence, Planner, Reflection, and CLI. They must **not** live under
`reflection/`.

| Today (code) | Target (design) |
|--------------|-----------------|
| `src/labpilot/experiments/` | `src/labpilot/research_engine/shared/experiments/` |

```text
src/labpilot/research_engine/shared/          # planned package
  __init__.py
  experiments/
    models.py hypothesis.py comparator.py knowledge.py
    graph.py ranking.py search.py report.py
    manifest.py logger.py store.py index.py
```

**Import path (after move):** `labpilot.research_engine.shared.experiments`

**Why `shared/`, not `reflection/experiments`:** Planner and Intelligence already
depend on `Hypothesis` and related helpers. Nesting under Reflection would force
those pillars to import through Reflection and break peer import hygiene.

**When:** Separate package-move PR (not required to start Reflection Plans 2–5).
Until then, Reflection reads/writes via `labpilot.experiments`.

---

## 3. Migration map (legacy → pillars)

| Legacy | Disposition |
|--------|-------------|
| `src/labpilot/reflection/` | **Migrate** prompts/models into `reflection/critic` (+ schemas); **delete** top-level (Plan 9) |
| `execution/micro_agents/reflection_generator/` | Thin re-export → `reflection.critic` |
| `execution/capabilities/reporting/` | **Call** reflection library (Plan 5); stop JSON-only stubs |
| `src/labpilot/experiments/` | **Keep** for M6 implementation; **move** → `research_engine/shared/experiments/` (follow-on) |
| `src/labpilot/report/` | **Delete** in Plan 9 (Pipeline-era per-run HTML); relocate dashboard Jinja first; Journal is narrative SoR |
| `src/labpilot/improvement/` | **Delete** in Plan 9 after splitting live bits (see §5) |
| `src/labpilot/kernel/` | **Moved** → `accessor/kaggle/exporter.py` (done); top-level package removed |

---

## 4. Import rules

**Until shared move:**

```text
reflection → accessor, labpilot.experiments, planner/execution READ APIs
execution  → reflection (library calls from Reporting)
intelligence → may READ claims/beliefs; reflection does NOT import analyzers
```

**After shared move:**

```text
reflection → accessor, shared.experiments, planner/execution READ APIs
intelligence / planner / cli → shared.experiments (not via reflection)
shared.experiments → accessor only (no pillar deep-imports)
```

No control-flow “reflection agent.” Engineer and CLI are the entrypoints;
`pipeline.py` is a library function sequence.

---

## 5. `improvement/` cleanup (Plan 9) — design

Pipeline-era `research improve` is already gone. The package is quarantined; Plan 9
**deletes the whole tree** after relocating what still has callers.

| Piece today | Callers | Target |
|-------------|---------|--------|
| `DEFAULT_TABULAR_MODEL_PARAMS` | `offline_codegen/renderer.py` only | Move into `execution/capabilities/code_engineering/offline_codegen/` (e.g. `defaults.py`) |
| `TrainingOverrides`, `ImprovementPlan`, load/save | `experiments/graph.py` (read historical `runs/*/`) | Move into `experiments/` (e.g. `legacy_run_overrides.py` or `models.py`) — graph SoR, not codegen |
| `planner.py`, `fork.py`, `tuner.py`, `recipes.py` | Unit tests only | **Delete** with the package |

**Do not** put `ImprovementPlan` under codegen — that would couple experiment-graph
reads of legacy run folders to the code-engineering capability.

### Follow-on TODO (after Plan 9)

> **TODO:** Once template-based **offline codegen** is removed (LLM / non-template
> code engineering SoR), delete `offline_codegen/defaults.py` (or wherever
> `DEFAULT_TABULAR_MODEL_PARAMS` landed) and any Jinja tabular defaults that exist
> only to feed that renderer. Track next to Code Engineering capability docs /
> Engineer follow-ons — not blocked on Reflection Plans 2–8.

Until that TODO lands, codegen keeps owning the default LightGBM-ish param dict.
