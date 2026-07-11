# LabPilot Milestones

## North Star

**One command closes the first 80% of a Kaggle competition loop** — from competition page to uploaded submission and a written reflection — without hand-written code.

```bash
research run --competition titanic
```

After a few hours, a complete run should produce:

```
✔ Parsed competition
✔ Downloaded data
✔ Built research brief
✔ Generated baseline
✔ Trained model
✔ Evaluated CV
✔ Generated submission
✔ Uploaded submission
✔ Logged experiment
✔ Wrote reflection
```

---

## Milestone Roadmap

### P0 — Research Engine v0.1 (Core Loop Proof)

**Goal:** Prove the full pipeline works end-to-end for **one competition archetype** (tabular classification/regression), then generalize it beyond a single hardcoded competition.

| Dimension | Target |
|-----------|--------|
| Command | `research run --competition <slug>` |
| Problem types | Tabular only (classification + regression) |
| Human input | Kaggle API credentials + LLM API key |
| Runtime | 1–4 hours unattended |
| Proof | Real Kaggle competitions complete the full loop |

**Success criteria:**

- Single command runs start-to-finish without manual code edits
- All 8 P0 capabilities produce real artifacts on disk
- CV score is logged and aligns with the competition metric direction
- Submission uploads successfully via Kaggle API with a persisted leaderboard score
- Reflection cites actual run metrics and suggests 3–5 concrete next steps
- The loop generalizes to a new competition via a local config, not hardcoded logic

**Explicit P0 constraint:** One baseline per problem type, no search, no agents, no memory across runs.

**Validation competitions:** any tabular binary/multi-class classification or regression competition with a train/test/sample-submission split — see `configs/competitions/README.md` for how a new competition's contract is supplied locally.

---

### P1 — v0.2 (Problem Type Expansion)

**Goal:** Same loop, more competition types.

- Auto-detect problem type from profiler (tabular / text / image)
- Baseline templates for NLP and image classification
- Metric-aware evaluation (AUC, RMSE, log loss, etc.)
- Competition rules parser (submission format, limits, deadlines)

---

### P2 — v0.3 (Iteration Loop)

**Goal:** Turn reflection into action.

- `research improve --run-id <id>` reads reflection and re-runs targeted steps
- Manual or semi-automatic feature engineering from brief suggestions
- Simple hyperparameter tuning (grid/random, not AutoML)
- Diff between runs in experiment tracker

---

### P3 — v1.0 (Production Quality)

**Goal:** Reliable tool for repeated competition use.

- Multi-competition project workspace
- Config overrides (`--config`, `--dry-run`, `--submit`)
- CI-tested templates per problem type

---

### Future (Explicitly Deferred)

Build these only after the core loop is proven:

| Capability | Why deferred |
|------------|--------------|
| Multi-agent systems | Orchestrator + templates are enough for P0 |
| Vector databases | Brief uses competition page + profiler, not retrieval |
| Knowledge graphs | No cross-competition reasoning needed yet |
| Long-term memory | Each run is self-contained in P0 |
| Autonomous planning | Fixed pipeline DAG is sufficient |
| Self-modifying code | Templates + parameterization first |
| AutoML search | One strong baseline proves the loop |
| Multi-model orchestration | Single model per run in P0 |

---

## P0 Scope

**In scope:** one-command competition init, automatic dataset profiling, AI-generated research
brief, baseline template selection, training and evaluation, Kaggle submission, experiment
tracking, reflection with next-step recommendations.

**Out of scope:** see [Future (Explicitly Deferred)](#future-explicitly-deferred) above.

---

## Current Implementation Status

| Layer | Status |
|-------|--------|
| CLI + orchestrator | Wired — all 12 stages run in sequence |
| Manifest / status | Crash-safe — always records failure, even on `SystemExit` |
| Competition parser | Reads a local, per-competition contract (`configs/competitions/<slug>.yaml`) |
| Data download | Kaggle API download, unzip, and per-competition cache |
| Dataset profiler | Detects train/test/submission roles, target, and ID (patterns overridable per competition) |
| Brief / reflection | Fallback text only — no LLM calls |
| Baseline selection | Works from competition and dataset contracts |
| Code generation | Works — renders Jinja2 templates |
| Training | Fold-fitted preprocessing + LightGBM (classification + regression) |
| CV evaluation | Validates a real `cv_<metric>` from training |
| Submission | Validated against sample columns, row count, and labels (metric-aware) |
| Kaggle upload | Real API upload with leaderboard score polling, explicit `--submit` opt-in |
| Experiment logging | Works |
| Tests | Unit tests plus mocked end-to-end pipeline runs for both classification and regression |

---

## P0 Validation Status

Credentialed and validated for real, on two independent Kaggle competitions:

| Competition | Type | CV score | Public score |
|-------------|------|----------|---------------|
| Titanic | classification | `cv_accuracy` 0.7306 | 0.72488 |
| House Prices | regression | `cv_rmse` 30573.96 | 0.13259 |

Both runs exercised every stage (parse → download → profile → brief → baseline → code → train →
evaluate → submission → upload → log → reflection) with no manual code edits, and Kaggle accepted
and scored the `--submit` upload on both.

Proving this end-to-end surfaced and fixed several real bugs (relative run-path resolution,
leaderboard-score polling, crash-safe manifests, regression-template hardening, and an RMSE
compatibility fix) — see git history on this branch for details.

---

## P0 Remaining Work: Generalization & CLI Ergonomics

The core loop is proven, but P0 isn't done until the engine stops depending on
competition-specific hand-holding:

- Automatic competition metadata resolution from the Kaggle URL/slug (remove the need for a
  hand-written local contract file)
- LLM-backed brief/reflection (`OpenAI` call in `BriefGenerator.generate()` /
  `ReflectionGenerator.generate()` — currently accurate but template-based fallback text)
- Multi-class classification support
- `--resume --run-id <id>` — restart from failed stage
- `--verbose`/`--quiet` flag to control log level across all major classes
- Clearer environment diagnostics for Python and LightGBM

---

## P0 Baseline Strategy

One template per tabular type — no search, no AutoML:

| Type | Model | Features | CV |
|------|-------|----------|-----|
| Classification | LightGBM | Fold-fitted imputation + ordinal encoding | Stratified 5-fold |
| Regression | LightGBM | Same preprocessing | 5-fold |

Template selection rules:

```
if target is categorical and n_classes <= 20:
    → tabular_classification
elif target is numeric:
    → tabular_regression
else:
    → raise UnsupportedProblemType  # defer to P1
```
