# Completed Milestones

Back to [MILESTONES.md](../MILESTONES.md).

---

## P0 — Research Engine v0.1 (Core Loop Proof)

**Goal:** Prove the full pipeline works end-to-end for tabular classification/regression, then generalize beyond a single hardcoded competition.

| Dimension | Target |
|-----------|--------|
| Command | `research run --competition <slug>` |
| Problem types | Tabular only (classification + regression) |
| Human input | Kaggle API credentials (+ optional LLM API key) |
| Runtime | 1–4 hours unattended |
| Proof | Real Kaggle competitions complete the full loop |

**Success criteria (all met):**

- Single command runs start-to-finish without manual code edits
- All pipeline stages produce real artifacts on disk
- CV score logged; submission uploads via Kaggle API with leaderboard score
- Reflection cites run metrics and suggests next steps
- Loop generalizes to unseen competitions without hand-written local config

**Explicit P0 constraint:** One baseline per problem type, no search, no agents, no memory across runs.

---

## Implementation Status (P0)

| Layer | Status |
|-------|--------|
| CLI + orchestrator | `run`, `init`, `build`, `resume`, `status`, `list-runs`, `doctor`; global `--verbose`/`--quiet`/`--yes` |
| Manifest / status | Crash-safe; `resume` re-runs failed/incomplete stages; `init`/`build` split |
| Competition parser | Auto-resolves title/description/metric from Kaggle API; local YAML optional override |
| Data download | Kaggle API download, unzip, per-competition cache |
| Dataset profiler | Train/test/submission roles, target, ID detection |
| Baseline selection | Infers problem type from target dtype/cardinality |
| Brief / reflection | OpenAI or Gemini via `llm/client.py`; template fallback |
| Code generation | Jinja2 templates |
| Training | Fold-fitted preprocessing + LightGBM (binary/multi-class + regression) |
| CV evaluation | Validates `cv_<metric>` from training |
| Submission | Validated against sample columns, row count, labels |
| Kaggle upload | Real API upload + leaderboard polling; `--submit` opt-in |
| Experiment logging | Local JSON store |
| Environment diagnostics | `research doctor` — Python, LightGBM, Kaggle credentials |
| Tests | Unit + mocked integration (classification, regression, resume, auto-metadata) |

---

## P0 Validation

Credentialed end-to-end runs on two independent competitions:

| Competition | Type | CV score | Public score |
|-------------|------|----------|---------------|
| Titanic | classification | `cv_accuracy` 0.7306 | 0.72488 |
| House Prices | regression | `cv_rmse` 30573.96 | 0.13259 |

**Generalization validated:** `research run --competition titanic` with no local `configs/competitions/titanic.yaml` still completes the full loop — metadata from Kaggle API, problem type inferred from profiled data.

---

## P0 Baseline Strategy

| Type | Model | Features | CV |
|------|-------|----------|-----|
| Classification | LightGBM | Fold-fitted imputation + ordinal encoding | Stratified 5-fold |
| Regression | LightGBM | Same preprocessing | 5-fold |

Template selection (`BaselineSelector._infer_problem_type`):

```
if competition.problem_type is explicitly set:
    → use it (local config override)
elif target is not numeric:
    → tabular_classification
elif target has <= 20 distinct values AND not every row unique:
    → tabular_classification
elif target is numeric:
    → tabular_regression
```

---

## P1 — v0.2 (Problem Type Expansion)

**Goal:** Same one-command loop, more competition types.

| Deliverable | Status |
|-------------|--------|
| Metric-aware evaluation (`MetricSpec.key`, `compute_metric`, LLM tie-breaker) | Done |
| Competition rules parser + `## Competition Context` in `brief.md` | Done |
| Modality auto-detection (tabular / text / image) | Done |
| `text_classification` template (TF-IDF + LogisticRegression) | Done |
| `image_classification` template (ResNet18 + LightGBM, `image` extra) | Done |
| Opt-in `*_deep` transfer-learning templates (`deep` extra, `baseline_strategy: deep`) | Done |

**Templates:** `tabular_*`, `text_classification`, `image_classification`,
`text_classification_deep`, `image_classification_deep`.

**Optional extras:** `uv sync --extra llm` (briefs), `--extra image` (image baseline),
`--extra deep` (transfer-learning baselines). See [README.md](../../README.md#optional-installs).

---

## P1 Validation

Credentialed smoke runs on five Kaggle competitions (2026-07-12). Run IDs under `runs/`.

| Competition | Modality | Template | `cv_<metric>` | Notes |
|-------------|----------|----------|---------------|-------|
| `spaceship-titanic` | tabular | `tabular_classification` | `cv_accuracy` 0.767 | `evaluation_metric.key: accuracy`; `## Competition Context` in brief; multi-class labels (`True`/`False`); `--submit` scored **0.76455** |
| `santander-customer-transaction-prediction` | tabular | `tabular_classification` | `cv_auc` 0.880 | Metric mapped from Kaggle AUC string; deadline warning logged (comp closed) |
| `nlp-getting-started` | text | `text_classification` | `cv_f1` 0.734 | `modality: text`, `text_column: text`; TF-IDF pipeline; 3263-row submission |
| `aerial-cactus-identification` | image | `image_classification` | `cv_auc` 0.998 | No `test.csv` (sample submission proxy); zip extract on cache reuse; ResNet18 + LightGBM on CPU |
| `nlp-getting-started` (deep YAML) | text | `text_classification_deep` | `cv_accuracy` 0.820 | `baseline_strategy: deep` via local YAML; DistilBERT on CPU (3-fold, clamped samples) |

**Smoke pass:** Runs 1–5 completed with expected artifacts (`competition.json`, `profile.json`, `brief.md`, `baseline_choice.json`, `metrics.json`, `submission.csv`, `reflection.md`). Run 6 upload pre-flight + Kaggle submit validated on `spaceship-titanic` (`submission_result.json` status `scored`).

**Fixes surfaced during smoke:** zip extraction on cached downloads; image competitions without `test.csv`; image filename column detection (ID column holds paths); deep templates now write lightweight `models/fold_*.json` manifests for artifact checks.

---

## P3 — v0.4 (Iteration Loop)

**Goal:** Turn reflection into action via `research improve`.

| Deliverable | Status |
|-------------|--------|
| Run fork + lineage (`parent_run_id`, `iteration` in manifest) | Done |
| `research improve --run-id <parent>` CLI | Done |
| Structured `improvement_plan.json` + `ImprovementPlanner` (auto/tune/features) | Done |
| `training_overrides.json` + tabular template parameterization | Done |
| LightGBM grid tuning (`improvement/tuner.py`) | Done |
| Feature recipes: `target_encoding`, `log_numeric` | Done |
| Cross-run diff: `research runs diff --base/--compare` | Done |

**Smoke (integration):** Titanic fixture — baseline run → `improve --strategy tune` → child run
with `improvement_plan.json`, `training_overrides.json`, new `metrics.json`, and lineage metadata;
`runs diff` reports param/metric deltas.

**Version:** `0.4.0`

---

## P4 — v1.0 (Production Quality)

**Goal:** Reliable tool for repeated competition use with CI confidence and safe validation.

| Deliverable | Status |
|-------------|--------|
| GitHub Actions CI (tabular / llm / image / deep jobs) | Done |
| Integration tests per template family (text, image, deep) | Done |
| Optional project workspace (`project.yaml`, `research workspace init/status`) | Done |
| Config layering (package → project → CLI → env) | Done |
| `--dry-run` on `run`/`build`/`improve` (codegen only) | Done |
| Runtime registry config (`configs/runtimes/`, `research runtime list/show/register/doctor`) | Done |
| Kernel slug fix (`{username}/{slug}` metadata) | Done |
| `research templates` list command | Done |

**P2 split:** Runtime **configuration** shipped in P4; remote **execution**
(`--remote-train`, scheduler, artifact sync) remains deferred — see [TODO.md](TODO.md).

**Version:** `1.0.0`
