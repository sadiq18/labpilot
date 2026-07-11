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

**Goal:** Prove the full pipeline works end-to-end for **one competition archetype** (tabular classification/regression).

| Dimension | Target |
|-----------|--------|
| Command | `research run --competition <slug>` |
| Problem types | Tabular only (classification + regression) |
| Human input | Kaggle API credentials + LLM API key |
| Runtime | 1–4 hours unattended |
| Proof | 1 real Kaggle competition completes full loop |

**Success criteria:**

- Single command runs start-to-finish without manual code edits
- All 8 P0 capabilities produce real artifacts on disk
- CV score is logged and aligns with the competition metric direction
- Submission uploads successfully via Kaggle API
- Reflection cites actual run metrics and suggests 3–5 concrete next steps

**Explicit P0 constraint:** One baseline per problem type, no search, no agents, no memory across runs.

**First target competition:** Titanic (tabular binary classification).

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

- Retry/resume from failed pipeline stages
- Leaderboard polling and score tracking
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

### In Scope

- One-command competition initialization
- Automatic dataset profiling
- AI-generated research brief
- Baseline template selection
- Training and evaluation
- Kaggle submission
- Experiment tracking
- Reflection report with next-step recommendations

### Out of Scope

See [Future (Explicitly Deferred)](#future-explicitly-deferred) above.

---

## Current Implementation Status

| Layer | Status |
|-------|--------|
| CLI + orchestrator | Wired — all 12 stages run in sequence |
| Manifest / status | Works, including skipped upload stages |
| Competition parser | Titanic metadata contract implemented |
| Data download | Kaggle API download and unzip implemented |
| Dataset profiler | Detects train/test/submission roles, target, and ID |
| Brief / reflection | Fallback text only — no LLM calls |
| Baseline selection | Works from competition and dataset contracts |
| Code generation | Works — renders Jinja2 templates |
| Training | Fold-fitted preprocessing + LightGBM binary classifier |
| CV evaluation | Validates real `cv_accuracy` from training |
| Submission | Validated against sample columns, row count, and labels |
| Kaggle upload | Real API upload, explicit `--submit` opt-in |
| Experiment logging | Works |
| Tests | Unit tests plus mocked end-to-end Titanic pipeline |

---

## P0 Pending Tasks

### Required before P0 is complete

#### 1. Credentialed Titanic smoke run

- Join Titanic and accept its rules on Kaggle
- Configure `KAGGLE_API_TOKEN` (preferred) or legacy username/key credentials
- Run once without `--submit` and inspect `metrics.json` and `submission.csv`
- Run separately with `--submit` and verify Kaggle accepts the file
- Poll and persist the public leaderboard score

#### 2. LLM brief and reflection

Both generators have `TODO` and use fallback markdown. Implement:

- OpenAI call in `BriefGenerator.generate()`
- OpenAI call in `ReflectionGenerator.generate()`

#### 3. Generalization and CLI ergonomics

- Generic competition metadata resolution beyond Titanic
- Multi-class classification support
- `--resume --run-id <id>` — restart from failed stage
- Public leaderboard score polling
- Clearer environment diagnostics for Python and LightGBM

---

## Completed Executable-Titanic Slice

```
✓ Python 3.11+ project environment and macOS libomp setup
✓ Kaggle download and archive extraction
✓ Train/test/sample submission detection
✓ Target, ID, metric, and submission contract
✓ Fold-fitted preprocessing and LightGBM training
✓ Real CV accuracy and fail-hard artifact validation
✓ Correct local submission generation
✓ Opt-in Kaggle upload with --submit
✓ Mocked end-to-end tests for local and submitted modes
```

---

## P0 Validation Checklist

Before declaring P0 complete on one contest:

```
[x] uv sync --extra dev succeeds
[x] Mocked run downloads the 3 Titanic CSV roles
[x] profile.json has target=Survived, id=PassengerId
[x] pipeline/train.py runs without manual edits
[x] metrics.json has real cv_accuracy
[x] submission.csv row count and columns match the sample
[x] Upload is skipped unless --submit is provided
[ ] Credentialed Kaggle download succeeds
[ ] Kaggle accepts the submission
[ ] submission_result.json has a public score
[ ] reflection.md references actual metrics
```

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
