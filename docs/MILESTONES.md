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

**Validation competition:** any tabular binary/multi-class classification competition with a
train/test/sample-submission split. Titanic is used as the first end-to-end validation target
because it is small and free of licensing friction, but nothing in the design is Titanic-specific
— see `configs/competitions/README.md` for how a new competition's contract is supplied locally.

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
| Competition parser | Reads a local, per-competition contract (`configs/competitions/<slug>.yaml`) |
| Data download | Kaggle API download, unzip, and per-competition cache implemented |
| Dataset profiler | Detects train/test/submission roles, target, and ID (patterns overridable per competition) |
| Brief / reflection | Fallback text only — no LLM calls |
| Baseline selection | Works from competition and dataset contracts |
| Code generation | Works — renders Jinja2 templates |
| Training | Fold-fitted preprocessing + LightGBM binary classifier |
| CV evaluation | Validates real `cv_accuracy` from training |
| Submission | Validated against sample columns, row count, and labels |
| Kaggle upload | Real API upload with leaderboard score polling, explicit `--submit` opt-in |
| Experiment logging | Works |
| Tests | Unit tests plus mocked end-to-end pipeline runs for both classification and regression |

---

## P0 Pending Tasks

### P0 is complete

The core loop has been credentialed-smoke-tested end-to-end on two real Kaggle competitions:

- **Titanic** (tabular classification) — real download, training, local submission, `--submit`
  upload accepted by Kaggle, and a persisted public score (`0.72488`).
- **House Prices - Advanced Regression Techniques** (tabular regression) — same, proving the
  regression baseline path and generic competition-contract resolution on a second, unrelated
  competition. Real download, training, local submission, `--submit` upload accepted by Kaggle,
  and a persisted public score (`0.13259`).

Both runs exercised every P0 stage (parse → download → profile → brief → baseline → code →
train → evaluate → submission → upload → log → reflection) without manual code edits.

Bugs found and fixed along the way (see git history for details): a relative `run_dir` was
double-resolved when the generated script ran as a subprocess; `kaggle>=2.0` raises `SystemExit`
(not a catchable `Exception`) on missing credentials, which used to leave a run's manifest stuck
at `"running"` forever; the regression template guessed its target column instead of using the
inferred contract and used a train/test-inconsistent category encoding; `SubmissionValidator`
always required an integer target, which is wrong for regression; and `mean_squared_error(...,
squared=False)` no longer exists in current scikit-learn. `KaggleClient.upload_submission` also
did not poll for the leaderboard score, which is why it's persisted now.

### Generalization and CLI ergonomics (deferred to P1/P3, not blockers)

- Automatic competition metadata resolution from the Kaggle URL/slug (remove the need for a
  hand-written local contract file)
- LLM-backed brief/reflection (`OpenAI` call in `BriefGenerator.generate()` /
  `ReflectionGenerator.generate()` — currently accurate but template-based fallback text)
- Multi-class classification support
- `--resume --run-id <id>` — restart from failed stage
- `--verbose`/`--quiet` flag to control log level across all major classes
- Clearer environment diagnostics for Python and LightGBM

---

## Completed Executable-Baseline Slice

Validated end-to-end against two independent, real competitions (Titanic and House Prices);
nothing below is specific to either one.

```
✓ Python 3.11+ project environment and macOS libomp setup
✓ Kaggle download, archive extraction, and per-competition cache
✓ Train/test/sample submission detection (overridable file-name patterns)
✓ Target, ID, metric, and submission contract
✓ Fold-fitted preprocessing and LightGBM training (classification + regression)
✓ Real CV score and fail-hard artifact validation
✓ Correct local submission generation
✓ Opt-in Kaggle upload with --submit, accepted by Kaggle on both competitions
✓ Public leaderboard score polled and persisted after upload
✓ Mocked end-to-end tests for local and submitted modes
```

---

## P0 Validation Checklist

All items validated for real, with credentials, on two independent competitions
(Titanic and House Prices):

```
[x] uv sync --extra dev succeeds
[x] Mocked run downloads the 3 dataset file roles (train/test/sample submission)
[x] profile.json correctly infers the target and ID columns
[x] pipeline/train.py runs without manual edits
[x] metrics.json has real cv_<metric>
[x] submission.csv row count and columns match the sample
[x] Upload is skipped unless --submit is provided
[x] Credentialed Kaggle download succeeds
[x] Kaggle accepts the submission
[x] submission_result.json has a public score
[x] reflection.md references actual metrics
```

**P0 — Research Engine v0.1 (Core Loop Proof) is complete.**

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
