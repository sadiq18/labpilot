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
- Config overrides (`--config`, `--dry-run`, `--skip-upload`)
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
| Manifest / status | Works |
| Competition parser | Stub — placeholder metadata only |
| Data download | Stub — writes `README.txt`, no real data |
| Dataset profiler | Partial — profiles largest CSV, no target/id detection |
| Brief / reflection | Fallback text only — no LLM calls |
| Baseline selection | Works — defaults to `tabular_classification` |
| Code generation | Works — renders Jinja2 templates |
| Training | Template exists — target/submission logic is fragile |
| CV evaluation | Stub — does not compute metrics |
| Submission | Stub — writes fake row if training did not produce one |
| Kaggle upload | Stub — status `"pending"`, no API call |
| Experiment logging | Works |
| Tests | 2 unit tests only — no integration test |

---

## P0 Pending Tasks

### Blockers (must fix for a real run)

#### 1. Kaggle data download

`DataDownloader` is a placeholder. Implement:

- `KaggleApi.competition_download_files(competition, path, force=True)`
- Unzip into `runs/<id>/data/raw/`
- Wire `KAGGLE_USERNAME` / `KAGGLE_KEY` from `.env` (or write `~/.kaggle/kaggle.json`)

#### 2. Kaggle submission upload

`KaggleClient.upload_submission` returns a fake result. Implement:

- `api.competition_submit(submission_path, message, competition)`
- Optionally poll for public leaderboard score
- Persist real `submission_result.json`

#### 3. Target and ID column detection

The profiler never sets `target_column` or `id_column`. Implement:

- Identify `train*.csv`, `test*.csv`, `sample_submission*.csv` separately (do not profile the largest CSV)
- Infer target from columns in train but not in test, or from `sample_submission.csv`
- Infer ID column from the first column of `sample_submission.csv`
- Pass detected columns into `baseline_choice.json` and training templates

#### 4. Training template target heuristic

Templates assume target = last column. For Titanic, last column is `Embarked`, not `Survived`. Implement:

- Render `target_col` and `id_col` from `BaselineChoice` into templates
- Use `sample_submission.csv` column names for the submission file
- Handle multi-class (not only binary `predict_proba[:, 1]`)

#### 5. Remove stub evaluate/submission stages

`evaluate_cv` and `generate_submission` write placeholder data when artifacts are missing, masking failures. Implement:

- `evaluate_cv`: read `metrics.json` from training, fail if missing or invalid
- `generate_submission`: use `SubmissionFormatter` with correct columns; fail if pipeline did not produce predictions
- Remove placeholder fallbacks

#### 6. Competition metadata (minimum viable)

Parser returns `problem_type: unknown` and `metric: unknown`. Implement:

- Competition-specific config for the first target (e.g. `configs/competitions/titanic.yaml`), or
- Kaggle API + page scrape for metric and submission format

#### 7. Environment setup

- `pyproject.toml` requires Python ≥3.11
- LightGBM on macOS may need `libomp` (`brew install libomp`)
- Package must be installable before any real run

---

### Important (not blockers, but required for P0 "done")

#### 8. LLM brief and reflection

Both generators have `TODO` and use fallback markdown. Implement:

- OpenAI call in `BriefGenerator.generate()`
- OpenAI call in `ReflectionGenerator.generate()`
- Read `OPENAI_API_KEY` from `Settings` (class exists but is unused)

#### 9. Credentials integration

`Settings` with env vars exists but `load_config()` never uses them. Implement:

- Merge YAML config + `.env` settings in one place
- Validate credentials at CLI startup with clear error messages

#### 10. Integration test

Only manifest and profiler unit tests exist. Add:

- Fixture with synthetic `train.csv` / `test.csv` / `sample_submission.csv`
- Full pipeline test with mocked Kaggle API
- Optional `--dry-run` mode that skips upload

#### 11. CLI ergonomics

- `--skip-upload` — train locally without submitting
- `--resume --run-id <id>` — restart from failed stage
- Pre-flight check: credentials, Python version, competition slug

---

## Suggested Implementation Order

```
1. Fix Python env + pip install
2. Kaggle download
3. Train/test/sample file detection
4. Target + ID + submission format
5. Fix training template rendering
6. Remove stub evaluate/submission stages
7. Kaggle upload
8. End-to-end test on Titanic
9. LLM brief + reflection
10. Generic competition parser
```

**Minimum for a real Kaggle submission:** steps 1–7.  
**Minimum to call P0 done:** steps 1–9.

---

## P0 Validation Checklist

Before declaring P0 complete on one contest:

```
[ ] pip install -e ".[dev,llm]" succeeds
[ ] KAGGLE_USERNAME + KAGGLE_KEY in .env work
[ ] research run --competition titanic downloads 3 CSVs
[ ] profile.json has target=Survived, id=PassengerId
[ ] pipeline/train.py runs without manual edits
[ ] metrics.json has real cv_auc (not 0.0 placeholder)
[ ] submission.csv has 418 rows, correct columns
[ ] Kaggle accepts the submission
[ ] submission_result.json has a public score
[ ] reflection.md references actual metrics
```

---

## P0 Baseline Strategy

One template per tabular type — no search, no AutoML:

| Type | Model | Features | CV |
|------|-------|----------|-----|
| Classification | LightGBM | Label-encode categoricals, median impute numerics | Stratified 5-fold |
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
