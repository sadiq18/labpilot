# LabPilot

[![CI](https://github.com/sadiq18/labpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/sadiq18/labpilot/actions/workflows/ci.yml)

One command to solve the first 80% of a Kaggle competition — without writing code manually.

```bash
research run --competition titanic
```

After a few hours, LabPilot produces:

- Parsed competition metadata
- Downloaded and profiled dataset
- AI-generated research brief
- Baseline training pipeline
- Cross-validated model + metrics
- Validated Kaggle submission (local by default)
- Experiment log
- Reflection report with next-step recommendations
- Standalone HTML report (`report.html`)

## Quick Start

Install from source (PyPI packaging is deferred):

```bash
# Create a Python 3.11+ environment and install dependencies
uv sync --extra dev

# Configure credentials
cp .env.example .env
# Set KAGGLE_API_TOKEN in .env
# Optionally set OPENAI_API_KEY (or GEMINI_API_KEY, with LABPILOT_LLM_PROVIDER=gemini)
# for AI-generated brief.md/reflection.md — omit both and LabPilot falls back to
# template-only text instead of failing.

# Sanity-check your environment (Python, LightGBM, Kaggle credentials)
uv run research doctor

# Download Titanic data, train, evaluate, and generate submission.csv
uv run research run --competition titanic

# Upload only after inspecting the local submission
uv run research run --competition titanic --submit

# Check run status
uv run research status --run-id <run_id>

# Validate through codegen without training
uv run research run --competition titanic --dry-run --yes
```

You must join the competition and accept its rules on Kaggle before downloading data.
On macOS, LightGBM also requires `brew install libomp`.

## Optional installs

Core dependencies (LightGBM, scikit-learn, Kaggle API, etc.) are installed by default and cover
**tabular** competitions. Everything below is optional — LabPilot degrades gracefully when an
extra or API key is missing.

| Extra | Install | Enables | Required when |
|-------|---------|---------|---------------|
| `dev` | `uv sync --extra dev` | pytest, ruff, coverage | Running tests or linting locally |
| `llm` | `uv sync --extra llm` | OpenAI + Gemini clients for `brief.md` / `reflection.md` | You want AI-generated briefs (not template fallback) |
| `image` | `uv sync --extra image` | `image_classification` template (frozen ResNet18 + LightGBM) | Profiler detects an image competition and selects the lightweight image baseline |
| `deep` | `uv sync --extra deep` | `text_classification_deep` + `image_classification_deep` (fine-tuned DistilBERT / ResNet18) | Local YAML sets `baseline_strategy: deep` for a text or image competition |

Combine extras as needed:

```bash
uv sync --extra dev --extra llm              # tabular + AI briefs + tests
uv sync --extra dev --extra llm --extra image   # add image baselines
uv sync --extra dev --extra deep             # deep includes torch/torchvision/transformers (covers image too)
```

### Local CI parity

Run the same test slices as GitHub Actions:

```bash
uv sync --extra dev
uv run pytest -m "not llm and not image and not deep"   # tabular (required)
uv sync --extra llm && uv run pytest -m llm             # LLM unit tests
uv sync --extra image && uv run pytest -m image         # image integration
uv sync --extra deep && uv run pytest -m deep           # deep integration
```

**LLM (`llm` extra)** — set one API key in `.env`:

- `OPENAI_API_KEY` — default provider (`llm.provider: openai` in `configs/default.yaml`)
- `GEMINI_API_KEY` + `LABPILOT_LLM_PROVIDER=gemini` — Gemini instead

Without a key or package, `research run` warns and falls back to template text for
`brief.md`/`reflection.md` (use `--yes` to skip the prompt).

**Image / deep baselines** — no extra API keys. `research doctor` reports optional
`image`/`deep` import status; training fails fast with a clear message only when the
selected template needs a missing extra (tabular and text lightweight runs never require
`torch`).

**Deep transfer learning (opt-in)** — add to a local competition contract
(`configs/competitions/<slug>.yaml`):

```yaml
baseline_strategy: deep   # default is lightweight
```

GPU is used automatically when available; on CPU, epoch and sample counts are clamped to
keep runs bounded (see `deep_baseline` in `configs/default.yaml`).

See [configs/competitions/README.md](configs/competitions/README.md) for the full YAML schema.

## Commands

Global flags (apply to every command, placed before the subcommand):

| Flag | Description |
|------|-------------|
| `--verbose` / `-v` | Debug logging for every stage |
| `--quiet` / `-q` | Only log warnings and errors |

### `research run --competition <slug>`

Runs the full pipeline start to finish in one call: parse → download → profile → brief →
baseline → code → train → evaluate → submission → upload → log → reflection → HTML report.

| Option | Description |
|--------|-------------|
| `--competition, -c` | Kaggle competition slug (required) |
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory |
| `--competitions-dir` | Directory with local per-competition contracts (`<slug>.yaml`); see [configs/competitions/README.md](configs/competitions/README.md) |
| `--submit` | Upload the validated submission to Kaggle (disabled by default) |
| `--force-submit` | With `--submit`: upload even when the competition deadline has passed |
| `--yes, -y` | Skip confirmation prompts, e.g. proceed without LLM if unavailable |

### `research init --competition <slug>`

Runs just the first half — parse → download → profile → brief — then stops, so you can review
`competition.json`, the dataset profile, and `brief.md` before committing to training. Continue
with `research build`. Takes the same options as `run` except `--submit` (nothing is uploaded
yet).

### `research build --run-id <id>`

Runs the second half of an already-`init`'d run — baseline → code → train → evaluate →
submission → upload → log → reflection. Fails fast with a clear message if `init` hasn't
finished yet.

| Option | Description |
|--------|-------------|
| `--run-id, -r` | Run ID to build, from `research init` (required) |
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory (must match the `init` call's) |
| `--submit` | Upload the validated submission to Kaggle (disabled by default) |
| `--force-submit` | With `--submit`: upload even when the competition deadline has passed |
| `--yes, -y` | Skip confirmation prompts, e.g. proceed without LLM if unavailable |

### `research workspace init --name <name>`

Create a multi-competition project with `project.yaml`, `runs/`, `competitions/`, and
`configs/runtimes/`. Auto-detected when you run pipeline commands from the project directory
(or pass `--project-dir`).

```bash
research workspace init --name kaggle-2026
research workspace status
research run --competition titanic --project-dir .
```

### `research runtime list|show|register|doctor`

Manage training runtime profiles (local, Kaggle kernel, Google Colab, other). **P2 v0.3**
ships registry + validation; training still runs locally until **P2 execution**
(`--remote-train`) lands.

```bash
research runtime list
research runtime register --provider kaggle_kernel --id kaggle-gpu-free
research runtime doctor
```

See [configs/runtimes/README.md](configs/runtimes/README.md) for the schema.

### `research templates`

List registered baseline templates (tabular, text, image, deep variants).

### `research run --dry-run`

Add `--dry-run` to `run`, `build`, or `improve` to validate through code generation without
training or submission. Produces `pipeline/train.py` and `dry_run.json`. Mutually exclusive
with `--submit`.

### `research improve --run-id <parent>`

Fork a **completed** parent run, reuse init artifacts (competition, data, profile, brief),
apply an improvement plan, and re-run from code generation through reflection.

| Option | Description |
|--------|-------------|
| `--run-id, -r` | Parent run ID (required; must be `completed`) |
| `--strategy` | `auto` (LLM plan, fallback to tune), `tune` (LightGBM grid), or `features` |
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory |
| `--submit` | Upload the child run's submission to Kaggle |
| `--force-submit` | With `--submit`: upload even when the deadline has passed |
| `--yes, -y` | Skip confirmation prompts |

```bash
# Auto-plan from reflection + metrics, fork, retrain
research improve --run-id 20260712-014250-spaceship-titanic

# Explicit tuning strategy
research improve --run-id <parent> --strategy tune

# Compare parent vs child
research runs diff --base <parent> --compare <child>
```

Child runs record lineage in `manifest.json` (`parent_run_id`, `iteration`) and persist
`improvement_plan.json` plus `training_overrides.json` (model params and feature recipes).

### `research runs diff --base <a> --compare <b>`

Side-by-side comparison of two runs: CV metrics, param deltas, lineage, and submission status.

### `research experiments graph --competition <slug>` / `research experiments show <run_id>`

Explore the experiment graph (Milestone 2, Plan 1): every `run`/`init`/`improve` invocation
records a best-effort `git_commit` and a `config.json` snapshot of the config actually used, on
top of the existing `parent_run_id`/`iteration` lineage in `manifest.json` metadata.

```bash
# ASCII lineage tree for a competition, annotated with a metric and the best-scoring path
research experiments graph --competition titanic --metric cv_accuracy

# Single-experiment detail view: status, progress, description, artifacts, metrics
research experiments show <run_id>
research experiments show <run_id> --format json
```

| Option | Description |
|--------|-------------|
| `--competition, -c` | Kaggle competition slug (`graph` only, required) |
| `--metric` | Metric key to annotate scores and highlight the best root-to-leaf path (`graph` only) |
| `--format` | `table` (default) or `json` (`show` only) |
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory |

### `research hypothesis add|list|show|update`

Manual structured hypotheses for a competition (Milestone 2, Plan 2). Stored as
`knowledge/<slug>/hypotheses/H-NNN.json` (gitignored). Attach one to a run with
`--hypothesis H-001` on `research run` or `research improve` (auto-marks `proposed` →
`testing`).

```bash
research hypothesis add --competition titanic \
  --observation "Rare classes perform poorly" \
  --reason "Dataset imbalance" \
  --prediction "Focal Loss will improve Macro F1" \
  --confidence 0.74 \
  --tags loss,class-imbalance

research hypothesis list --competition titanic [--status testing]
research hypothesis show H-001 --competition titanic
research hypothesis update H-001 --competition titanic --status confirmed --evidence-run <run_id>

research run --competition titanic --hypothesis H-001
research improve --run-id <parent> --hypothesis H-001 --strategy tune
```

| Option | Description |
|--------|-------------|
| `--competition, -c` | Kaggle competition slug (required on all hypothesis commands) |
| `--status` | Filter (`list`) or set (`update`): proposed, testing, confirmed, rejected, inconclusive |
| `--evidence-run` | Run id appended to `evidence_for` (confirmed) or `evidence_against` (rejected) |
| `--knowledge-dir` | Override the knowledge directory (default: `knowledge/`) |

### `research resume --run-id <id>`

Resumes a run from its first failed or incomplete stage. Stages already `completed` or
`skipped` are left untouched; everything else (failed, stuck "running" from a killed process,
or never reached) is re-executed in pipeline order. One exception: if the run finished with
`upload_submission` left as `skipped` (i.e. it ran without `--submit`) and you now pass
`--submit`, that stage is re-run for real instead of staying skipped — so `research resume
--run-id <id> --submit` is the command to upload a submission after the fact, without
re-running the rest of an already-completed pipeline. Add `--force-submit` when the
competition deadline has passed but Kaggle may still accept uploads.

**Kernel-only competitions** (e.g. `aerial-cactus-identification`): LabPilot detects
`submission_mode: kernel` from Kaggle metadata (with a rules-page fallback). Training still
runs locally; `export_kernel` writes `runs/<id>/kernel/` for Kaggle's notebook API. The same
`--submit` flag pushes the kernel, waits for the run, submits via `competition_submit_code`,
and polls the leaderboard. Without `--submit`, `submission_result.json` is written with status
`kernel_ready`. After upload, the CLI and `reflection.md` include links to the submissions
page and kernel notebook.

| Option | Description |
|--------|-------------|
| `--run-id, -r` | Run ID to resume (required) |
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory (must match the original run's) |
| `--competitions-dir` | Directory with local per-competition contracts |
| `--submit` | Upload the validated submission to Kaggle (disabled by default) |
| `--force-submit` | With `--submit`: upload even when the competition deadline has passed |
| `--yes, -y` | Skip confirmation prompts, e.g. proceed without LLM if unavailable |

`run`/`init`/`build`/`resume` all check LLM availability once up front (before doing any work):
if neither `OPENAI_API_KEY` nor `GEMINI_API_KEY` is set (or the matching optional package isn't
installed), they print a warning and ask for confirmation before continuing with template-only
`brief.md`/`reflection.md` — pass `--yes` to skip the prompt (also skipped automatically for
non-interactive/CI runs).

### `research doctor`

Checks that the local environment has everything LabPilot needs (Python version, LightGBM
import, Kaggle credentials) and also reports optional `image`/`deep` dependency status.
Exits non-zero if a **core** check fails. `run`/`init`/`build`/`resume` run core checks
automatically and fail fast on a bad environment; optional extras are informational only
unless the selected baseline template requires them at train time.

### `research status --run-id <id>`

Shows the per-stage status and artifacts for one run.

| Option | Description |
|--------|-------------|
| `--run-id, -r` | Run ID to inspect (required) |
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory |

### `research report --run-id <id>`

Generate or refresh the standalone HTML report (`report.html`) from existing run artifacts.
The pipeline writes this automatically at the end of every completed build/run.

| Option | Description |
|--------|-------------|
| `--run-id, -r` | Run ID to render (required) |
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory |

### `research list-runs`

Lists every run under the runs directory with its competition and overall status
(`running` / `partial` / `completed` / `failed`).

| Option | Description |
|--------|-------------|
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory |

## Repository Layout

```
src/labpilot/     Core engine modules
templates/        Baseline code templates (Jinja2)
configs/          Default configuration
runs/             Generated run artifacts (gitignored)
docs/             Architecture and milestone docs
tests/            Unit and integration tests
```

## License

MIT — see [LICENSE](LICENSE).
