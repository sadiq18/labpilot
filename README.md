# LabPilot

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

## Quick Start

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
baseline → code → train → evaluate → submission → upload → log → reflection.

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

### `research list-runs`

Lists every run under the runs directory with its competition and overall status
(`running` / `partial` / `completed` / `failed`).

| Option | Description |
|--------|-------------|
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory |

## Project Status

**P0 — Research Engine v0.1** (complete): tabular classification/regression end-to-end.

**P1 — Problem Type Expansion v0.2** (complete): metric-aware evaluation, competition rules
in `brief.md`, modality detection (tabular/text/image), NLP and image baselines, opt-in deep
transfer-learning templates. See [Optional installs](#optional-installs) for `llm` / `image` /
`deep` extras.

See [docs/MILESTONES.md](docs/MILESTONES.md) for the roadmap (split into
[Completed](docs/milestones/COMPLETED.md), [In progress](docs/milestones/IN-PROGRESS.md),
and [TODO](docs/milestones/TODO.md)), and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for
module design.

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
