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
| `--yes, -y` | Skip confirmation prompts, e.g. proceed without LLM if unavailable |

### `research resume --run-id <id>`

Resumes a run from its first failed or incomplete stage. Stages already `completed` or
`skipped` are left untouched; everything else (failed, stuck "running" from a killed process,
or never reached) is re-executed in pipeline order. One exception: if the run finished with
`upload_submission` left as `skipped` (i.e. it ran without `--submit`) and you now pass
`--submit`, that stage is re-run for real instead of staying skipped — so `research resume
--run-id <id> --submit` is the command to upload a submission after the fact, without
re-running the rest of an already-completed pipeline.

| Option | Description |
|--------|-------------|
| `--run-id, -r` | Run ID to resume (required) |
| `--config` | Path to config file (default: `configs/default.yaml`) |
| `--runs-dir` | Override the runs directory (must match the original run's) |
| `--competitions-dir` | Directory with local per-competition contracts |
| `--submit` | Upload the validated submission to Kaggle (disabled by default) |
| `--yes, -y` | Skip confirmation prompts, e.g. proceed without LLM if unavailable |

`run`/`init`/`build`/`resume` all check LLM availability once up front (before doing any work):
if neither `OPENAI_API_KEY` nor `GEMINI_API_KEY` is set (or the matching optional package isn't
installed), they print a warning and ask for confirmation before continuing with template-only
`brief.md`/`reflection.md` — pass `--yes` to skip the prompt (also skipped automatically for
non-interactive/CI runs).

### `research doctor`

Checks that the local environment has everything LabPilot needs (Python version, LightGBM
import, Kaggle credentials) and exits non-zero if anything's missing. `run`/`init`/`build`/
`resume` run this automatically and fail fast on a bad environment.

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

**P0 — Research Engine v0.1** (in progress): tabular competitions only.

See [docs/MILESTONES.md](docs/MILESTONES.md) for the roadmap, P0 scope, and pending tasks, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module design.

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
