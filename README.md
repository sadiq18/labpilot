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

Full flag tables and examples for every subcommand: **[docs/CLI.md](docs/CLI.md)**.

How to run LabPilot day-to-day (baseline → improve → rank → submit):
**[docs/SOP.md](docs/SOP.md)**.

| I want to… | Command |
|------------|---------|
| Full pipeline | `research run --competition <slug>` |
| Review brief before train | `research init` → `research build --run-id <id>` |
| Resume / upload later | `research resume --run-id <id> [--submit]` |
| Fork + retrain | `research improve --run-id <parent>` (`--strategy auto`, `tune`, or `features`) |
| Environment check | `research doctor` |
| Lineage / compare / KB / rank / search | `research experiments …` — see [CLI.md](docs/CLI.md#4-experiments) |
| Hypotheses | `research hypothesize <slug>` to generate; `list` / `show` / `update` to manage — see [CLI.md](docs/CLI.md#5-hypotheses) |
| Per-run HTML | `research report --run-id <id>` |
| Competition HTML | `research experiments dashboard --competition <slug>` |

Global flags (before the subcommand): `--verbose` / `-v`, `--quiet` / `-q`.

Default policy: **do not pass `--submit`** until you have inspected local metrics and
`submission.csv`. Use `--dry-run` on `run` / `build` / `improve` to validate through
codegen without training.

Workspace / runtime / templates details: [CLI.md §6](docs/CLI.md#6-environment--project),
[configs/runtimes/README.md](configs/runtimes/README.md),
[configs/competitions/README.md](configs/competitions/README.md).

## Repository Layout

```
src/labpilot/     Core engine modules
templates/        Baseline code templates (Jinja2)
configs/          Default configuration
runs/             Generated run artifacts (gitignored)
knowledge/        Hypotheses + knowledge base + dashboard (gitignored)
docs/             Architecture, CLI reference, SOP, milestones
tests/            Unit and integration tests
```

## Docs

| Doc | Contents |
|-----|----------|
| [docs/SOP.md](docs/SOP.md) | How to use LabPilot (procedure) |
| [docs/CLI.md](docs/CLI.md) | All `research` commands + examples |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Modules, stages, artifact contracts |
| [docs/MILESTONES.md](docs/MILESTONES.md) | Roadmap / completed / backlog |

## License

MIT — see [LICENSE](LICENSE).
