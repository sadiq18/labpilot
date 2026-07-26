# LabPilot

[![CI](https://github.com/sadiq18/labpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/sadiq18/labpilot/actions/workflows/ci.yml)

Plan-driven Kaggle research OS: analyze the competition, compile a research plan,
then let the **Research Engineer** implement, verify, train, evaluate, and package
a submission — without hand-writing the baseline.

```bash
research analyze <slug>
research plan create <slug> --baseline
research run --plan P-001 --competition <slug>
```

After a successful execution you get:

- Competition workspace under `competitions/<slug>/` (data, profile, pipeline code)
- Knowledge under `knowledge/<slug>/research/` (analyze brief, plans, executions, evidence)
- Cross-validated metrics and a packaged submission (upload only with `--submit`)
- Execution evidence for every plan task

## Quick Start

Install from source (PyPI packaging is deferred):

```bash
# Create a Python 3.11+ environment and install dependencies
uv sync --extra dev

# Configure credentials
cp .env.example .env
# Set KAGGLE_API_TOKEN in .env
# Optionally set OPENAI_API_KEY (or GEMINI_API_KEY + LABPILOT_LLM_PROVIDER=gemini)
# for LLM-assisted analyze / code / reflection — omit and LabPilot falls back to
# deterministic rule_engine / templates.

# Sanity-check your environment (Python, LightGBM, Kaggle credentials)
uv run research doctor

# Happy path (SoR)
uv run research analyze titanic
uv run research plan create titanic --baseline          # → P-001
uv run research run --plan P-001 --competition titanic

# Dry-run: syntax smoke + stub path; no full train / no upload
uv run research run --plan P-001 --competition titanic --dry-run --no-install-packages

# Upload only after inspecting local metrics + submission.csv
uv run research run --plan P-001 --competition titanic --submit

# Resume an interrupted execution
uv run research resume --execution E-001 --competition titanic
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
| `llm` | `uv sync --extra llm` | OpenAI + Gemini clients | You want LLM-assisted analyze / code / narrative |
| `image` | `uv sync --extra image` | `image_classification` template (frozen ResNet18 + LightGBM) | Profiler selects the lightweight image baseline |
| `deep` | `uv sync --extra deep` | `text_classification_deep` + `image_classification_deep` | Local YAML sets `baseline_strategy: deep` |

Combine extras as needed:

```bash
uv sync --extra dev --extra llm
uv sync --extra dev --extra llm --extra image
uv sync --extra dev --extra deep   # deep includes torch/torchvision/transformers
```

### Local CI parity

Same slices as GitHub Actions (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```bash
uv sync --extra dev
uv run pytest -m "not llm and not image and not deep"   # tabular (required)
uv sync --extra llm && uv run pytest -m llm tests/unit/test_llm_client.py
uv sync --extra image && uv run pytest -m image         # optional; may be empty
uv sync --extra deep && uv run pytest -m deep           # optional; may be empty
```

**LLM (`llm` extra)** — set one API key in `.env`:

- `OPENAI_API_KEY` — default (`llm.provider: openai` in `configs/default.yaml`)
- `GEMINI_API_KEY` + `LABPILOT_LLM_PROVIDER=gemini` — Gemini instead

Without a key or package, Micro Agents / brief paths fall back to `rule_engine` templates.

**Deep transfer learning (opt-in)** — in a local competition contract
(`configs/competitions/<slug>.yaml`):

```yaml
baseline_strategy: deep   # default is lightweight
```

See [configs/competitions/README.md](configs/competitions/README.md) for the full YAML schema.

## Commands

Full flag tables: **[docs/CLI.md](docs/CLI.md)**.  
Day-to-day procedure: **[docs/SOP.md](docs/SOP.md)**.

| I want to… | Command |
|------------|---------|
| Landscape + Research Brief | `research analyze <slug>` |
| Baseline plan (P-001) | `research plan create <slug> --baseline` |
| Hypothesis plan | `research plan create <slug> --hypothesis H-xxx` |
| Execute an approved plan | `research run --plan P-001 --competition <slug>` |
| Resume an execution | `research resume --execution E-001 --competition <slug>` |
| Environment check | `research doctor` |
| Lineage / compare / KB / rank | `research experiments …` — [CLI.md](docs/CLI.md#4-experiments) |
| Hypotheses | `research hypothesize <slug>` / `list` / `show` / `update` |
| Inspect legacy `runs/` HTML | `research report --run-id <id>` (historical artifacts) |
| Competition HTML | `research experiments dashboard --competition <slug>` |

Global flags (before the subcommand): `--verbose` / `-v`, `--quiet` / `-q`.

Default policy: **do not pass `--submit`** until you have inspected local metrics and
`submission.csv`. Use `--dry-run` on `run` / `resume` to validate without full train/upload.

Legacy linear Pipeline (`init` / `build` / `improve` / plan-less `run`) has been
**removed**. Tracker:
[docs/milestones/research-engineer/pipeline-deprecation.md](docs/milestones/research-engineer/pipeline-deprecation.md).

## Repository Layout

```
src/labpilot/
  accessor/           # SQLite, Kaggle client, data download, profiler, common helpers
  research_engine/
    intelligence/     # analyze → knowledge → hypothesize
    planner/          # hypothesis / baseline → ResearchPlan DAG
    execution/        # Research Engineer (capabilities, templates, evidence)
  experiments/        # experiment graph, hypotheses helpers, run manifests, compare
  cli/                # Typer entrypoint
configs/              # default.yaml, competitions/, runtimes/
competitions/         # per-slug workspace (gitignored when generated)
knowledge/            # research DB, plans, executions, hypotheses (gitignored)
runs/                 # legacy run artifacts still readable for inspect (gitignored)
docs/                 # Architecture, CLI, SOP, milestones
tests/                # Unit tests
```

## Docs

| Doc | Contents |
|-----|----------|
| [docs/SOP.md](docs/SOP.md) | How to use LabPilot (procedure) |
| [docs/CLI.md](docs/CLI.md) | All `research` commands + examples |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Modules, execution flow, artifact contracts |
| [docs/MILESTONES.md](docs/MILESTONES.md) | Roadmap / completed / backlog |
| [docs/milestones/research-engineer/](docs/milestones/research-engineer/) | Engineer design + deprecation notes |

## License

MIT — see [LICENSE](LICENSE).
