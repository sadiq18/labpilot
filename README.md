# LabPilot

[![CI](https://github.com/sadiq18/labpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/sadiq18/labpilot/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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

## How it works

LabPilot separates *deciding what to try* from *building it*:

1. **Analyze** — download the competition, profile the data, and write a Research Brief.
2. **Hypothesize** — turn the brief (plus literature/repo search, when keys are set) into
   ranked, testable hypotheses.
3. **Plan** — compile a hypothesis (or the baseline) into a `ResearchPlan` DAG of tasks.
4. **Execute** — the Research Engineer generates pipeline code, verifies it, trains,
   cross-validates, and packages `submission.csv`, recording evidence per task.
5. **Reflect** — score the result against the hypothesis, update the knowledge base,
   and feed the next round.

Every step is a separate command, so you can drive the loop by hand — or hand the whole
loop to the Conductor (see [Autonomous campaigns](#autonomous-campaigns)).

## Requirements

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for dependency management
- A **Kaggle account** with an API token — LabPilot downloads data and (optionally) submits on your behalf
- macOS only: `brew install libomp` (LightGBM needs OpenMP)
- Optional: an LLM API key (Gemini or OpenAI), or a local [Ollama](https://ollama.com) for coding tasks.
  Without one, LabPilot falls back to deterministic `rule_engine` / template paths.

## Install

Install from source (PyPI packaging is deferred):

```bash
git clone https://github.com/sadiq18/labpilot.git
cd labpilot
uv sync --extra dev
```

Then configure credentials:

```bash
cp .env.example .env
# Set KAGGLE_API_TOKEN in .env (create one at https://www.kaggle.com/settings)
# Optionally set GEMINI_API_KEY (default) or OPENAI_API_KEY + LABPILOT_LLM_PROVIDER=openai
```

Sanity-check the environment before your first run:

```bash
uv run research doctor
```

Every command below is shown as `uv run research …`. If you prefer a bare `research`,
activate the environment first (`source .venv/bin/activate`).

## Quick start

```bash
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

Default policy: **do not pass `--submit`** until you have inspected local metrics and
`submission.csv`. Use `--dry-run` on `run` / `resume` to validate without full train/upload.

## Autonomous campaigns

`research conduct` runs the whole loop — hypothesize, plan, execute, reflect, repeat —
against a goal, stopping at approval gates and budget/failure breakers:

```bash
uv run research conduct run "Win titanic" --competition titanic
uv run research conduct continue          # resume the latest active session
```

Useful flags: `--max-steps N`, `--branches K` (test the top K hypotheses in parallel git
worktrees), `--yes` (auto-approve gated tools), `--offline`. Start with a small
`--max-steps` and no `--yes` until you trust the loop on your competition.
Operator guide: [docs/research-os/COMPETITION-SOP.md](docs/research-os/COMPETITION-SOP.md).

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

**LLM (`llm` extra)** — set one API key in `.env`:

- `GEMINI_API_KEY` — default (`llm.provider: gemini` in `configs/default.yaml`)
- `OPENAI_API_KEY` + `LABPILOT_LLM_PROVIDER=openai` — OpenAI instead
- Coding task uses local Ollama (`qwen2.5-coder:14b`) when `force_local` is set

Without a key/package (and without reachable Ollama for local tasks), Micro Agents /
brief paths fall back to `rule_engine` templates.

**Deep transfer learning (opt-in)** — in a local competition contract
(`configs/competitions/<slug>.yaml`):

```yaml
baseline_strategy: deep   # default is lightweight
```

See [configs/competitions/README.md](configs/competitions/README.md) for the full YAML schema.

## Commands

Full flag tables: **[docs/research-pipeline/CLI.md](docs/research-pipeline/CLI.md)**.  
Day-to-day procedure: **[docs/research-pipeline/SOP.md](docs/research-pipeline/SOP.md)**.  
Docs index: **[docs/README.md](docs/README.md)** (pipeline vs Research OS).

| I want to… | Command |
|------------|---------|
| Landscape + Research Brief | `research analyze <slug>` |
| Baseline plan (P-001) | `research plan create <slug> --baseline` |
| Hypothesis plan | `research plan create <slug> --hypothesis H-xxx` |
| Execute an approved plan | `research run --plan P-001 --competition <slug>` |
| Resume an execution | `research resume --execution E-001 --competition <slug>` |
| Run the full autonomous loop | `research conduct run "<goal>" --competition <slug>` |
| Environment check | `research doctor` |
| Lineage / compare / KB / rank | `research experiments …` — [CLI.md](docs/research-pipeline/CLI.md#4-experiments) |
| Hypotheses | `research hypothesize new <slug>` / `list` / `show` / `update` |
| Inspect legacy `runs/` HTML | `research report --run-id <id>` (historical artifacts) |
| Competition HTML | `research experiments dashboard --competition <slug>` |

Global flags (before the subcommand): `--verbose` / `-v`, `--quiet` / `-q`.

Legacy linear Pipeline (`init` / `build` / `improve` / plan-less `run`) has been
**removed**. Tracker:
[docs/research-pipeline/milestones/research-engineer/pipeline-deprecation.md](docs/research-pipeline/milestones/research-engineer/pipeline-deprecation.md).

## Repository layout

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
docs/
  research-pipeline/  # V1 operator docs + shipped milestones
  research-os/        # Research OS design (Conductor roadmap)
tests/                # Unit tests
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `403` / `404` downloading data | Join the competition and accept its rules on Kaggle first |
| Kaggle auth fails | Re-check `KAGGLE_API_TOKEN` in `.env`; `research doctor` reports what it sees |
| `libomp` / LightGBM import error on macOS | `brew install libomp` |
| Analyze or codegen output looks templated | No LLM key or package — that is the `rule_engine` fallback; set `GEMINI_API_KEY` or `OPENAI_API_KEY` |
| Campaign branches fail in bursts | `--branches K` too high for your LLM rate limit; lower `K` |
| Something else | `research doctor`, then [docs/research-pipeline/SOP.md](docs/research-pipeline/SOP.md) |

## Docs

| Doc | Contents |
|-----|----------|
| [docs/README.md](docs/README.md) | Docs router (pipeline vs Research OS) |
| [docs/research-pipeline/SOP.md](docs/research-pipeline/SOP.md) | How to use LabPilot (procedure) |
| [docs/research-pipeline/CLI.md](docs/research-pipeline/CLI.md) | All `research` commands + examples |
| [docs/research-pipeline/ARCHITECTURE.md](docs/research-pipeline/ARCHITECTURE.md) | Modules, execution flow, artifact contracts |
| [docs/research-pipeline/MILESTONES.md](docs/research-pipeline/MILESTONES.md) | V1 roadmap / completed / backlog |
| [docs/research-os/](docs/research-os/) | Research OS north star + execution plan |
| [docs/research-pipeline/milestones/research-engineer/](docs/research-pipeline/milestones/research-engineer/) | Engineer design + deprecation notes |

## Contributing

Issues and pull requests are welcome. Before opening a PR:

```bash
uv sync --extra dev
uv run pytest -m "not llm and not image and not deep"   # tabular slice (required)
uv run ruff check <paths you touched>                   # pass explicit paths
```

Optional CI slices, matching [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```bash
uv sync --extra llm  && uv run pytest -m llm tests/unit/test_llm_client.py
uv sync --extra image && uv run pytest -m image         # optional; may be empty
uv sync --extra deep  && uv run pytest -m deep          # optional; may be empty
```

Conventions live in [AGENTS.md](AGENTS.md): conventional commits (`feat:`, `fix:`,
`docs:`…), tests under `tests/`, no secrets in the tree, and never hand-edit a generated
competition workspace.

## Disclaimer

LabPilot is an independent project and is not affiliated with or endorsed by Kaggle.
You are responsible for following each competition's rules — including limits on external
data, model licences, and automated submissions — and for any API costs incurred by the
LLM providers you configure. Review generated code and metrics before submitting.

## License

MIT — see [LICENSE](LICENSE).
