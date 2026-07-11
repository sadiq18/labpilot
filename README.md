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

# Download Titanic data, train, evaluate, and generate submission.csv
uv run research run --competition titanic

# Upload only after inspecting the local submission
uv run research run --competition titanic --submit

# Check run status
uv run research status --run-id <run_id>
```

You must join the competition and accept its rules on Kaggle before downloading data.
On macOS, LightGBM also requires `brew install libomp`.

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
