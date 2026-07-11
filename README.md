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
- Kaggle submission (uploaded)
- Experiment log
- Reflection report with next-step recommendations

## Quick Start

```bash
# Install (editable dev mode)
pip install -e ".[dev,llm]"

# Configure credentials
cp .env.example .env
# Edit .env with your Kaggle + LLM API keys

# Run a competition
research run --competition titanic

# Check run status
research status --run-id <run_id>
```

## Project Status

**P0 — Research Engine v0.1** (in progress): tabular competitions only.

See [docs/MILESTONES.md](docs/MILESTONES.md) for the full roadmap and [docs/P0_SCOPE.md](docs/P0_SCOPE.md) for what is and isn't in scope.

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
