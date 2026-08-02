# Tests

Automated verification for LabPilot lives under this directory so CI and
repository scanners can find it.

## Layout

| Path | Role |
|------|------|
| `tests/unit/` | Fast unit tests (default CI job) |
| `tests/integration/` | Heavier / optional integration (when present) |

Markers used by [`.github/workflows/ci.yml`](../.github/workflows/ci.yml):

| Marker | CI job | Notes |
|--------|--------|-------|
| *(none / default)* | `tabular` | `pytest -m "not llm and not image and not deep"` |
| `llm` | `llm` | Requires `--extra llm` |
| `image` | `image` | `continue-on-error` |
| `deep` | `deep` | `continue-on-error` |

## Running locally

```bash
uv sync --extra dev
uv run pytest -m "not llm and not image and not deep"
uv run pytest tests/unit/test_verify_ai_artifact.py -q
```

Config: [`pyproject.toml`](../pyproject.toml) → `[tool.pytest.ini_options]` with
`testpaths = ["tests"]`.
