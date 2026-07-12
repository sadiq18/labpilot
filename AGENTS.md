# AGENTS.md

## Cursor Cloud specific instructions

LabPilot is a single-product Python 3.11+ CLI (`research`, entry point
`labpilot.cli.main:app`) managed with **uv**. There are no long-running servers,
databases, or web frontends — the "app" is a batch pipeline that writes artifacts
to `runs/<id>/` (including a standalone `report.html`). Standard install/test/run
commands live in `README.md` and `.github/workflows/ci.yml`; the notes below only
cover non-obvious caveats.

- **Run commands through uv:** `uv run research <command>` (e.g. `uv run research doctor`,
  `uv run research run --competition <slug>`). The startup update script installs `uv`
  into `~/.local/bin` and runs `uv sync --extra dev`; if `uv` is not on `PATH`, invoke it
  as `~/.local/bin/uv`.
- **Python version:** `uv` builds the venv against the system interpreter (currently
  3.12), which satisfies `requires-python >=3.11`. There is no pinned `.python-version`.
- **Tests:** the required/tabular slice is `uv run pytest -m "not llm and not image and not deep"`.
  Other markers (`llm`, `image`, `deep`) need their extras installed first
  (`uv sync --extra <name>`); CI runs each slice as a separate job.
- **Lint:** `uv run ruff check .` currently reports pre-existing failures (mostly in
  `tests/`). CI does **not** run ruff (only pytest), so those failures are not gating.
  Do not "fix" them as part of unrelated work.
- **Optional extras (`torch`/`transformers`) are NOT installed** by the default
  `--extra dev` sync, so `research doctor` reports Image/deep deps as FAIL — that is
  expected and informational only. Install `--extra image` / `--extra deep` only when
  testing image/deep baselines.
- **Running the pipeline offline (no real Kaggle account):** `research run` fails fast if
  no Kaggle credentials are *present* and otherwise downloads competition data from Kaggle.
  To exercise the full pipeline without network/credentials:
  1. Seed the data cache: put `train.csv`, `test.csv`, and a sample-submission CSV into
     `.cache/kaggle/<slug>/` (the downloader reuses the cache and skips the network).
  2. Add a local contract `configs/competitions/<slug>.yaml` (this dir is gitignored) with
     `title`, `problem_type`, `evaluation_metric`, and `submission_columns` so the problem
     type is known without a metadata fetch.
  3. Set a placeholder `KAGGLE_API_TOKEN=...` so the `doctor` pre-flight passes (no real
     download happens because the cache is used).
  4. Run `uv run research run --competition <slug> --yes` (omit `--submit` so nothing
     uploads). A stray "Authentication required to call the Kaggle API" traceback may print
     during best-effort metadata lookup — it is caught and the run still completes.
- **LLM brief/reflection are optional:** without `OPENAI_API_KEY` / `GEMINI_API_KEY` (and
  the `--extra llm` package), `brief.md` and `reflection.md` fall back to template text.
  Pass `--yes` to skip the interactive confirmation prompt.
