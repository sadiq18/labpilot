# LabPilot CLI Reference

Complete command reference for the `research` CLI. For a step-by-step workflow
(first run → improve → decide what to try next), see [SOP.md](SOP.md).

Architecture and module contracts live in [ARCHITECTURE.md](ARCHITECTURE.md).

```bash
# Prefer uv when developing from source
uv run research <command> ...

# Or after install:
research <command> ...
```

---

## Contents

1. [Global flags](#1-global-flags)
2. [Pipeline](#2-pipeline) — `run`, `init`, `build`, `resume`, `improve`
3. [Inspect a run](#3-inspect-a-run) — `status`, `report`, `list-runs`, `runs diff`
4. [Experiments](#4-experiments) — graph, show, compare, knowledge, rank, search, report, dashboard
5. [Hypotheses](#5-hypotheses) — add, list, show, update
6. [Environment & project](#6-environment--project) — doctor, workspace, runtime, templates
7. [Common option patterns](#7-common-option-patterns)

---

## 1. Global flags

Place these **before** the subcommand:

| Flag | Description |
|------|-------------|
| `--verbose` / `-v` | Debug logging for every stage |
| `--quiet` / `-q` | Only warnings and errors |

```bash
research -v run --competition titanic
research -q status --run-id 20260715-003000-house-prices-advanced-regression-techniques
```

---

## 2. Pipeline

### `research run`

Full pipeline in one shot: parse → download → profile → brief → baseline → code →
train → evaluate → submission → (optional upload) → log → reflection → HTML report.

```bash
# Local artifacts only (default — does not upload to Kaggle)
research run --competition house-prices-advanced-regression-techniques

# Upload after you have inspected submission.csv
research run --competition titanic --submit

# Attach a hypothesis from the start (marks it testing)
research run --competition titanic --hypothesis H-001

# Validate through codegen without training
research run --competition titanic --dry-run --yes
```

| Option | Description |
|--------|-------------|
| `--competition, -c` | Kaggle competition slug (**required**) |
| `--config` | Config YAML (default: `configs/default.yaml`) |
| `--project-dir` | Project root containing `project.yaml` |
| `--runs-dir` | Override runs directory |
| `--knowledge-dir` | Override knowledge directory |
| `--competitions-dir` | Local per-competition contracts (`<slug>.yaml`) |
| `--hypothesis` | Hypothesis ID to tag on this root run (e.g. `H-001`) |
| `--submit` | Upload validated submission to Kaggle |
| `--force-submit` | With `--submit`: allow upload past deadline |
| `--dry-run` | Stop after code generation; no train/submit |
| `--yes, -y` | Skip prompts (e.g. continue without LLM) |

You must join the competition and accept its rules on Kaggle before data download works.

---

### `research init`

Init half only: parse → download → profile → brief. Review artifacts, then continue
with `build`.

```bash
research init --competition titanic
# Inspect runs/<id>/competition.json, profile.md, brief.md
research build --run-id <id>
```

| Option | Description |
|--------|-------------|
| `--competition, -c` | Required |
| `--config`, `--project-dir`, `--runs-dir`, `--competitions-dir` | Same idea as `run` |
| `--yes, -y` | Skip prompts |
| `--dry-run` | Reserved for symmetry; no effect on init-only |

---

### `research build`

Build half of an already-`init`'d run: baseline → … → reflection (+ report).

```bash
research build --run-id 20260715-003000-house-prices-advanced-regression-techniques
research build --run-id <id> --submit
research build --run-id <id> --dry-run --yes
```

| Option | Description |
|--------|-------------|
| `--run-id, -r` | Required (from `init`) |
| `--config`, `--project-dir`, `--runs-dir` | Standard |
| `--submit` / `--force-submit` / `--dry-run` / `--yes` | Same as `run` |

---

### `research resume`

Re-run from the first failed/incomplete stage. Completed and skipped stages stay put.

**Upload later without re-training:** if the run finished with `upload_submission`
skipped (no `--submit`), then:

```bash
research resume --run-id <id> --submit
```

re-runs only upload (add `--force-submit` past deadline).

```bash
research resume --run-id <id>
research resume --run-id <id> --submit --force-submit
```

| Option | Description |
|--------|-------------|
| `--run-id, -r` | Required |
| `--config`, `--runs-dir`, `--competitions-dir` | Must match the original run where relevant |
| `--submit` / `--force-submit` / `--yes` | Same as `run` |

---

### `research improve`

Fork a **completed** parent: reuse init artifacts, apply a plan, re-run from codegen
through reflection. Writes `comparison.json` / `comparison.md` on the child and updates
the knowledge base / linked hypothesis when applicable.

```bash
# Auto plan (LLM → tune fallback)
research improve --run-id <parent>

# Explicit strategies
research improve --run-id <parent> --strategy tune
research improve --run-id <parent> --strategy features

# Test a hypothesis
research improve --run-id <parent> --hypothesis H-004 --strategy features

# Plan + codegen only
research improve --run-id <parent> --dry-run --yes
```

| Option | Description |
|--------|-------------|
| `--run-id, -r` | Parent run ID (**required**; must be `completed`) |
| `--strategy` | `auto` (default), `tune`, or `features` |
| `--hypothesis` | Tag child with hypothesis ID |
| `--config`, `--project-dir`, `--runs-dir`, `--knowledge-dir` | Standard |
| `--submit` / `--force-submit` / `--dry-run` / `--yes` | Same as `run` |

Child lineage is stored in `manifest.json` (`parent_run_id`, `iteration`) plus
`improvement_plan.json` and `training_overrides.json`.

---

## 3. Inspect a run

### `research status`

```bash
research status --run-id 20260715-003000-house-prices-advanced-regression-techniques
```

### `research report`

Refresh per-run HTML (`runs/<id>/report.html`). Pipeline already writes this at the end;
re-run after generating a competition dashboard if you want the cross-link.

```bash
research report --run-id <id>
```

### `research list-runs`

```bash
research list-runs
research list-runs --runs-dir /path/to/runs
```

### `research runs diff`

Legacy/side-by-side metric + param summary (wrapping comparator under the hood).

```bash
research runs diff \
  --base 20260715-003000-house-prices-advanced-regression-techniques \
  --compare 20260715-003031-house-prices-advanced-regression-techniques
```

Prefer `research experiments compare` when you want categorized changes + verdict.

---

## 4. Experiments

Competition-scoped research memory (Milestone 2). Most commands need `--competition`
(`-c`) and optionally `--runs-dir` / `--knowledge-dir` / `--config`.

### `research experiments graph`

ASCII lineage tree; optional metric annotation and best path marker (`*`).

```bash
research experiments graph --competition house-prices-advanced-regression-techniques
research experiments graph --competition titanic --metric cv_accuracy
research experiments graph -c house-prices-advanced-regression-techniques --metric cv_rmsle
```

### `research experiments show`

```bash
research experiments show 20260715-003000-house-prices-advanced-regression-techniques
research experiments show <run_id> --format json
```

### `research experiments compare`

Categorized A/B + verdict (`worth_keeping` / `not_worth_keeping` / `regression` /
`inconclusive`). Improve already persists this on the child as `comparison.json`/`.md`.

```bash
research experiments compare <base_id> <compare_id>
research experiments compare <base_id> <compare_id> --format markdown
research experiments compare <base_id> <compare_id> --format json
```

### `research experiments knowledge list`

```bash
research experiments knowledge list --competition titanic
research experiments knowledge list -c titanic --effect hurts
research experiments knowledge list -c titanic --technique target_encoding
```

| `--effect` | `improves` \| `hurts` \| `neutral` \| `unknown` |

### `research experiments rank`

Rank **proposed** hypotheses (recommendation backlog — does not execute).

```bash
research experiments rank --competition titanic --top 5
```

### `research experiments search`

Composable **AND** filters. LabPilot config file is `--config-file` here because
`--config` means `key=value` filters.

```bash
research experiments search --competition titanic --metric-gt cv_accuracy:0.8
research experiments search -c titanic --recipe target_encoding --verdict worth_keeping
research experiments search -c titanic --runtime-max 4h --status completed
research experiments search -c titanic --config model_params.learning_rate=0.05
```

| Filter | Example |
|--------|---------|
| `--config key=value` | `--config model_params.ema=true` (repeatable) |
| `--recipe` | `--recipe target_encoding` (repeatable) |
| `--metric-gt` / `--metric-lt` | `--metric-lt cv_rmsle:0.15` |
| `--metric-delta-gt` / `--metric-delta-lt` | Comparison deltas |
| `--runtime-max` / `--runtime-min` | `4h`, `90m`, `30s` |
| `--verdict` | `worth_keeping`, `regression`, … |
| `--status` | Exact status string |
| `--template` | Exact `template_name` |
| `--config-file` | LabPilot YAML (not a filter) |

### `research experiments report`

Competition rollup in the terminal (or JSON for scripts).

```bash
research experiments report --competition house-prices-advanced-regression-techniques
research experiments report -c titanic --format json
```

### `research experiments dashboard`

Static HTML under `knowledge/<slug>/dashboard.html` (gitignored with the rest of
`knowledge/`).

```bash
research experiments dashboard --competition titanic
# open knowledge/titanic/dashboard.html
```

---

## 5. Hypotheses

Stored under `knowledge/<slug>/hypotheses/H-NNN.json` (local / gitignored).

### `research hypothesis add`

```bash
research hypothesis add --competition titanic \
  --observation "Rare classes perform poorly" \
  --reason "Dataset imbalance" \
  --prediction "Focal Loss will improve Macro F1" \
  --confidence 0.74 \
  --tags loss,class-imbalance
```

### `research hypothesis list` / `show` / `update`

```bash
research hypothesis list --competition titanic
research hypothesis list -c titanic --status proposed
research hypothesis show H-001 --competition titanic
research hypothesis update H-001 -c titanic --status confirmed --evidence-run <run_id>
research hypothesis update H-001 -c titanic --status rejected --evidence-run <run_id>
```

Statuses: `proposed`, `testing`, `confirmed`, `rejected`, `inconclusive`.

Attach to execution:

```bash
research run --competition titanic --hypothesis H-001
research improve --run-id <parent> --hypothesis H-001 --strategy tune
```

---

## 6. Environment & project

### `research doctor`

Core checks (Python, LightGBM, Kaggle credentials) plus optional image/deep imports.
Exits non-zero if a **core** check fails.

```bash
research doctor
```

### `research workspace init` / `status`

Multi-competition project (`project.yaml`, `runs/`, `competitions/`, `configs/runtimes/`).

```bash
research workspace init --name kaggle-2026
research workspace status
research run --competition titanic --project-dir .
```

### `research runtime list|show|register|doctor`

Runtime registry (local / kernel / Colab / other). Training still runs locally until
remote dispatch ships.

```bash
research runtime list
research runtime show local
research runtime register --provider kaggle_kernel --id kaggle-gpu-free
research runtime doctor
```

See [configs/runtimes/README.md](../configs/runtimes/README.md).

### `research templates`

```bash
research templates
```

Lists registered baseline templates (tabular / text / image / deep variants).

---

## 7. Common option patterns

| Pattern | Typical flags |
|---------|----------------|
| Point at another runs tree | `--runs-dir /path/to/runs` |
| Point at another knowledge tree | `--knowledge-dir /path/to/knowledge` |
| Workspace project | `--project-dir .` |
| Non-interactive / CI | `--yes` |
| No training | `--dry-run` (on `run` / `build` / `improve`) |
| Upload to Kaggle | `--submit` (optional `--force-submit`) |

**Kernel-only competitions:** LabPilot still trains locally, writes `kernel/`, and with
`--submit` pushes/polls the Kaggle notebook path. Without `--submit`,
`submission_result.json` is `kernel_ready`.

**LLM optional:** without `OPENAI_API_KEY` / `GEMINI_API_KEY` (or the `llm` extra),
brief/reflection fall back to templates. Pipeline commands warn once and ask to
confirm unless `--yes` (or non-TTY / CI).

---

## Quick lookup

| I want to… | Command |
|------------|---------|
| Start a competition | `research run -c <slug>` |
| Pause after brief | `init` → review → `build` |
| Fix a failed run | `research resume -r <id>` |
| Try something better | `research improve -r <parent>` |
| See lineage | `research experiments graph -c <slug>` |
| Decide A vs B | `research experiments compare <a> <b>` |
| Remember what worked | `research experiments knowledge list -c <slug>` |
| What to try next | `research experiments rank -c <slug>` |
| Competition summary | `research experiments report -c <slug>` |
| Shareable HTML overview | `research experiments dashboard -c <slug>` |

Workflow narrative: [SOP.md](SOP.md).
