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
5. [Hypotheses](#5-hypotheses) — list, show, update (+ generate via Intelligence)
6. [Research Intelligence](#6-research-intelligence) — `analyze`, `ingest`, `retrieve`, `hypothesize`, `fetch`
7. [Research Planner](#7-research-planner) — `plan create` / `show` / `list`
8. [Environment & project](#8-environment--project) — doctor, workspace, runtime, templates
9. [Common option patterns](#9-common-option-patterns)

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

Stored under `knowledge/<slug>/hypotheses/H-NNN.json` (local / gitignored) and mirrored
into `knowledge/<slug>/research/knowledge.db`. Every hypothesis command lives under the
single `research hypothesize` verb — hypotheses are generated from evidence, not hand
authored.

### `research hypothesize <slug>`

Generate new hypotheses (see [§6](#6-research-intelligence) for options). Techniques
already tried, or already covered by an open hypothesis, are skipped, so re-running
only adds genuinely new items.

### `research hypothesize list` / `show` / `update`

```bash
research hypothesize list --competition titanic
research hypothesize list -c titanic --status proposed
research hypothesize show H-001 --competition titanic
research hypothesize update H-001 -c titanic --status confirmed --evidence-run <run_id>
research hypothesize update H-001 -c titanic --status rejected --evidence-run <run_id>
```

Statuses: `proposed`, `testing`, `confirmed`, `rejected`, `inconclusive`. A technique
counts as *tried* only once a run attaches its hypothesis (`testing`) or you mark it
explicitly — `proposed` alone never blocks new suggestions.

Attach to execution:

```bash
research run --competition titanic --hypothesis H-001
research improve --run-id <parent> --hypothesis H-001 --strategy tune
```

---

## 6. Research Intelligence

Milestone 3 research partner: synthesize papers / repos / local experiments into a
validated `analyze.json` contract, then retrieve or hypothesize offline. **No HTML**
in v1; terminal is a view over JSON. Never auto-trains.

Canonical artifacts:

```text
knowledge/<slug>/research/reports/analyze.json
knowledge/<slug>/research/reports/research_brief.md
```

### `research analyze`

Understand the problem before experimentation: run analyzers, persist artifacts into
`knowledge.db`, ingest beliefs, generate hypotheses, and write a durable Research Brief.

**Products**

| Product | Where |
|---------|--------|
| Competition artifact | `analyze.json` + `knowledge.db` |
| Dataset artifact | `analyze.json` + `knowledge.db` (default analyzer) |
| Research artifacts (papers, repos, experiments, …) | `analyze.json` + `knowledge.db` |
| Beliefs | `knowledge.db` (Knowledge Hub) |
| Hypotheses | `hypotheses/H-*.json` + `knowledge.db` |
| Research Brief | `reports/research_brief.md` + `analyze.json` field `research_brief` |

Kernels / discussions stay on `research fetch` by default. Pass `--fetch-kaggle` to
pull 5 kernels by votes, 5 by score, and 5 discussions during analyze.

```bash
# All default analyzers (+ brief)
research analyze birdclef-2026

# Single analyzer
research analyze papers birdclef-2026
research analyze repositories birdclef-2026
research analyze experiments birdclef-2026
research analyze competition birdclef-2026

# Subset / skip
research analyze birdclef-2026 --include papers,repositories
research analyze birdclef-2026 --exclude dataset

# Also pull popular Kaggle kernels + discussions (5 / 5 / 5)
research analyze birdclef-2026 --fetch-kaggle

# Stdout format (files are always written when produced)
research analyze birdclef-2026 --format text
research analyze birdclef-2026 --format json

# Defer hub / hypothesis / brief (--skip-ingest also skips hypothesize + brief)
research analyze birdclef-2026 --skip-ingest
research analyze birdclef-2026 --skip-hypothesize
research analyze birdclef-2026 --skip-brief

# Re-fetch cached raw sources
research analyze birdclef-2026 --refresh
```

| Option | Description |
|--------|-------------|
| `--include` | Comma-separated analyzers to run |
| `--exclude` | Comma-separated analyzers to skip |
| `--format` | `text` (default) or `json` — stdout only; always writes `analyze.json` |
| `--refresh` | Re-fetch sources into cache |
| `--fetch-kaggle` | Pull kernels (votes×5 + score×5) and discussions (×5), then ingest over the full store |
| `--skip-ingest` | Defer Knowledge Hub ingestion (also skips hypotheses and Research Brief) |
| `--skip-hypothesize` | Skip generating new hypotheses after ingestion (also skips brief) |
| `--skip-brief` | Skip writing `research_brief.md` |
| `--config`, `--project-dir`, `--runs-dir`, `--knowledge-dir` | Same idea as pipeline |

Technique buckets in the report: **External Recommendations** are Suggested only;
external-only techniques are never labeled Established. Locally Validated fills only
after local promotion (e.g. improve corroboration).

Read `research_brief.md` first for the briefing; `analyze.json` is the full contract.
### `research ingest`

Run the Knowledge Hub over artifacts already in `knowledge.db` (after
`--skip-ingest`, `research fetch`, or a partial analyze), then generate new
hypotheses from the merged knowledge.

```bash
research ingest birdclef-2026
research ingest birdclef-2026 --force
research ingest birdclef-2026 --skip-hypothesize
```

| Option | Description |
|--------|-------------|
| `--force` | Re-ingest all stored artifacts even when receipts are current |
| `--skip-hypothesize` | Merge knowledge only; do not generate new hypotheses |

### `research retrieve`

Multi-stage symbolic retrieval + compressed `ResearchContext` (no network; reads
`knowledge.db` only). Not wired into `analyze` as a separate CLI step — Hypothesis
Assistant uses the same API.

```bash
research retrieve birdclef-2026 -q "Find techniques that improve Macro F1 on Audio"
research retrieve birdclef-2026 -q "Show experiments where Focal Loss hurt" \
  --query-type structured_query --format json
```

| Option | Description |
|--------|-------------|
| `--question` / `-q` | Natural-language or structured question |
| `--query-type` | Intent override (default `hypothesis_generation`) |
| `--pipeline` | CSV of current pipeline techniques (auto-profiled when omitted) |
| `--format` | `text` or `json` |

### `research hypothesize`

Generate **new** hypotheses only (persists Suggested M2 hypotheses +
`research/reports/hypotheses.json`) and report `N new hypothesis generated`. Skips
techniques already tried in local experiments / dispositioned hypotheses, and skips
ideas already covered by an open (`proposed` / `testing` / `confirmed`) hypothesis.

`research analyze` and `research ingest` run this automatically, so the standalone
command is for re-ranking without re-fetching.

```bash
research hypothesize birdclef-2026
research hypothesize new birdclef-2026          # explicit form
research hypothesize birdclef-2026 -q "Suggest five literature-backed experiments" \
  --limit 5 --format json
```

| Option | Description |
|--------|-------------|
| `--question` / `-q` | Ranking question (default: suggest next experiments) |
| `--pipeline` | CSV current pipeline (auto when omitted) |
| `--limit` | 1–10 new hypotheses (default 10) |
| `--format` | `text` or `json` |

Backlog management uses the same verb: `research hypothesize list` / `show` / `update`
(see [§5](#5-hypotheses)).

### `research fetch`

Pull Kaggle **kernels** and/or **competition discussions** into the local research
store via the **official Kaggle API** (no HTML scrape). Designed so a later cron /
worker can call the same library.

```bash
# Both sources — 20 unique NEW artifacts each
research fetch birdclef-2026

# Kernels only (most votes / best scores)
research fetch birdclef-2026 --source kernels --sort votes --limit 10
research fetch birdclef-2026 --source kernels --sort score --limit 10

# Discussions only (UI votes → API top)
research fetch birdclef-2026 --source discussions --limit 15

# Re-pull existing ids
research fetch birdclef-2026 --refresh
```

| Option | Description |
|--------|-------------|
| `--source` | `discussions` \| `kernels` \| `all` (default) |
| `--sort` | `votes` (default) or `score` — applies to kernels; discussions always use vote/top order |
| `--limit` | Unique **new** artifacts to store **per selected source** (pages until met) |
| `--refresh` | Re-fetch and rewrite existing artifacts / raw versions |
| `--config`, `--project-dir`, `--knowledge-dir` | Same idea as analyze |

Storage:

- Kernels → `ResearchArtifact(type=repository, source=kaggle)`, id `kaggle-kernel:{owner}/{slug}`
- Discussions → `ResearchArtifact(type=discussion, source=kaggle)`, id `kaggle-discussion:{slug}:{topic_id}`
- Raw blobs under `research/raw/kernels/` and `research/raw/discussions/`
- Micro Agents enrich when an LLM is configured; otherwise `rule_engine`

Does **not** register `DiscussionAnalyzer` — run `research ingest` later if you want hub beliefs.

---

## 7. Research Planner

Turn a durable hypothesis into an **inspectable, non-executing** task DAG. The planner
never writes code, mutates configs, or starts training — it only emits typed instruction
nodes (`WRITE_CODE`, `RUN_TRAINING`, …) for a human (or a future executor) to act on.

Plans live in `knowledge/<slug>/research/knowledge.db` (`research_plans` /
`research_tasks` / `research_task_deps`) with derived projections under
`knowledge/<slug>/research/plans/<plan_id>.{json,md}`.

### `research plan create`

```bash
research plan create birdclef-2026 --hypothesis H-001
research plan create birdclef-2026 -H H-001 --priority 2 --format json
research plan create birdclef-2026 -H H-001 --format markdown
```

| Flag | Description |
|------|-------------|
| `--hypothesis` / `-H` | Required hypothesis id (`H-001`) |
| `--priority` | Integer priority stored on the plan (default `0`) |
| `--format` | `text` (default), `json`, or `markdown` |
| `--config`, `--project-dir`, `--knowledge-dir` | Same idea as analyze |

Compiles via the planning compiler (template baseline → optional one-shot LLM revision).
With no LLM key, uses `rule_engine` templates. **No `--execute` flag.**

### `research plan show` / `list`

```bash
research plan show birdclef-2026 P-001
research plan show birdclef-2026 P-001 --format json
research plan list birdclef-2026
research plan list birdclef-2026 --status ready
```

Statuses: `draft`, `ready`, `in_progress`, `done`, `abandoned`. Text output prints
topological DAG levels.

After planning, a human still decides whether to `improve` / `run` — the plan does not
auto-execute.

Design: [milestones/research-planner/README.md](milestones/research-planner/README.md).

---

## 8. Environment & project

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

## 9. Common option patterns

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
| Research landscape + briefing | `research analyze <slug>` (read `research_brief.md`) |
| Offline retrieve from knowledge.db | `research retrieve <slug> -q "…"` |
| Rank untried literature-backed ideas | `research hypothesize <slug>` |
| Compile a plan DAG from a hypothesis | `research plan create <slug> -H H-xxx` |
| Inspect / list plans | `research plan show` / `list` |
| Pull Kaggle kernels / discussions | `research fetch <slug>` or `analyze --fetch-kaggle`

Workflow narrative: [SOP.md](SOP.md).
