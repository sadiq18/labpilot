# LabPilot CLI Reference

Complete command reference for the `research` CLI. For a step-by-step workflow
(first run → plan → Engineer → decide what to try next), see [SOP.md](SOP.md).

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
2. [Competition workspace](#2-competition-workspace) — `init`, discovery
3. [Execution](#3-execution) — `run`, `resume` (Research Engineer)
4. [Inspect a run](#4-inspect-a-run) — `status`, `report`, `list-runs`, `runs diff`
5. [Experiments](#5-experiments) — graph, show, compare, knowledge, rank, search, report, dashboard
6. [Hypotheses](#6-hypotheses) — list, show, update (+ generate via Intelligence)
7. [Research Intelligence](#7-research-intelligence) — `analyze`, `ingest`, `retrieve`, `hypothesize`, `fetch`
8. [Research Planner](#8-research-planner) — `plan create` / `show` / `list`
9. [Environment](#9-environment) — doctor, runtime, templates
10. [Common option patterns](#10-common-option-patterns)

---

## 1. Global flags

Place these **before** the subcommand:

| Flag | Description |
|------|-------------|
| `--verbose` / `-v` | Debug logging for every stage |
| `--quiet` / `-q` | Only warnings and errors |

```bash
research -v run --plan P-001 --competition titanic
research -q status --run-id 20260715-003000-house-prices-advanced-regression-techniques
```

---

## 2. Competition workspace

Design reference (layout, discovery, path rewiring, deferred installable CLI):
[design/competition-workspace.md](design/competition-workspace.md).

### `research init`

Scaffold a **client-owned** folder for one competition. Does not download data or run analyze.

```bash
# From the LabPilot clone
uv run research init <slug-or-kaggle-url> --path ~/kaggle
uv run research init <slug> --path ~/kaggle --git      # git init + scaffold commit
uv run research init <slug> --path ~/kaggle --no-git   # skip prompt
```

Creates ``~/kaggle/<slug>/`` with:

- `labpilot.yaml` — discovery marker + relative paths
- `knowledge/`, `pipeline/`, `data/`, `artifacts/`, `logs/`, `models/`, `.cache/`
- `configs/default.yaml` — thin overlay
- `.env.example` — copy to `.env` and set `KAGGLE_API_TOKEN` (workspace-local)
- `.gitignore` — `data/`, `.cache/`, `.env`, …
- optional git repo

| Option | Description |
|--------|-------------|
| `--path, -p` | Parent directory (**required**); creates `<path>/<slug>/` |
| `--git` / `--no-git` | Initialize git without prompting (default: prompt) |
| `--force` | Allow scaffolding into a non-empty directory |

**After init**, `cd` into the workspace. Commands walk parents for `labpilot.yaml`
(and shell `PWD` when tools chdir). Prefer `uv run --project` so CWD stays here:

```bash
cd ~/kaggle/<slug>
cp .env.example .env   # set KAGGLE_API_TOKEN — see docs/research-pipeline/SOP.md § Credentials
uv run --project /path/to/labpilot research analyze
uv run --project /path/to/labpilot research plan create --baseline
uv run --project /path/to/labpilot research run --plan P-001
```

Convenience alias:

```bash
alias research='uv run --project /path/to/labpilot research'
```

Without a marker, LabPilot keeps legacy CWD-relative `knowledge/` + `competitions/<slug>/`.

---

## 3. Execution

### `research run`

Plan-driven Research Engineer (SoR). Requires an approved plan:

```bash
research plan create <slug> --baseline
research run --plan P-001 --competition <slug>

# Inside a labpilot.yaml workspace, slug is optional:
research plan create --baseline
research run --plan P-001

# Dry-run: syntax smoke + stub metrics; no upload
research run --plan P-001 --competition <slug> --dry-run --no-install-packages

# Allow Kaggle upload from the submission capability
research run --plan P-001 --competition <slug> --submit
```

| Option | Description |
|--------|-------------|
| `--plan, -p` | Research plan id (**required**, e.g. `P-001`) |
| `--competition, -c` | Competition slug (optional inside a workspace) |
| `--workspace` | Explicit workspace root (directory with `labpilot.yaml`) |
| `--config` | Config YAML (default: `configs/default.yaml` or workspace overlay) |
| `--knowledge-dir` | Override knowledge directory (wins over workspace) |
| `--submit` | Allow Kaggle upload (default: package only) |
| `--dry-run` | Syntax/smoke stub path; no full train/upload |
| `--install-packages / --no-install-packages` | Dependency capability pip install |

Running without `--plan` errors with a migration message. Legacy Pipeline
`build` / `improve` remain **removed** — see
[pipeline-deprecation.md](milestones/research-engineer/pipeline-deprecation.md).
(`research init` now means competition workspace scaffold, not the old Pipeline init.)

A successful run always records local learning: `artifacts/submission_<E-id>.csv`,
`artifacts/execution_outcome.json`, and a `research_artifacts` row
(`type=experiment`, id `exp:execution:<E-id>`). Upload is separate — prefer
[`research submit`](#research-submit).

---

### `research submit`

Upload an execution-scoped submission CSV and patch leaderboard learning onto the
same experiment artifact / hypothesis:

```bash
# After a successful run that wrote artifacts/submission_E-001.csv
research submit --execution E-001
research submit --execution E-001 --dry-run
research submit --execution E-001 --path /custom/file.csv -m "baseline v2"
```

| Option | Description |
|--------|-------------|
| `--execution, -e` | Execution id (**required**) — selects `artifacts/submission_<E-id>.csv` |
| `--competition, -c` | Competition slug (optional inside a workspace) |
| `--path` | Override CSV path (default: execution-scoped filename) |
| `--message, -m` | Kaggle submission message |
| `--dry-run` | Resolve file only; no upload |

On score: sets hypothesis `public_score`, refreshes `actual_outcome`, upserts
belief/technique links, notifies open **proposed** hypotheses of the new
experiment, and promotes strong beliefs to claims. A new hypothesis is minted
**only** when it is an improvement fork with expected gain (e.g. overfit →
regularization) — not a baseline-like follow-up with no benefit.

---

### `research resume`

Resume a Research Engineer execution:

```bash
research resume --execution E-001 --competition <slug>
research resume --execution E-001 --competition <slug> --dry-run
# Inside a workspace:
research resume --execution E-001
```

| Option | Description |
|--------|-------------|
| `--execution, -e` | Execution id (**required**, e.g. `E-001`) |
| `--competition, -c` | Competition slug (optional inside a workspace) |
| `--workspace` | Explicit workspace root |
| `--submit` / `--dry-run` / `--install-packages` | Same idea as `run` |


---


## 4. Inspect a run

These commands read **legacy** `runs/<run_id>/` manifests (pre-Engineer artifacts).
New executions live under `knowledge/<slug>/research/executions/` and the
competition code root (`pipeline/` in a workspace, or legacy `competitions/<slug>/`).

### `research status`

```bash
research status --run-id 20260715-003000-house-prices-advanced-regression-techniques
```

### `research report`

Refresh per-run HTML (`runs/<id>/report.html`) for historical runs.

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

## 5. Experiments

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
`inconclusive`). Useful when comparing two historical `runs/` experiments.

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

## 6. Hypotheses

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
```

`show` prints parent/fork lineage and, when present, the **Evidence Card**
(control vs treatment, expected/observed gains, attribution, decision). See
[design/evidence-card.md](design/evidence-card.md).

```bash
research hypothesize update H-001 -c titanic --status confirmed --evidence-run <run_id>
research hypothesize update H-001 -c titanic --status rejected --evidence-run <run_id>
```

Statuses: `proposed`, `testing`, `confirmed`, `rejected`, `inconclusive`. A technique
counts as *tried* only once a run attaches its hypothesis (`testing`) or you mark it
explicitly — `proposed` alone never blocks new suggestions.

Attach via a plan:

```bash
research plan create <slug> --hypothesis H-001
research run --plan P-002 --competition <slug>
```

---

## 7. Research Intelligence

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
| `--config`, `--runs-dir`, `--knowledge-dir` | Standard path overrides |

Technique buckets in the report: **External Recommendations** are Suggested only;
external-only techniques are never labeled Established. Locally Validated fills only
after local promotion (confirmed runs / knowledge updates).

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
| `--config`, `--knowledge-dir` | Same idea as analyze |

Storage:

- Kernels → `ResearchArtifact(type=repository, source=kaggle)`, id `kaggle-kernel:{owner}/{slug}`
- Discussions → `ResearchArtifact(type=discussion, source=kaggle)`, id `kaggle-discussion:{slug}:{topic_id}`
- Raw blobs under `research/raw/kernels/` and `research/raw/discussions/`
- Micro Agents enrich when an LLM is configured; otherwise `rule_engine`

Does **not** register `DiscussionAnalyzer` — run `research ingest` later if you want hub beliefs.

---

## 8. Research Planner

Turn a durable hypothesis into an **inspectable, non-executing** task DAG. The planner
never writes code, mutates configs, or starts training — it only emits typed instruction
nodes (`WRITE_CODE`, `RUN_TRAINING`, …) for the Research Engineer (`research run --plan`).

Plans live in `knowledge/<slug>/research/knowledge.db` (`research_plans` /
`research_tasks` / `research_task_deps`) with derived projections under
`knowledge/<slug>/research/plans/<plan_id>.{json,md}`.

### `research plan create`

```bash
research plan create birdclef-2026 --hypothesis H-001
research plan create birdclef-2026 -H H-001 --priority 2 --format json
research plan create birdclef-2026 --baseline
research plan create --baseline          # slug from labpilot.yaml
```

| Flag | Description |
|------|-------------|
| competition (arg) | Optional inside a workspace |
| `--hypothesis` / `-H` | Hypothesis id (`H-001`) — mutually exclusive with `--baseline` |
| `--baseline` | Create **P-001** from Analyze context (no hypothesis); must be first plan |
| `--priority` | Integer priority stored on the plan (default `0`) |
| `--format` | `text` (default), `json`, or `markdown` |
| `--config`, `--knowledge-dir`, `--workspace` | Same idea as analyze |

Provide **either** `--baseline` **or** `--hypothesis`. Baseline requires
`reports/analyze.json` and refuses if any plan already exists.

Compiles via the planning compiler (template baseline → optional one-shot LLM revision).
With no LLM key, uses `rule_engine` templates. **No `--execute` flag.**

### `research plan show` / `list`

```bash
research plan show P-001 -c birdclef-2026
research plan show P-001                 # inside a workspace
research plan list birdclef-2026
research plan list                       # inside a workspace
research plan list --status ready
```

Statuses: `draft`, `ready`, `in_progress`, `done`, `abandoned`. Text output prints
topological DAG levels.

After planning, run with the Research Engineer::

```bash
research run --plan P-001 --competition <slug>
```

Design: [milestones/research-planner/README.md](milestones/research-planner/README.md).

---

## 9. Environment

### `research doctor`

Core checks (Python, LightGBM, Kaggle credentials) plus optional image/deep imports.
Exits non-zero if a **core** check fails. Also reports the active competition
workspace (`labpilot.yaml` root + slug) when discovered from CWD.

```bash
research doctor
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

See [configs/runtimes/README.md](../../configs/runtimes/README.md).

### `research templates`

```bash
research templates
```

Lists registered baseline templates (tabular / text / image / deep variants).

---

## 10. Common option patterns

| Pattern | Typical flags |
|---------|----------------|
| Point at another runs tree | `--runs-dir /path/to/runs` |
| Point at another knowledge tree | `--knowledge-dir /path/to/knowledge` |
| Non-interactive / CI | `--yes` |
| No training | `--dry-run` (on `run` / `resume`) |
| Upload to Kaggle | `research submit --execution E-xxx` (preferred); or `--submit` on `run` / `resume` |

**Kernel-only competitions:** CSV packaging is SoR today. Kernel export/push is a
follow-on under Execution Submission/Runtime (helpers live at
`accessor/kaggle/exporter.py`; not wired into the CLI SoR yet).

**LLM optional:** without `OPENAI_API_KEY` / `GEMINI_API_KEY` (or the `llm` extra), and
without a reachable Ollama daemon, analyze / code / narrative Micro Agents fall back to
`rule_engine` templates.

**Local models (Ollama):** set `llm.mode: local` (or `LABPILOT_LLM_MODE=local`), start
Ollama, and pull task models from `configs/default.yaml` (`llm.tasks`), e.g.
`ollama pull qwen2.5-coder:14b` for coding (see `llm.tasks.coding`). Per-task
`force_local: true` always routes that skill to Ollama even in `auto`/`cloud` mode.
Other tasks inherit `llm.provider` (default Gemini). Callers use
`labpilot.llm.LLM.generate(task=...)` — never Ollama HTTP directly.

---

## Quick lookup

| I want to… | Command |
|------------|---------|
| Landscape + briefing | `research analyze <slug>` |
| Baseline then execute | `plan create --baseline` → `run --plan P-001 -c <slug>` |
| Fix a failed execution | `research resume --execution E-001 -c <slug>` |
| Try something better | `research plan create <slug> --hypothesis H-xxx` then `run --plan` |
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
| Compile baseline plan P-001 | `research plan create <slug> --baseline` |
| Inspect / list plans | `research plan show` / `list` |
| Pull Kaggle kernels / discussions | `research fetch <slug>` or `analyze --fetch-kaggle`

Workflow narrative: [SOP.md](SOP.md).
