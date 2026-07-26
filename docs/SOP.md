# LabPilot SOP — How to Use `research`

Standard operating procedure for running LabPilot on a real Kaggle competition:
setup once, baseline once, then iterate like a research engineer.

Command flags and every subcommand live in [CLI.md](CLI.md). This doc is the
**when / why / in what order**.

---

## 1. Prerequisites (once per machine)

1. **Python 3.11+** and install from source:
   ```bash
   uv sync --extra dev
   # Optional AI briefs/reflections:
   uv sync --extra llm
   ```
2. **Credentials** — copy `.env.example` → `.env`:
   - `KAGGLE_API_TOKEN` (required for download/submit)
   - `OPENAI_API_KEY` *or* `GEMINI_API_KEY` + `LABPILOT_LLM_PROVIDER=gemini` (optional)
3. **Join the competition** on Kaggle and accept the rules (otherwise download fails).
4. **Sanity-check:**
   ```bash
   uv run research doctor
   ```
5. On macOS, LightGBM often needs: `brew install libomp`.

Tabular competitions need only the core install. Image/deep baselines need
`--extra image` / `--extra deep` — see the root [README](../README.md).

---

## 2. Mental model

```
research run / improve     →  writes runs/<run_id>/     (one experiment)
knowledge/                 →  hypotheses + technique memory (per competition)
experiments *              →  read/aggregate across runs (does not train)
research analyze *         →  research partner over papers/repos/local memory
```

- **One slug** = one competition graph (all runs with that `competition` field).
- **`improve`** forks a *completed* parent; it does not re-download or re-profile.
- **Rank / report / dashboard** tell you what to try next; they never auto-train.
- **`analyze` / `retrieve` / `hypothesize`** synthesize evidence; they never auto-train.
- **`plan`** compiles a hypothesis into a DAG; it never executes tasks or starts training.

Artifacts to care about:

| Path | Why open it |
|------|-------------|
| `runs/<id>/metrics.json` | CV score |
| `runs/<id>/reflection.md` | Narrative + next ideas |
| `runs/<id>/report.html` | Full per-run HTML |
| `runs/<id>/comparison.md` | Child vs parent (after improve) |
| `knowledge/<slug>/hypotheses/` | Ideas under test |
| `knowledge/<slug>/knowledge_base.json` | What techniques helped/hurt |
| `knowledge/<slug>/dashboard.html` | Competition overview (generate on demand) |
| `knowledge/<slug>/research/reports/analyze.json` | Research Intelligence contract (M3) |
| `knowledge/<slug>/research/plans/P-*.json|.md` | Research Planner projections (plan-only) |

---

## 3. Day-1 procedure — first baseline

### A. One-shot (usual path)

```bash
uv run research run --competition house-prices-advanced-regression-techniques
# or: --competition titanic
```

Leave **without** `--submit` until you have inspected `submission.csv`.

### B. Two-step (review brief before training)

```bash
uv run research init --competition <slug>
# Read: competition.json, profile.md, brief.md under runs/<id>/
uv run research build --run-id <id>
```

### C. After the run finishes

```bash
uv run research status --run-id <id>
uv run research list-runs
open runs/<id>/report.html          # or refresh: research report --run-id <id>
```

Note the **run id** and primary metric key (e.g. `cv_accuracy`, `cv_rmsle`) from
`metrics.json` — you will use them for graph/search.

If a stage failed or the process died:

```bash
uv run research resume --run-id <id>
```

---

## 4. Day-2+ procedure — iterate

### Step 1 — See the experiment graph

```bash
uv run research experiments graph --competition <slug> --metric <primary_metric>
uv run research experiments show <run_id>
```

### Step 2 — Review hypotheses

`research analyze` / `research ingest` / reflection generate hypotheses under
`knowledge/<slug>/hypotheses/` (also mirrored into `knowledge.db`).

```bash
uv run research hypothesize list --competition <slug>
uv run research hypothesize show H-001 --competition <slug>

# Generate more from current knowledge (new ones only):
uv run research hypothesize <slug>
```

### Step 3 — Rank what to try next

```bash
uv run research experiments rank --competition <slug> --top 5
```

This scores **proposed** ideas only; it does not start training.

### Step 3b — Research partner (Milestone 3)

When you want literature / repo / cross-comp context before picking the next run:

```bash
# Understand the problem — artifacts, beliefs, hypotheses, Research Brief
uv run research analyze <slug>
# Read the briefing first:
#   knowledge/<slug>/research/reports/research_brief.md
# Full contract:
#   knowledge/<slug>/research/reports/analyze.json

# Also pull popular kernels (votes×5 + score×5) and discussions (×5)
uv run research analyze <slug> --fetch-kaggle

# Deeper Kaggle code/forum pull into knowledge/
uv run research fetch <slug> --source all --limit 20

# Ask a grounded question against knowledge.db (offline)
uv run research retrieve <slug> -q "Show experiments where Focal Loss hurt"

# Top-N untried ideas with evidence (also persists Suggested hypotheses)
uv run research hypothesize <slug> --limit 5
```

Treat suggestions as a backlog — pick one, compile a plan, then decide whether to
`improve`. External techniques stay **Suggested** until local runs promote them.
`research fetch` stores kernels as repository artifacts (`source=kaggle`) and
discussions as discussion artifacts; Forum Intelligence extraction still lands in
Plan F analyzers. Analyze defaults leave kernels/discussions to `fetch` unless
you pass `--fetch-kaggle`.

### Step 3c — Compile a research plan (plan-only)

Once you have a hypothesis id, turn it into an inspectable DAG **without** running
training or writing code:

```bash
uv run research plan create <slug> --hypothesis H-00N
uv run research plan show <slug> P-001
uv run research plan list <slug> --status ready
```

Derived projections land under `knowledge/<slug>/research/plans/`. There is no
`--execute` flag — a human still picks `improve` / `run`.

### Step 4 — Improve a completed parent

```bash
# Strategy: auto | tune | features
uv run research improve --run-id <parent> --strategy features --hypothesis H-00N
```

When it completes:

```bash
uv run research experiments compare <parent> <child>
# or read runs/<child>/comparison.md
```

Linked hypothesis status may update automatically (e.g. rejected after a clear
regression). Knowledge base updates from the comparison (and reflection tags).

### Step 5 — Remember and decide

```bash
uv run research experiments knowledge list --competition <slug>
uv run research experiments knowledge list -c <slug> --effect hurts

uv run research experiments search -c <slug> --metric-lt cv_rmsle:0.15
uv run research experiments search -c <slug> --verdict worth_keeping

uv run research experiments report --competition <slug>
uv run research experiments dashboard --competition <slug>
```

Then either rank again and improve, or submit (next section).

---

## 5. When to submit to Kaggle

Default policy: **train and validate locally first**.

```bash
# Inspect first
ls runs/<id>/submission.csv
cat runs/<id>/metrics.json

# Upload this run (without re-training), if upload was skipped earlier:
uv run research resume --run-id <id> --submit

# Or bake upload into a new run / improve:
uv run research run --competition <slug> --submit
uv run research improve --run-id <parent> --submit
```

Use `--force-submit` only if the deadline has passed but Kaggle may still accept
uploads.

Kernel-only competitions: LabPilot still trains locally, exports `kernel/`, and
`--submit` drives the notebook submission path. See [CLI.md](CLI.md).

---

## 6. Decision cheatsheet

| Situation | Do this |
|-----------|---------|
| Fresh competition | `research run -c <slug>` |
| Want to read brief before CPU time | `init` → review → `build` |
| Crash / failed stage | `resume -r <id>` |
| Parent looks good; try a tweak | `improve -r <parent> [--strategy …]` |
| Idea captured, not run yet | `hypothesize <slug>` → `plan create -H …` → `rank` → `improve --hypothesis` |
| “Did child help?” | `experiments compare` or open `comparison.md` |
| “What have we learned about recipe X?” | `experiments knowledge list --technique …` |
| “Show me everything” | `experiments report` + `dashboard` |
| Need landscape + briefing | `research analyze <slug>` → read `research_brief.md`, then `analyze.json` / hypotheses |
| Pull Kaggle code/forum into the store | `research fetch <slug>` (or `analyze --fetch-kaggle`) |
| Grounded Q over the knowledge store | `research retrieve <slug> -q "…"` |
| Literature-backed untried ideas | `research hypothesize <slug>` |
| Hypothesis → executable DAG (no train) | `research plan create <slug> -H H-xxx` |
| Need another operator on the team | Point them at this SOP + [CLI.md](CLI.md) |

---

## 7. Suggested weekly loop

1. `experiments report` / `dashboard` — where are we?
2. Optional: `research analyze` — refresh landscape + Research Brief (+ `--fetch-kaggle` when needed).
3. `experiments rank` or `research hypothesize` — pick one proposed hypothesis (or add one).
4. `research plan create <slug> -H H-xxx` — compile an inspectable DAG (no execution).
5. `improve --hypothesis H-xxx --strategy …` — one change at a time when possible.
6. `experiments compare` — keep / discard.
7. `knowledge list` — update your intuition from effects.
8. Repeat; submit only when local CV (and sanity checks) look worth a leaderboard hit.

Avoid burning daily submission quota on undiagnosed regressions — the comparator
and knowledge base exist so you don’t have to rediscover failures by hand.

---

## 8. Troubleshooting

| Symptom | Check |
|---------|--------|
| Download / API errors | Joined competition? `research doctor`? Valid `KAGGLE_*` in `.env`? |
| LightGBM import fails | macOS: `brew install libomp`; reinstall env |
| Brief/reflection look templated | No LLM key / `llm` extra — expected; set key or pass `--yes` |
| Image/deep train fails | Install `--extra image` or `--extra deep` |
| `improve` refuses parent | Parent must be `completed` (`research status`) |
| Empty graph / report | Wrong `--competition` slug, or `--runs-dir` pointed elsewhere |
| Dashboard / hyps missing later | `knowledge/` is gitignored — local only unless you sync it yourself |

More architecture detail: [ARCHITECTURE.md](ARCHITECTURE.md).  
Experiment Scientist design: [milestones/experiment-scientist/README.md](milestones/experiment-scientist/README.md).  
Research Intelligence design: [milestones/research-intelligence/README.md](milestones/research-intelligence/README.md).  
Research Planner: [milestones/research-planner/README.md](milestones/research-planner/README.md).
