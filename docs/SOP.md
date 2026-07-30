# LabPilot SOP — How to Use `research`

Standard operating procedure for running LabPilot on a real Kaggle competition:
setup once, then iterate like a research engineer via **Analyze → Plan → Run**.

Command flags and every subcommand live in [CLI.md](CLI.md). This doc is the
**when / why / in what order**.

---

## 1. Prerequisites (once per machine)

1. **Python 3.11+** and install from source:
   ```bash
   uv sync --extra dev
   # Optional LLM-assisted analyze / code / narrative:
   uv sync --extra llm
   ```
2. **Join the competition** on Kaggle and accept the rules (otherwise download fails).
3. On macOS, LightGBM often needs: `brew install libomp`.

Tabular competitions need only the core install. Image/deep baselines need
`--extra image` / `--extra deep` — see the root [README](../README.md).

### Credentials (per competition workspace)

LabPilot reads secrets from the **competition workspace** `.env` only — not from
the LabPilot package/repo `.env`.

```bash
uv run research init <slug> --path ~/kaggle
cd ~/kaggle/<slug>
cp .env.example .env
```

Edit `.env` and set:

| Variable | Required | Notes |
|----------|----------|--------|
| `KAGGLE_API_TOKEN` | Yes (download/submit/fetch) | [Kaggle settings → API](https://www.kaggle.com/settings) → Create New Token |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | Optional | LLM-assisted analyze / codegen |

Then sanity-check **from the workspace**:

```bash
uv run --project /path/to/labpilot research doctor
```

If auth fails, the CLI prints this same setup path. Optional alternatives:
`~/.kaggle/access_token`, or legacy `KAGGLE_USERNAME` + `KAGGLE_KEY` in the
workspace `.env`.

---

## 2. Mental model

**Preferred:** one client-owned folder per competition (created by `research init`).
Full design: [design/competition-workspace.md](design/competition-workspace.md).

```
~/kaggle/<slug>/                 ← cd here; labpilot.yaml is the discovery marker
  knowledge/<slug>/research/…    ← analyze / plans / executions
  pipeline/                      ← generated train.py
  data/                          ← gitignored
  …
```

```
research init <slug> --path ~/kaggle   →  scaffold workspace (no download)
research analyze                       →  knowledge/<slug>/research/ …
research plan create --baseline        →  ResearchPlan DAG (P-xxx)
research run --plan P-001              →  pipeline/ + E-xxx (Research Engineer)
research experiments *                 →  inspect / rank / compare
```

- **One slug** = one workspace folder. Slug and paths come from `labpilot.yaml`
  when your shell CWD is inside that folder (no `--competition` / `--knowledge-dir`
  required).
- **`plan`** compiles intent into typed tasks; it never executes them.
- **`run --plan`** is the system of record for implementation (Workspace → code →
  verify → train → eval → submit → report).
- **`experiments` / `hypothesize`** help you decide what to try next; they never
  auto-train.
- **Legacy:** with no `labpilot.yaml`, paths stay CWD-relative `knowledge/` +
  `competitions/<slug>/`. Writing into the LabPilot clone is legacy — prefer `init`
  for new competitions. The old linear Pipeline (`build` / `improve`) remains removed.

Artifacts to care about (inside a competition workspace):

| Path | Why open it |
|------|-------------|
| `knowledge/<slug>/research/reports/research_brief.md` | Pre-experiment briefing |
| `knowledge/<slug>/research/reports/analyze.json` | Full Analyze contract |
| `knowledge/<slug>/research/plans/P-*.json|.md` | Plan projections |
| `knowledge/<slug>/research/executions/E-xxx/` | Execution + task evidence |
| `profile.json` | Dataset profile from Workspace |
| `pipeline/` | Generated training code |
| `metrics.json` | CV / eval metrics |
| `artifacts/submission.csv` | Packaged submission |
| `knowledge/<slug>/hypotheses/` | Ideas under test |

Deprecation notes:
[milestones/research-engineer/pipeline-deprecation.md](milestones/research-engineer/pipeline-deprecation.md).

---

## 3. Day-1 procedure — first baseline

### A. Happy path (SoR)

```bash
# From the LabPilot clone — scaffold a client-owned folder (once per competition)
uv run research init <slug> --path ~/kaggle --git
cd ~/kaggle/<slug>
cp .env.example .env   # set KAGGLE_API_TOKEN (workspace-local; see § Credentials)

# Optional alias so commands feel local while CWD is the competition folder:
# alias research='uv run --project /path/to/labpilot research'

uv run --project /path/to/labpilot research analyze
# Read: knowledge/<slug>/research/reports/research_brief.md

uv run --project /path/to/labpilot research plan create --baseline
# → P-001

uv run --project /path/to/labpilot research run --plan P-001
# Workspace downloads + profiles data, scaffolds code, verifies, trains (unless --dry-run)
```

Leave **without** `--submit` until you have inspected metrics and `submission.csv`.

Dry-run (syntax/smoke stubs; no full train / no upload):

```bash
uv run --project /path/to/labpilot research run --plan P-001 --dry-run --no-install-packages
```

### B. After the execution finishes

```bash
# Execution status is printed by `run`; evidence lives under:
#   knowledge/<slug>/research/executions/E-xxx/evidence/

ls competitions/<slug>/artifacts/
cat competitions/<slug>/metrics.json
```

If the process died mid-flight:

```bash
uv run research resume --execution E-001 --competition <slug>
```

### C. Optional: inspect historical `runs/`

Older Pipeline artifacts (if any) remain under `runs/<id>/`:

```bash
uv run research list-runs
uv run research status --run-id <id>
uv run research report --run-id <id>
```

---

## 4. Day-2+ procedure — iterate

### Step 1 — Refresh landscape (as needed)

```bash
uv run research analyze <slug>
# Optional Kaggle code/forum pull:
uv run research analyze <slug> --fetch-kaggle
uv run research fetch <slug> --source all --limit 20
```

### Step 2 — Review hypotheses

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

Scores **proposed** ideas only; does not start training.

Grounded questions against the store (offline):

```bash
uv run research retrieve <slug> -q "Show experiments where Focal Loss hurt"
```

### Step 4 — Compile a plan and execute

```bash
uv run research plan create <slug> --hypothesis H-00N
uv run research plan show <slug> P-002
uv run research run --plan P-002 --competition <slug>
```

There is no `research improve`. Iteration is always **plan → run**.

### Step 5 — Compare, remember, decide

```bash
uv run research experiments knowledge list --competition <slug>
uv run research experiments knowledge list -c <slug> --effect hurts

uv run research experiments search -c <slug> --verdict worth_keeping
uv run research experiments report --competition <slug>
uv run research experiments dashboard --competition <slug>
```

Then either rank again and plan another hypothesis, or submit (next section).

---

## 5. When to submit to Kaggle

Default policy: **train and validate locally first**. A successful `research run`
already writes local learning (`execution_outcome.json`, experiment
`research_artifacts` card, hypothesis `actual_outcome`) and packages
`artifacts/submission_<E-id>.csv` — submission is not required for that.

```bash
# Inspect first
ls artifacts/submission_E-001.csv
cat metrics.json
cat artifacts/execution_outcome.json

# Recommended: upload by execution id (records public_score + overfit learning)
uv run research submit --execution E-001

# Or allow upload during a plan run / resume:
uv run research run --plan P-001 --submit
uv run research resume --execution E-001 --submit
```

Kernel-only competitions: CSV packaging is SoR today; kernel export/push remains a
follow-on under Execution Submission/Runtime. See [CLI.md](CLI.md) and Engineer
capstone notes.

---

## 6. Decision cheatsheet

| Situation | Do this |
|-----------|---------|
| Fresh competition | `analyze` → `plan create --baseline` → `run --plan P-001` |
| Crash / interrupted execution | `resume --execution E-xxx -c <slug>` |
| Idea captured, not run yet | `hypothesize` → `plan create -H …` → `run --plan` |
| “What have we learned about recipe X?” | `experiments knowledge list --technique …` |
| “Show me everything” | `experiments report` + `dashboard` |
| Need landscape + briefing | `research analyze <slug>` → read `research_brief.md` |
| Pull Kaggle code/forum into the store | `research fetch <slug>` (or `analyze --fetch-kaggle`) |
| Grounded Q over the knowledge store | `research retrieve <slug> -q "…"` |
| Hypothesis → DAG (no train) | `research plan create <slug> -H H-xxx` |
| Baseline plan | `research plan create <slug> --baseline` → **P-001** |
| Implement approved plan | `research run --plan P-xxx -c <slug>` |
| Upload + LB learning | `research submit --execution E-xxx` |
| Need another operator on the team | Point them at this SOP + [CLI.md](CLI.md) |

---

## 7. Suggested weekly loop

1. `experiments report` / `dashboard` — where are we?
2. Optional: `research analyze` — refresh landscape + Research Brief.
3. First time on a competition: `plan create --baseline` → `run --plan P-001`.
4. `experiments rank` or `research hypothesize` — pick one proposed hypothesis.
   Hypothesize scans the **full experiment ledger** (all artifacts/techniques,
   worked vs failed vs untried, unused beliefs/claims) and prefers **stacked**
   improvements on the winning line (higher confidence) over fresh restarts.
5. `plan create -H H-xxx` — compile an inspectable improve-on-prior DAG
   (technique-inlined tasks; compare vs parent metrics).
6. `run --plan P-xxx` — implement via Research Engineer. `WRITE_CODE` always
   overrides `pipeline/train.py` (backup under `artifacts/code_backups/`),
   keeping what worked and applying the hypothesis technique as a delta.
7. Review metrics / evidence / knowledge effects. Each run also updates
   competition-local skill overlays under `.labpilot/skills/` (summarized when
   long) so agents improve for the next cycle; packaged `skill.md` files are
   the global baseline.
8. Repeat; submit only when local CV looks worth a leaderboard hit.

Avoid burning daily submission quota on undiagnosed regressions — the knowledge
base and experiment tools exist so you don’t rediscover failures by hand.

---

## 8. Troubleshooting

| Symptom | Check |
|---------|--------|
| Download / API errors | Joined competition? Workspace `.env` with `KAGGLE_API_TOKEN`? `research doctor`? |
| LightGBM import fails | macOS: `brew install libomp`; reinstall env |
| Analyze / code look templated | No LLM key / `llm` extra — expected; set key or rely on rule_engine |
| Image/deep train fails | Install `--extra image` or `--extra deep` |
| `run` without `--plan` errors | Required — create a plan first (`--baseline` or `-H`) |
| Empty experiments graph / report | Wrong `--competition` slug, or still looking only at empty `runs/` |
| Dashboard / hyps missing later | `knowledge/` is gitignored — local only unless you sync it yourself |

More architecture: [ARCHITECTURE.md](ARCHITECTURE.md).  
Research Engineer: [milestones/research-engineer/README.md](milestones/research-engineer/README.md).  
Pipeline removal: [milestones/research-engineer/pipeline-deprecation.md](milestones/research-engineer/pipeline-deprecation.md).  
Research Intelligence: [milestones/research-intelligence/README.md](milestones/research-intelligence/README.md).  
Research Planner: [milestones/research-planner/README.md](milestones/research-planner/README.md).  
Experiment Scientist: [milestones/experiment-scientist/README.md](milestones/experiment-scientist/README.md).
