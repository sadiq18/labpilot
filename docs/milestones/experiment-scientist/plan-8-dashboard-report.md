# Plan 8 — Experiment Dashboard & Report

Back to [Milestone 2](README.md).

**Status:** Shipped. **Depends on:** Plan 1 (`ExperimentGraph`), Plan 3
(`ExperimentComparison`), Plan 5 (`KnowledgeBase`), Plan 6 (`ranking`). **This is the
milestone's capstone / deliverable plan** — everything else composes here.

---

## Goal

```
$ research experiments report --competition birdclef-2026

BirdCLEF-2026
142 Experiments
Best: 0.851

Top Discoveries
  SpecAugment  (+0.010)
  EMA          (+0.006)
  ConvNeXt     (+0.012)

Known Failures
  Large ViT
  CutMix
  Warmup > 10 epochs

Current Best Pipeline
  ConvNeXt → SpecAugment → EMA → Focal Loss → Pseudo Labels

Recommended Next
  Self-Distillation
  Confidence: 78%
```

Task 6 (dashboard) + the milestone's closing deliverable in the brief. This is intentionally
the *last* plan: it has no new reasoning logic of its own, it's a formatter over Plans 1/3/5/6.
That's by design — if this plan needs new reasoning logic to produce a field, that's a sign a
field belongs in an earlier plan instead.

The mockup above is a *summary* — top discoveries, known failures, one best pipeline. It
deliberately does not show every experiment. But "see all experimentation and its progress and
short description" is also a first-class requirement of this plan: both the terminal and HTML
reports additionally render a full **all-experiments table** (every `Experiment` in the graph,
with `status`, `progress`, and `description` from Plan 1), so nothing is hidden behind the
top-3 summaries above. See §1 and §2 below.

## Design

### 1. Terminal report — `research experiments report`

```python
def build_report(competition: str, runs_dir: Path, knowledge_dir: Path) -> ExperimentReport:
    graph = build_graph(runs_dir, competition)                     # Plan 1
    kb = KnowledgeBase.load(knowledge_dir, competition)             # Plan 5
    ranked = rank_candidates(competition, runs_dir, knowledge_dir)  # Plan 6
    metric_key = _primary_metric_key(competition, runs_dir)         # from competition.json

    return ExperimentReport(
        competition=competition,
        experiment_count=len(graph.nodes),
        best_experiment_id=..., best_score=...,        # max/min over graph.nodes by metric_key + direction
        top_discoveries=kb.top_discoveries(3),
        known_failures=kb.known_failures(3),
        best_pipeline=graph.best_path(metric_key),      # Plan 1
        recommended_next=ranked[0] if ranked else None, # Plan 6
        experiments=sorted(graph.nodes.values(), key=lambda e: e.created_at, reverse=True),
    )
```

```python
class ExperimentReport(BaseModel):
    competition: str
    experiment_count: int
    best_experiment_id: str | None
    best_score: float | None
    top_discoveries: list[KnowledgeEntry]
    known_failures: list[KnowledgeEntry]
    best_pipeline: list[Experiment]
    recommended_next: RankedCandidate | None
    experiments: list[Experiment]         # NEW — full list, newest first; powers the all-experiments table (§1/§2)
```

The terminal report (`rich.Table`) renders `experiments` as a table with columns `id`, `status`,
`progress`, `description`, and the primary metric — this is the "see everything" view;
`top_discoveries`/`known_failures`/`best_pipeline` above stay as the curated summary.

Rendered with `rich` (`Console`/`Table`/`Panel`), consistent with the existing CLI's use of
`rich` elsewhere in `cli/main.py`. `--format json` dumps `ExperimentReport.model_dump_json()`
for scripting (and for this plan's own tests).

### 2. HTML dashboard — `research experiments dashboard` (stretch within this plan)

The brief's Task 6 framing ("more useful than TensorBoard") implies something browsable, not
just a terminal print. Reuse the existing pattern from `report/generator.py` (Jinja2 +
`markdown_to_html` helper) rather than inventing a new rendering stack:

- New template `report/templates/experiments_dashboard.html.j2`, sibling to the existing
  `report.html.j2`.
- New `experiments/report.py:render_dashboard_html(report: ExperimentReport, graph: ExperimentGraph) -> str`
  builds the same context shape `ReportGenerator.build_context()` uses today (a dict passed to
  `.render(**context)`), but aggregated across the whole competition instead of one run:
  experiment count, best score, top discoveries/failures tables, an HTML rendering of the
  experiment tree (reuse `ExperimentGraph.to_tree_text()` from Plan 1, wrapped in a `<pre>`
  block for v1 — a real interactive tree widget is a plausible follow-up, not required here).
- **All-experiments table** — the dashboard's main content, not an afterthought: one row per
  `ExperimentReport.experiments` entry (id, status, progress, description, primary metric,
  created_at), each row's id linking to that run's own `runs/<id>/report.html` (Milestone 1's
  per-run report) via a relative or `file://` link so a reader can go from "which experiments
  exist" straight to "full detail on this one" in one click. This is the direct answer to
  "see all experimentation and its progress and short description in HTML" — the
  discoveries/failures/best-pipeline panels above remain a curated summary, not a replacement
  for this table. Rows for experiments with a parent (i.e. `runs/<id>/comparison.md` exists,
  per Plan 3) additionally link to that comparison write-up, so "what changed vs. its parent"
  is one click away alongside the full per-run report.
- Output path: `knowledge/<competition-slug>/dashboard.html` — **not** committed, gitignored
  like `runs/` already is, since it's a generated artifact.

### 3. New/changed files

| File | Change |
|---|---|
| `src/labpilot/experiments/models.py` | + `ExperimentReport` |
| `src/labpilot/experiments/report.py` | new — `build_report()`, terminal rendering, `render_dashboard_html()` |
| `src/labpilot/report/templates/experiments_dashboard.html.j2` | new template |
| `src/labpilot/cli/main.py` | + `experiments report`, `experiments dashboard` |
| `.gitignore` | Unchanged — entire `knowledge/` stays ignored (hyps, KB, and generated `dashboard.html`) |

### 4. CLI

```
research experiments report --competition <slug> [--format text|json]
research experiments dashboard --competition <slug>   # writes + prints path to dashboard.html
```

## Non-goals

- No live-updating dashboard (no websockets/polling/server process) — a static generated HTML
  file, regenerated on demand, matching the existing `report.html` per-run pattern exactly.
  A "watch mode" that regenerates on new runs completing is a plausible follow-up, not required.
- No charting/plots in v1 (no matplotlib/plotly dependency) — tables and the ASCII/HTML tree
  only. Trend visualization (mentioned in the broader "Week 2" notes from the discussion, not
  in the milestone's 8 tasks) is out of scope here; revisit if the text/HTML tables prove
  insufficient in practice.
- Report and dashboard both fail loud (not silently blank) if a competition has zero
  experiments — this is a "you haven't run anything yet" message, not a crash, but it is not
  a partially-populated report either.

## Open questions

1. Should `research report` (the existing **per-run** HTML report command, Milestone 1) gain
   a link to the new per-competition dashboard, or should these stay fully separate commands
   with no cross-linking? → **Resolved:** bidirectional cross-link when the sibling artifact
   exists (per-run report links to `knowledge/<slug>/dashboard.html` if present; dashboard
   rows link to `runs/<id>/report.html` and `comparison.md`).
2. Is `knowledge/` the right gitignore boundary, or should `hypotheses/*.json` and
   `knowledge_base.json` remain committed (they're arguably valuable to keep in version control
   per-competition, unlike `runs/` which is large binary-ish artifacts)? → **Resolved for now:**
   keep **entire `knowledge/` gitignored** (same posture as `runs/`). The design note below
   about committing hyps/KB is deferred — revisit if multi-contributor research memory becomes
   a priority. Generated `dashboard.html` stays under that ignore either way.

## Acceptance criteria

- `research experiments report --competition <slug>` on a fixture with a populated graph,
  knowledge base, and ranking backlog produces output matching every field in the brief's
  deliverable mockup (experiment count, best score, top 3 discoveries, top 3 failures, best
  pipeline path, recommended next + confidence).
- The terminal report's all-experiments table lists every `Experiment` in the fixture graph
  (not just the top 3 discoveries/failures) with correct `status`, `progress`, and
  `description` for each — including at least one still-`running`/`partial` experiment to
  confirm incomplete runs are listed, not filtered out.
- `--format json` output round-trips through `ExperimentReport.model_validate_json(...)`,
  including the full `experiments` list.
- `research experiments dashboard --competition <slug>` writes a valid, openable HTML file
  containing the same data as the text report, including the all-experiments table, and each
  row's link to its per-run `report.html` resolves to a real file on disk in the fixture; rows
  for non-root experiments additionally link to a resolvable `comparison.md`.
- Running the report/dashboard commands against a competition slug with zero runs produces a
  clear "no experiments yet" message, not a traceback.
