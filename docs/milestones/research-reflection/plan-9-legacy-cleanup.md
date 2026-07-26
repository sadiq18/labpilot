# Plan 9 — Legacy cleanup

Back to [Research Reflection](README.md). Design: [package-layout.md](package-layout.md).

**Status:** Ready. **Depends on:** Plans 3–5, 8 (migration complete).

---

## Goal

Remove overlaps; leave a single reflection pillar. Retire Pipeline-era HTML
reporting and the quarantined `improvement/` package. Kernel exporter already
lives under `accessor/kaggle/` (top-level `kernel/` removed).

---

## Disposition

| Package | Action |
|---------|--------|
| `src/labpilot/reflection/` | Delete after migrate into `research_engine/reflection` |
| `execution/micro_agents/reflection_generator/` | Delete or thin re-export → reflection/critic |
| **`src/labpilot/report/`** | **Delete** Pipeline-era HTML reporter (see below) |
| **`src/labpilot/improvement/`** | **Delete** after relocating live bits (see below) |
| `src/labpilot/kernel/` | **Done** — `exporter` → `accessor/kaggle/exporter.py`; package removed |
| `src/labpilot/experiments/` (→ planned `shared/experiments/`) | Keep as shared SoR; reflection owns post-run writes; package move is follow-on |
| Reporting stubs | Already replaced in Plan 5 |

---

## Kernel exporter — relocated (done)

| Before | After |
|--------|-------|
| `labpilot.kernel.exporter` | `labpilot.accessor.kaggle.exporter` (`export_kernel`, `build_kernel_metadata`) |

- Still unused by product SoR (CSV submission). Future: wire from Execution
  Submission/Runtime for kernel-mode competitions.
- API accepts `competition_slug: str` or any object with `.slug` (no Intelligence
  import into accessor).
- Unit tests: `tests/unit/test_kernel_exporter.py`

### What goes away

| Piece | Today |
|-------|--------|
| `labpilot.report.generator.ReportGenerator` | Builds `runs/<id>/report.html` from manifest + brief + profile |
| `report/templates/report.html.j2` | Per-run HTML shell |
| CLI `research report` | Invokes `ReportGenerator` |

**Narrative SoR after removal:** `research journal` (markdown/JSON projection).
Do **not** resurrect Pipeline report writing from Engineer Reporting.

### What must relocate first (dashboard)

`experiments/report.py` → `write_dashboard()` loads
`report/templates/experiments_dashboard.html.j2`. Before deleting `report/`:

1. Move `experiments_dashboard.html.j2` next to the dashboard owner, e.g.
   `experiments/templates/experiments_dashboard.html.j2` (or later
   `shared/experiments/templates/`).
2. Point `FileSystemLoader` at the new path.
3. Keep `research experiments dashboard` working (Experiment Scientist surface).

### What stays on disk (read-only)

Existing `runs/*/report.html` files remain readable historical artifacts.
Graph / dashboard may still *detect* `has_report` / link to them — they must not
*generate* new Pipeline reports.

### CLI / docs

- Remove or deprecate `research report` (prefer hard remove if unused in SOP).
- Point operators to `research journal --competition <slug>`.
- Update ARCHITECTURE / CLI docs that still describe `report/generator.py` as
  current SoR.

---

## `improvement/` — split then delete

`research improve` / linear Pipeline are already gone. Plan 9 removes the package.

### Relocate before delete

| Symbol / module | Move to | Why |
|-----------------|---------|-----|
| `DEFAULT_TABULAR_MODEL_PARAMS` | `execution/.../offline_codegen/defaults.py` (name flexible) | Only live production caller is `CodeRenderer` |
| `TrainingOverrides`, `ImprovementPlan`, load/save helpers | `experiments/` (graph-adjacent module) | `assemble_experiment` reads historical `runs/*/training_overrides.json` and `improvement_plan.json` |
| `planner.py`, `fork.py`, `tuner.py`, `recipes.py` | — | Delete; no CLI; drop `tests/unit/test_improvement.py` (or keep only persistence tests under experiments) |

### After relocate

- Delete `src/labpilot/improvement/` entirely.
- No imports of `labpilot.improvement` remain.
- Graph / search tests import overrides helpers from `labpilot.experiments…`.

### Follow-on TODO (not Plan 9 exit criteria)

```text
TODO: When template-based offline codegen is removed, delete
offline_codegen defaults (DEFAULT_TABULAR_MODEL_PARAMS) and any Jinja
tabular param wiring that exists only for that renderer.
```

Track under Engineer / Code Engineering follow-ons. Historical
`TrainingOverrides` / `ImprovementPlan` readers in `experiments/` can stay until
legacy `runs/` support is dropped separately.

---

## Out of scope for Plan 9

- Building a full HTML skin for Journal (optional follow-on).
- Deleting on-disk `runs/*/report.html` or `improvement_plan.json`.
- Moving `experiments/` into `shared/` (separate PR).
- Removing offline template codegen itself (follow-on TODO above).

---

## Acceptance criteria

- [ ] No imports of `labpilot.reflection` remain
- [ ] `labpilot.report` package deleted; no `ReportGenerator` / `research report`
- [ ] Dashboard template relocated; `research experiments dashboard` still green
- [ ] `labpilot.improvement` deleted; codegen imports defaults locally; graph imports run-override DTOs from `experiments`
- [ ] Follow-on TODO recorded (offline codegen defaults cleanup)
- [ ] CI green; no Pipeline reflect/report/improve resurrection
- [ ] Docs point narrative surface at `research journal` / `research_engine.reflection`
