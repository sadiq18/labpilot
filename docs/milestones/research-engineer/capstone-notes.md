# Capstone — Autonomous Research Engineer (Plan 11)

Back to [README.md](README.md).

**Status:** Dry-run / offline path verified in CI. Live unattended train+upload
depends on competition data + credentials.

---

## 1. Command sequence

```bash
research analyze <competition>
research plan create <competition> --baseline
research run --plan P-001 --competition <competition>
# optional:
research resume --execution E-001 --competition <competition>
```

Dry-run (no full train / no upload):

```bash
research run --plan P-001 --competition <competition> --dry-run --no-install-packages
```

---

## 2. What “green” looks like

| Check | Where |
|-------|--------|
| Smoke before train | DAG deps + `artifacts/smoke_ok.json` |
| Execution terminal | `research_executions` status `succeeded` |
| Workspace | `competitions/<slug>/` with `pipeline/train.py` |
| Metrics | `competitions/<slug>/metrics.json` (+ DB `experiments` row when evaluate runs) |
| Submission packaged | `competitions/<slug>/artifacts/submission.csv` (upload gated by `--submit`) |
| Report | `knowledge/<slug>/research/reports/E-xxx_report.md` |
| Plan done | all `research_tasks` `done`; plan status `done` |

---

## 3. Automated coverage

- `tests/unit/test_engineer_capabilities.py::test_baseline_dry_run_end_to_end`
- `tests/unit/test_engineer_capabilities.py::test_cli_run_plan_dry`
- Smoke-fail gate: `test_smoke_fail_stops_before_train`

Preferred fixture competition for live runs:
`biohub-cell-tracking-during-development` (or any competition with Analyze artifacts).

---

## 4. Known gaps

- Full Jinja train scripts still expect real data under the workspace; dry-run uses
  syntax-only smoke + stub metrics.
- Remote Runtime providers are dry-run mocked unless explicitly configured later.
- Legacy `research init` / `build` / `improve` removed; WorkspaceCapability
  downloads + profiles so init is unnecessary.
- Live Kaggle upload requires `--submit` and credentials.

---

## 5. Operator note

Happy path after `research run --plan …` starts needs **no mid-flight prompts**.
LLM soft-fail keeps rule_engine scaffolds; upload stays off unless `--submit`.
