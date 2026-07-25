# Plan 5 — CLI: `research plan`

Back to [Research Planner](README.md). Design: [README.md](README.md) §7.

**Status:** Not started. **Depends on:** Plans 3–4. **Unlocks:** Plan 6 capstone.

---

## Goal

Ship a Typer sub-app for plan-only operations:

```bash
research plan create <competition> --hypothesis H-xxx [--priority N] [--format text|json|markdown]
research plan show <competition> <plan-id> [--format text|json|markdown]
research plan list <competition> [--status draft|ready|...]
```

No `--execute`. Thin CLI over `compile_research_plan` + PlanStore.

## Why this matters

This is the user-facing heart of the Research Planner. Same thin-CLI pattern as
`hypothesize` / `analyze`.

## In scope

- Typer `plan_app` under `cli/main.py` (or dedicated module imported by main)
- `create`: load HypothesisStore → compile → upsert → write projections → print DAG summary
- `show` / `list`: read PlanStore; format text (topo levels), json, markdown
- Clear errors: missing hypothesis, missing plan, bad format
- Wire optional LLM client the same way other commands do (env/config); default offline OK

## Out of scope

- Executing tasks / calling `improve` / `run`
- Budget / multi-hypothesis ranking UI
- Revising plans in place (`--revise`) — future

## Design summary

- CLI does not contain planning logic; it calls the compiler.
- Terminal output mirrors analyze/hypothesize style (rich console OK).

## Implementation checklist

| Path | Work |
|------|------|
| `cli/main.py` (or `cli/plan.py`) | Subcommands |
| Docs: `docs/CLI.md`, `docs/SOP.md` | Document commands + weekly loop step |
| Tests | Typer runner: create/show/list on temp knowledge dir |

## Acceptance criteria

- `research plan --help` / `create --help` document flags; no execute flag.
- Create with fixture hypothesis writes DB + `plans/*.json` + `*.md`.
- Show/list work after create; missing ids fail clearly.
- Format `json` prints valid JSON of the plan model.

## Test plan

- CLI: create → show → list on temp workspace (rule_engine only).
- CLI: missing hypothesis exits non-zero with message.
- Assert no `runs/` directory created.

## Review notes

- Match hypothesize_app patterns for config/knowledge_dir options.
