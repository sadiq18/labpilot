# M1 Foundation — Capstone notes

Back to [README](README.md) · [plan-4-capstone.md](plan-4-capstone.md).

## Same UX, new internals

Operators still run:

```bash
research analyze <competition>
research plan create --baseline
research run --plan P-001 --dry-run
research reflect run --execution E-001
research submit --execution E-001
```

Under the hood (Strangler Phase A), stage CLIs resolve a Research OS
`Workspace` and invoke named tools from `ToolRegistry`:

| CLI | Tool |
|-----|------|
| `research analyze` | `analyze_competition` |
| `research plan create` | `generate_plan` |
| `research run --plan` | `run_plan` |
| `research reflect run` | `reflect` |
| `research submit` | `submit_learn` |

Tools persist primary outputs through artifact adapters. Handlers do **not**
chain into the next stage — callers own orchestration (CLI today; Conductor in M2).

## Dry-run story (offline)

1. Seed / run analyze so `analyze.json` exists.
2. `generate_plan(baseline=True)` → `P-001` + projections.
3. `run_plan(plan_id="P-001", dry_run=True)` → `E-001` succeeded (smoke stubs).

Covered by `tests/unit/test_foundation_capstone.py` and
`tests/unit/test_cli_strangler.py`.

## M2 readiness checklist

- [x] Artifact adapters stable (`labpilot.research_engine.artifacts`)
- [x] Tool registry + stage catalog (`build_default_tool_registry`)
- [x] Workspace facade (`Workspace.from_competition` / `from_cwd` / `from_client`)
- [x] Stage CLIs call tools (no stage→stage execute from CLI handlers)
- [x] Capstone / strangler unit tests green
- [ ] Conductor / task queue — **start on `research-os-m2-conductor`**, not this branch

## Non-goals completed as non-goals

- No goal-string CLI (M3)
- No Conductor behind CLI (M2)
- Resume still uses Engineer directly (no `resume` tool yet)
