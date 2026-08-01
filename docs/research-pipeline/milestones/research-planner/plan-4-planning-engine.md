# Plan 4 — Planning Engine Micro Agent (ONE LLM)

Back to [Research Planner](README.md). Design: [README.md](README.md) §3–4 ·
[package-layout.md](package-layout.md) §4.

**Status:** Not started. **Depends on:** Plan 3. **Unlocks:** richer plans for Plan 5 CLI.

---

## Goal

Add the single Planning Engine Micro Agent (`ResearchPlannerAgent`): optional LLM proposes a
typed plan draft; invalid/unavailable LLM soft-falls back to Plan 3 `rule_engine` templates.
Exactly **one** planner LLM call per compile — no multi-agent loops.

## Why this matters

Judgement (risk, missing prerequisites, overengineering, task selection) is the only place
the LLM belongs. Everything else stays deterministic.

## In scope

```
planner/micro_agents/planning_engine/
  agent.py
  skill.md
planner/prompts/
  planning_engine_system.md
  planning_engine_user.md   # or .j2 — match repo convention
```

- Output: slim `ResearchPlanDraft` (or full plan fields) coerced into `ResearchPlan`
- Merge type-default verification after LLM draft
- Re-validate DAG; on failure → rule_engine template (log note)
- `generated_by`: `llm` | `rule_engine`
- LLM client from `labpilot.llm` (task-routed; soft-fail when unavailable)

## Out of scope

- Helper micro-agents (risk_checker, …) — optional stubs only
- Multi-turn "fix your plan" loops
- CLI (Plan 5)
- Executing any task type

## Design summary

- Same contract as `BaseMicroAgent` / Research Brief: `StructuredContext` in → typed out.
- No memory, no tools, no side effects; caller persists.
- LLM never browses SQLite; only sees assembled context.

## Implementation checklist

| Path | Work |
|------|------|
| `micro_agents/planning_engine/` | Agent + skill.md |
| `prompts/` | System/user prompts |
| `planner.py` | Call agent before/instead of pure template when client set |
| Tests | rule_engine path unchanged; LLM path mocked → valid plan; bad JSON → fallback |

## Acceptance criteria

- With `llm_client=None`, behavior identical to Plan 3.
- With mock LLM returning valid draft JSON, plan `generated_by=llm` and DAG validates.
- With mock LLM returning garbage, soft-fail to template; no crash.
- No second LLM call inside one `compile_research_plan`.

## Test plan

- Unit: agent `rule_engine` path.
- Unit: parse draft → ResearchPlan; missing verification filled from defaults.
- Unit: invalid draft triggers template fallback + note.

## Review notes

- Resist natural-language-only output; parse JSON into schema.
- Do not invent competing planners in this plan.
