# Plan 3 — Micro Agents scaffold

Back to [Milestone 3](README.md). Design: README §2.4 · §11 package layout.

**Status:** Not started. **Depends on:** Plan 1. **Unlocks:** Plans 6–8, 10; execution reflection agent.

---

## Goal

Add optional Micro Agents under `intelligence/micro_agents/` and
`execution/micro_agents/`: each agent is an `*Agent` class + `skill.md`. Ship Protocol,
`rule_engine` fallback, and stub agents so later plans plug real extract/reason prompts
without inventing layout.

## Why this matters

Selective LLM policy requires a single place for typed extractors. Analyzers stay
fetch/orchestrate; Agents are the reasoning slice. System must work with Agents disabled.

## In scope

- `MicroAgent` Protocol (`name`, `run(context) -> BaseModel`)
- `intelligence/micro_agents/`: `base.py` + stub packages:
  `paper_analyzer`, `repository_analyzer`, `forum_analyzer`, `hypothesis_generator`,
  `concept_normalizer`, `experiment_reviewer` (agent.py + skill.md; rule_engine stubs OK)
- `execution/micro_agents/reflection_generator/` (agent.py + skill.md)
- Wiring: optional flag / config; no LLM required for tests
- skill.md documents inputs, output schema, prompt skeleton

## Out of scope

- Full Paper/Repo extract quality (Plans 6–7)
- Concept merge production quality (Plan 8)
- Hypothesis top-10 product (Plan 10)
- Autonomous agents / ReAct / memory

## Design summary

- LLM never remembers; never searches KB; never calls Kaggle/GitHub/arXiv.
- Same typed schemas whether Agent or `rule_engine` fills them.

## Implementation checklist

| Path | Work |
|------|------|
| `intelligence/micro_agents/**` | Layout + stubs |
| `execution/micro_agents/**` | Reflection stub |
| `common/llm/` | Reuse existing client if present |
| Tests | Run stubs without API key; schema validate |

## Acceptance criteria

- Importing any Micro Agent works with no API key; `rule_engine` path returns valid models.
- Every listed agent directory has `agent.py` + `skill.md`.
- Disabling Agents does not break Plan 1 dry-run analyze.
- No `agents/` autonomous package.

## Test plan

- Unit: each stub `run()` with fixture context → Pydantic OK.
- Unit: skill.md files exist and are non-empty.
- No network / no paid LLM in CI.

## Review notes

- Naming always `*Agent` suffix.
- Extract modules in later plans may thin-wrap these packages.
