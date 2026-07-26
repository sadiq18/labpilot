# Plan 5 — Code Engineering + Research Review

Back to [Research Engineer](README.md). Design: [capabilities.md](capabilities.md) ·
[architecture.md](architecture.md) §5.

**Status:** Not started. **Depends on:** Plan 4. **Unlocks:** Plan 6 (verify generated code).

---

## Goal

Move/wrap existing **codegen** into **Code Engineering**; add **Research Review** as a
gate/suggest capability. LLM only inside these capabilities (and review), never in the
Engineer controller. Micro-agents remain **stateless** slices under `execution/micro_agents/`
if needed.

## In scope

- `capabilities/code_engineering/` — generate / patch train script, model, dataset adapters
  from TaskContext + Knowledge + plan node
- Migrate or wrap `research_engine/codegen/` (prefer import-wrap first; delete later in Plan 10)
- `capabilities/research_review/` — LLM review of draft vs hypothesis/baseline intent;
  allow / revise-suggestion / block policy (deterministic apply of decision)
- Bounded edits; record file paths + hashes in evidence
- Soft-fail paths when `llm_client=None` (document behavior: skip review? fail task? —
  prefer explicit fail for review-required nodes; codegen may use templates)

## Out of scope

- Unit/smoke verification (Plan 6)
- Training execution (Plan 8)
- Unbounded whole-repo rewrite

## Acceptance criteria

- Code Engineering writes files under competition workspace without Engineer calling LLM
- Review can block progression (task fails → recovery policy applies)
- Diff/evidence recorded for generated files

## Test plan

- Unit: template/codegen path without LLM
- Unit: review block → task failed status
- Integration: workspace → code → review mini DAG
