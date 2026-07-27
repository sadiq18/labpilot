# Plan 3 — ExperimentCritic

Back to [Research Reflection](README.md).

**Status:** Done. **Depends on:** Plan 2. **Unlocks:** Plan 4–5.

---

## Goal

Assess an experiment: root-cause sketch, confidence, recommendation — with LLM
and `rule_engine` fallback. Promote/wire `ReflectionGeneratorAgent`.

## In scope

- `reflection/critic/` (facade + micro_agent + prompts)
- Migrate useful bits from top-level `labpilot.reflection` / StructuredReflection
- Input: `ExperimentEvidence` (+ plan/hypothesis text)
- Output: structured assessment consumed by BeliefUpdater / HypothesisEvaluator

## Out of scope

- Writing beliefs/hypotheses (Plan 4)
- Lessons / claims

## Acceptance criteria

- [x] rule_engine path works in CI without API keys
- [x] LLM path optional behind existing provider config
- [x] Reporting can import critic without circular imports
