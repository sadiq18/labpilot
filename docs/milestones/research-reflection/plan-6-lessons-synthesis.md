# Plan 6 — Lessons + Knowledge synthesis

Back to [Research Reflection](README.md).

**Status:** Ready. **Depends on:** Plans 4–5. **Unlocks:** Plans 7–8.

---

## Goal

Cross-competition lessons and a “Current Understanding” rollup for journal /
Planner inputs.

## In scope

- `lessons/generator.py` → `lessons` table (LLM + rule_engine)
- `synthesis/synthesizer.py` — rollup of beliefs, recent evidence, open hyps
- Dual-write / migrate writers from `experiments/knowledge.py` as needed
  (path today: `labpilot.experiments`; target: `shared.experiments`)

## Out of scope

- Claim promotion (Plan 7)
- CLI journal formatting polish (Plan 8)

## Acceptance criteria

- [ ] Lesson rows persist after reflect
- [ ] Synthesis output is stable under rule_engine
- [ ] Cross-competition lessons can have null competition_slug
