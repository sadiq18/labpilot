# Plan 10 — Hypothesis Assistant

Back to [Milestone 3](README.md). Design: README §10 · §2.4 · Micro Agents.

**Status:** Implemented (progressive ContextBuilder + top-10; recommendations only). **Depends on:** Plan 9. **Unlocks:** Plan 11.

---

## Goal

Connect compressed `ResearchContext` to top-10 experiment recommendations (impact,
confidence, evidence, effort) via `HypothesisAssistant` + `HypothesisGeneratorAgent`.
Write Suggested hypotheses into M2 store. **Recommendations only — no autonomous run.**

## Why this matters

This is the user-visible payoff of analyze: what to try next, grounded in retrieved
knowledge and local failures — not embedding search over papers.

## In scope

- Pipeline-diff improve path: current pipeline → missing techniques → suggest experiments
- Ranking: explicit score formula as SoR (§2.4 / §10.4); Agent may draft text fields only
- `HypothesisGeneratorAgent` + skill.md
- Persist top-10 into analyze report section + M2 HypothesisStore (Suggested)
- Progressive optional: multi-step ContextBuilder rounds if cheap (fixed plan OK)

## Out of scope

- Auto `research improve` / train
- Replacing M2 rank_candidates entirely (reuse/extend)
- Embeddings-based ranking SoR

## Implementation checklist

| Path | Work |
|------|------|
| `intelligence/hypothesize.py` | Assistant |
| `micro_agents/hypothesis_generator/` | Agent |
| Bridge to M2 hypothesis store | Suggested + provenance fields (§12.3) |
| Tests | Fixture ResearchContext → 10 cards; no execute |

## Acceptance criteria

- Given fixture context, emits ≤10 cards with required fields.
- Does not start training or fork runs.
- Works with Agent disabled (template/rule_engine drafts).
- Provenance uses created_by/generator/origin — not single source:llm|analyze.

## Test plan

- Unit: score formula ordering stable.
- Unit: M2 hyp files written Suggested only.
- No network / no auto-run.

## Review notes

- ExperimentReviewerAgent may be stub-wired; full critique polish can follow.
