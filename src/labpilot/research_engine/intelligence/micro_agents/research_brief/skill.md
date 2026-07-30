# ResearchBriefAgent

Draft concise researcher briefing prose (`problem_summary`, `key_risks`,
`recommended_focus`) from already-structured analyze evidence.

- Input: `StructuredContext` with competition slug + compressed evidence text/data
- Output: `ResearchBriefNarrative`
- LLM optional; `rule_engine` templates from structured fields when unavailable
- Used by `research analyze` after ingest + hypothesize

## Prose rules

1. Write complete sentences that end with `.` `!` or `?` — never with `...` or `…`.
2. Do not truncate mid-thought with an ellipsis. Prefer a shorter full sentence
   over a cut-off longer one.
3. `problem_summary` must stand alone as finished prose (one or a few complete
   sentences), not a teaser that trails off.


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Surface worked/failed/untried techniques; unused beliefs/claims; winning stack.
