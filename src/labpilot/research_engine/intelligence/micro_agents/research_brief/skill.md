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
