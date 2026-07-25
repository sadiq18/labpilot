# ResearchBriefAgent

Draft concise researcher briefing prose (`problem_summary`, `key_risks`,
`recommended_focus`) from already-structured analyze evidence.

- Input: `StructuredContext` with competition slug + compressed evidence text/data
- Output: `ResearchBriefNarrative`
- LLM optional; `rule_engine` templates from structured fields when unavailable
- Used by `research analyze` after ingest + hypothesize
