# ExperimentReviewerAgent

Interprets a completed experiment. The comparator (CV/LB deltas) stays
deterministic (§2.4 "No"); this agent only diagnoses the numbers.

## Inputs (`StructuredContext`)

- `data` — deterministic signals: `cv_delta` (float), `lb_delta` (float),
  `changes` (`list[str]`).
- `text` — optional extra notes.

## Output schema (`ExperimentReview`)

```json
{
  "diagnosis": "CV improved but LB regressed: likely validation/test mismatch",
  "suggestions": ["Re-examine effect of: Mixup", "Re-examine effect of: EMA"],
  "confidence": 0.5
}
```

## Prompt skeleton

- **System:** "You diagnose an ML experiment given deterministic CV/LB deltas …
  respond ONLY with the JSON object above."
- **User:** the metric signals plus any notes.

## Fallback (`rule_engine`)

Rule-based diagnosis from the sign of `cv_delta` / `lb_delta` (e.g. CV up + LB
down → distribution mismatch) and echoes `changes` as follow-ups.

## Notes

Distinct from the Execution-side `ReflectionGeneratorAgent`: this reviews an
experiment from the intelligence side (evidence for hypotheses), while
reflection writes `reflection.json` for a run.
