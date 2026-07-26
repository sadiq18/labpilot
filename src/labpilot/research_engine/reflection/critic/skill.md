# RootCauseAgent

Diagnoses a finished experiment run and drafts the structured critic body.
The comparator stays deterministic; this agent only interprets the numbers and
never triggers a re-run. Offline: `rule_engine` sign heuristics.

## Inputs (`StructuredContext`)

- `data` — deterministic run signals: `cv_delta`, `lb_delta`, `changes`, `strength`.
- `text` — optional extra run notes.

## Output schema (`ReflectionDraft`)

```json
{
  "summary": "CV delta +0.0120; LB delta -0.0060",
  "likely_cause": "Validation/test distribution mismatch or CV overfitting.",
  "next_steps": ["Investigate: Mixup", "Investigate: EMA"]
}
```

## Fallback (`rule_engine`)

Builds `summary` from CV/LB deltas, infers `likely_cause` from their signs, and
lists `changes` as follow-up `next_steps`. Fully offline.
