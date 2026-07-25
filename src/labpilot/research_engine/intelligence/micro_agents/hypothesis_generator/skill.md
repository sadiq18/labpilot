# HypothesisGeneratorAgent

Drafts ONE testable experiment hypothesis from already-retrieved evidence.
Recommendation only — never executes and never auto-runs (§2.3).

## Inputs (`StructuredContext`)

- `question` — the framing (e.g. "How to improve rare-class recall?").
- `text` — compressed evidence (techniques, findings, failures) from retrieval.
- `data` — optional `rule_engine` fields: `observation`, `prediction`,
  `rationale`, `expected_impact` (float), `confidence` (float).

## Output schema (`HypothesisDraft`)

```json
{
  "observation": "Rare classes have low recall",
  "prediction": "Focal Loss will raise Macro F1 on rare classes",
  "rationale": "Focal Loss down-weights easy majority examples",
  "expected_impact": 0.015,
  "confidence": 0.6
}
```

## Prompt skeleton

- **System:** "You draft ONE testable ML experiment hypothesis … respond ONLY
  with the JSON object above; ground every field in the evidence."
- **User:** the question plus evidence text.

## Notes

Draft shape maps to the M2 Hypothesis store. The caller sets provenance
(`created_by` / `generator` / `origin` / `evidence`) — the agent does not
persist and never executes.
