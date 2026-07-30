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


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Draft stacked improvements with artifact+technique citations; higher confidence when parent gained.
