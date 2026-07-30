# HypothesisRevisionAgent

Produce why-text and optional revised prediction for hypothesis status updates.
Offline: uses critic hypothesis_outcome + likely_cause.
Status enum mutation stays in HypothesisEvaluator (deterministic).


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Mint improvement hyps with parent_hypothesis_id + technique_stack, not fresh restarts.
