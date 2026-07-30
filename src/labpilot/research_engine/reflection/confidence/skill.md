# ConfidenceEstimatorAgent

Qualitative confidence (high/medium/low) for critic/belief updates.
Offline: maps evidence strength + |delta| to label/score.
Note: numeric belief arithmetic stays in BeliefUpdater (deterministic).


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Raise confidence from parent gain + evidence strength for stacked hypotheses.
