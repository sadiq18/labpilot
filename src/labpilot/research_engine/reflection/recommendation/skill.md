# RecommendationAgent

Suggest the next experiment (action + rationale + CLI hint).
Offline: prefers open hypotheses, else baseline recheck / critic hint.
Journal assembly stays deterministic in JournalProjector.


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Prefer stacking unused techniques on the winning line.
