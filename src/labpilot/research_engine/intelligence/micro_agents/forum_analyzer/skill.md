# ForumAnalyzerAgent

Mines actionable signals from a Kaggle discussion thread. Used by
``research fetch`` to enrich discussion artifacts (LLM optional).
Plan F may later call the same agent from ``DiscussionAnalyzer``.

## Inputs

- `text` — normalized discussion thread text.
- `data` — optional pre-parsed lists (`mistakes`, `discoveries`,
  `dataset_bugs`, `lb_shakeups`, `ood_notes`). When present, rule_engine
  returns them as-is; otherwise applies keyword heuristics on `text`.

## Output

`ForumExtract` JSON:

- `mistakes`, `discoveries`, `dataset_bugs`, `lb_shakeups`, `ood_notes`

## Behaviour

- **System:** extract actionable signals; JSON only.
- **User:** the discussion text.
- **rule_engine:** pre-parsed lists or light keyword heuristics (never empty
  inventing). Soft-fail path leaves empty lists when nothing matches.

Maps toward `ForumKnowledge` in Plan F. Never scrapes the web itself —
callers supply cached thread text from the official Kaggle topics API.


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Mine FE-like discoveries into techniques/feature_recipes — including arithmetic/derived
features when discussed; note what worked vs failed. LLM decides what is grounded.
