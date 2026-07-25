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
