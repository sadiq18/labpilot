# IntentClassifierAgent

Classify-only Stage 1 agent: free-text research questions → `RetrievalIntent`.

Never invents answers, techniques, or experiments — only fills intent fields so
Symbolic Retrieval can run precisely.

## Inputs (`StructuredContext`)

- `text` / `items` — the user question (or empty when using structured flags).
- `data.profile` — competition profile hints (`task`, `dataset`, `domain`, `metric`, …).
- `data.pipeline` — current local technique list.
- `data.query_type` — optional forced query type.

## Output schema (`RetrievalIntent`)

```json
{
  "task": "Audio Classification",
  "dataset": "BirdCLEF",
  "domain": "bioacoustics",
  "goal": "Improve Macro F1",
  "metric": "macro_f1",
  "query_type": "hypothesis_generation",
  "need_experiments": true,
  "need_papers": true,
  "need_repositories": true,
  "need_forums": false,
  "current_pipeline": ["EMA", "Mixup"],
  "question": "How can I improve BirdCLEF?",
  "classified_by": "llm"
}
```

## Fallback (`rule_engine`)

Delegates to deterministic `classify_intent_rules` (keyword / profile mapping).
Always valid without an API key.


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Support intents for improve-prior, stack technique, and explore unused knowledge.
