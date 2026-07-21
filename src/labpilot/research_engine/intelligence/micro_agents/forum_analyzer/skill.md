# ForumAnalyzerAgent

Mines actionable signals from a Kaggle discussion thread. **Not a Phase 1
default** — Discussion analysis is provider-gated (§2.2). When no provider or
LLM is present, the `rule_engine` path surfaces pre-parsed signals only.

## Inputs (`StructuredContext`)

- `text` — normalized discussion thread text.
- `data` — optional `rule_engine` lists: `mistakes`, `discoveries`,
  `dataset_bugs`, `lb_shakeups`, `ood_notes`.

## Output schema (`ForumExtract`)

```json
{
  "mistakes": ["Leaky validation split"],
  "discoveries": ["Public LB does not reflect hidden test"],
  "dataset_bugs": ["Duplicate audio clips"],
  "lb_shakeups": ["Top-10 reshuffled on private LB"],
  "ood_notes": ["Test contains unseen locations"]
}
```

## Prompt skeleton

- **System:** "You extract actionable signals from a Kaggle discussion thread …
  respond ONLY with the JSON object above."
- **User:** the discussion text.

## Notes

Maps to `ForumKnowledge` in the discussion plan (Plan F). Never scrapes
directly — a `DiscussionProvider` fetches upstream.
