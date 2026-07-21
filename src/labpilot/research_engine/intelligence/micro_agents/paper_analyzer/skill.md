# PaperAnalyzerAgent

Structured extraction from a single ML paper. **Not** a summarizer — the LLM
is used as an information extractor that populates a typed artifact.

## Inputs (`StructuredContext`)

- `text` — normalized paper body (deterministically fetched upstream).
- `data` — optional pre-parsed signals used by the `rule_engine` fallback:
  `techniques`, `models`, `datasets`, `limitations`, `hypotheses`, `claims`
  (each a `list[str]`).

## Output schema (`PaperExtract`)

```json
{
  "techniques": ["SpecAugment", "EMA"],
  "models": ["ConvNeXt"],
  "datasets": ["BirdCLEF"],
  "limitations": ["Requires large batch sizes"],
  "hypotheses": ["EMA may improve rare-class recall"],
  "claims": ["EMA improved Macro F1 by 1.2%"]
}
```

## Prompt skeleton

- **System:** "You extract structured research knowledge from an ML paper …
  respond ONLY with the JSON object above; do not summarize."
- **User:** the paper text.

## Fallback (`rule_engine`)

With no LLM configured, echoes the pre-parsed lists from `context.data` so the
pipeline still produces a valid `PaperExtract` (thinner semantic depth).

## Notes

Later plans (6) may replace `PaperExtract` with a richer `PaperKnowledge` and
seed `origin=paper` hypotheses. The agent never calls arXiv/GitHub/Kaggle and
never persists — the caller stores the result.
