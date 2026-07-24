# CompetitionPageAnalyzerAgent

Structured extraction from Kaggle competition **overview** and **rules** pages
(Plan 5b). The LLM is an information extractor that fills a typed
`CompetitionPageExtract` — never a free-form page summary as system of record.

## Inputs (`StructuredContext`)

- `text` — concatenated overview + rules plain text (fetched/cached upstream).
- `data.use_data_signals` — when true, `rule_engine` echoes pre-parsed fields from
  `data` (tests / upstream injectors).

## Output schema (`CompetitionPageExtract`)

```json
{
  "external_data_allowed": true,
  "pretrained_weights_allowed": true,
  "external_data_notes": "External data is allowed…",
  "runtime_notes": "Kernels limited to 9 hours…",
  "hardware_notes": "GPU P100 available…",
  "internet_allowed": false,
  "inference_notes": "Internet access is disabled…",
  "evaluation_formula": "score = mean F1 …",
  "evaluation_description": "…",
  "submission_format": "csv",
  "submission_columns_notes": "id,label columns required",
  "sample_submission_notes": "",
  "overview_summary": "Detect and track …",
  "other_notes": ""
}
```

Unknown booleans must be `null`; unknown strings empty. Do **not** invent policy.

## Prompt skeleton

- **System:** extract JSON only matching the schema above from overview/rules text.
- **User:** the combined page text.

## Fallback (`rule_engine`)

Heading/keyword heuristics for external-data, internet, evaluation, submission,
and runtime sections. Same schema; thinner semantic depth.

## Notes

Caller (`CompetitionAnalyzer`) maps this into `CompetitionProfile` and persists
via `KnowledgeStore`. Winning solutions / related-comp search are out of scope
(spike-kaggle-discussions).
