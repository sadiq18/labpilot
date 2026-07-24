# PaperAnalyzerAgent

Structured extraction from a single ML paper into **`PaperKnowledge`**.
**Not** a summarizer — the LLM fills typed fields only.

## Inputs (`StructuredContext`)

- `text` — abstract / metadata prose (deterministically fetched upstream).
- `competition` — slug so ideas_worth_testing can be competition-grounded.
- `data` — optional pre-parsed signals for `rule_engine`:
  `paper_id`, `title`, `contributions`, `methods`, `limitations`,
  `ideas_worth_testing` / `hypotheses`, `techniques`, `datasets_used` /
  `datasets`, `benchmarks`, `code_urls` / `github_urls`, `claims`,
  `grounded_in`, `confidence`.

## Output schema (`PaperKnowledge`)

```json
{
  "paper_id": "arxiv:1912.09732",
  "title": "SpecAugment",
  "contributions": ["Time/freq masking improves ASR without LM changes"],
  "methods": ["time mask", "freq mask", "time warp"],
  "limitations": ["tuned on speech; may need retuning for sparse bird calls"],
  "ideas_worth_testing": ["narrower freq masks for rare species"],
  "techniques": ["SpecAugment"],
  "datasets_used": ["LibriSpeech"],
  "benchmarks": [],
  "code_urls": [],
  "confidence": 0.7,
  "grounded_in": "abstract"
}
```

## Prompt skeleton

- **System:** extract contributions / methods / limitations / ideas — respond
  ONLY with the JSON object; do not summarize.
- **User:** competition + paper abstract/metadata text.

## Fallback (`rule_engine`)

Echoes pre-parsed lists from `context.data`, or applies thin abstract
heuristics when only free text is present.

## Notes

Never calls literature APIs or persists — the Deterministic Engine
(`PaperAnalyzer` / `LiteratureProvider`) owns fetch + cache.
