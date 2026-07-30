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


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Extract FE recipes + techniques with precise transforms — including arithmetic/derived
features (`new=f1+f2`, `new=f1/f2`, interactions); emit claims usable as hyp evidence.
LLM decides which creations are grounded; do not invent formulas.
