# RepositoryAnalyzerAgent

Turns a GitHub repository into a structured card. **Not** a summarizer.
Deterministic fetch happens upstream; this agent explains architecture and
reuse effort.

## Inputs (`StructuredContext`)

- `text` — normalized repo content (README, key files) fetched upstream.
- `data` — optional `rule_engine` signals: `architecture` (str),
  `components`/`techniques`/`files_worth_reading` (`list[str]`),
  `integration_difficulty` (`Easy`|`Medium`|`Hard`).

## Output schema (`RepoExtract`)

```json
{
  "architecture": "ConvNeXt Tiny",
  "components": ["SpecAugment", "Mixup", "EMA", "Custom Sampler"],
  "files_worth_reading": ["dataset.py", "loss.py", "augment.py"],
  "techniques": ["SpecAugment", "Mixup", "EMA"],
  "integration_difficulty": "Easy"
}
```

## Prompt skeleton

- **System:** "You analyze an ML GitHub repository … respond ONLY with the JSON
  object above."
- **User:** the repository text.

## Fallback (`rule_engine`)

Reads `context.data`; `components` defaults to `techniques` when absent;
`integration_difficulty` normalizes to `unknown` if unrecognized.

## Notes

Later plans (7) map this to `RepoKnowledge` + `TransferOpportunity.effort`.
