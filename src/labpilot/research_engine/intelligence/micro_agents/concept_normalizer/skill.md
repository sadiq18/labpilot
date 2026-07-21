# ConceptNormalizerAgent

Collapses a set of alias strings into one canonical concept — the hard-for-rules
merge step (§7). Example: `SpecAugment`, `Time Masking`, `Frequency Masking`,
`Random Erasing` → canonical *input/spectrogram augmentation*.

## Inputs (`StructuredContext`)

- `items` — candidate concept/technique strings to unify.
- `data.category` — optional category hint used by the `rule_engine` path.

## Output schema (`ConceptNormalization`)

```json
{
  "canonical": "SpecAugment",
  "aliases": ["Time Masking", "Frequency Masking"],
  "category": "augmentation"
}
```

## Prompt skeleton

- **System:** "You normalize a set of technique/concept strings into one
  canonical concept … respond ONLY with the JSON object above."
- **User:** the newline-joined candidates.

## Fallback (`rule_engine`)

Deduplicates `items` preserving order; first item becomes `canonical`, the rest
become `aliases`. Deterministic but does not do semantic clustering.

## Notes

Feeds the Knowledge Merger (§7/§8): merged evidence lands in one
`KnowledgeClaim` / `Technique` Knowledge Object.
