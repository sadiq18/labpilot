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

Feeds the Knowledge Merger (§7/§8): merged evidence lands in one `KnowledgeUnit`
row (`techniques` table) with one evidence link per artifact.

How the merger calls this agent (Plan 8):

- Clustering is decided **before** the agent runs — normalized keys, the
  `ALIAS_SEEDS` table, then a conservative containment pass. The agent is only
  asked to pick the canonical label for an already-formed cluster.
- `canonical` must be one of the supplied `items`. Anything else is discarded and
  the deterministic pick (most frequent variant) is used instead, so a
  hallucinated label can never rename a technique.
- `category` is stored on the `KnowledgeUnit` when returned.


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Normalize FE/technique labels; use category feature_engineering for feature-creation concepts.
