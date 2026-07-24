# RepositoryAnalyzerAgent

Extracts transferable ML engineering knowledge from cached GitHub content.
**Not** a README or repository summarizer. Deterministic fetch happens upstream.

## Inputs (`StructuredContext`)

- `text` — normalized repo content (README, key files) fetched upstream.
- `data` — identity, dependencies, key paths, and deterministic signals.

## Output schema (`RepoKnowledge`)

```json
{
  "repo_id": "github:owner/repo",
  "full_name": "owner/repo",
  "architecture": ["ConvNeXt Tiny"],
  "loss": ["Focal Loss"],
  "augmentation": ["SpecAugment", "Mixup"],
  "training_tricks": ["EMA"],
  "interesting_files": ["dataset.py", "loss.py", "augment.py"],
  "dependencies": ["torch", "timm"],
  "techniques": ["SpecAugment", "Mixup", "EMA"],
  "confidence": 0.8,
  "grounded_in": "mixed"
}
```

## Prompt skeleton

- **System:** extract only supported engineering choices into the JSON above.
- **User:** the repository text.

## Fallback (`rule_engine`)

Combines upstream dependency/path signals with deterministic term detection for
architectures, losses, augmentation, and training tricks.

## Notes

The agent never calls GitHub. `RepoDiffer` computes effort/gain separately.
