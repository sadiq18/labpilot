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


## LabPilot performance rules (all competitions)

- Prefer structured fields over vague prose; accuracy over verbosity.
- Cite artifact ids and technique names whenever recommending a change.
- Distinguish what worked vs failed; never revive deprecated/failed patterns without a recovery angle.
- Prefer improve-on-prior / stack unused techniques over independent restarts when a winning line exists.
- When feature engineering appears, capture concrete recipes (name, inputs, outputs, transform).
- Use beliefs, claims, and lessons when proposing next actions.

### Agent-specific focus

Extract FE recipes and pipeline techniques from repos/kernels — including arithmetic/derived
columns (`f1+f2`, `f1/f2`) when code/text shows them; transferable deltas only.
LLM decides what is grounded.
