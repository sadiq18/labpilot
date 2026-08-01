# Plan 5 — Git code evolution

Back to [README.md](README.md).

## Goal

Git is the memory of **code** evolution (Research Graph remains reasoning memory).
Simple, code-only commits tied to experiments.

Ship:

1. **Branch per research session/experiment** — e.g. `research/S-001/E-042`
2. **Auto-commit before execution** — message like `experiment: baseline + specaugment`
3. **Store commit hash on Experiment Artifact**

```json
{
  "experiment_id": "E-042",
  "git_commit": "a81f2c",
  "metrics": { "score": 0.91 }
}
```

4. **Rollback CLI** — `research revert E-042` (checks out / restores code to that commit)

Commit **code changes only** — not knowledge DB, artifacts store, or Research Graph.

## Acceptance

- [ ] Creating/running an experiment can create `research/<session>/<experiment>` branch
- [ ] Code changes are committed before/at experiment boundary; hash recorded on artifact
- [ ] `research revert <experiment_id>` restores workspace code to that commit
- [ ] Knowledge / artifact stores are not committed by this path
- [ ] Tests cover commit hash persistence + revert smoke (temp git repo fixture)
