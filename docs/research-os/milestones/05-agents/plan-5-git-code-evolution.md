# Plan 5 — Git code evolution

Back to [README.md](README.md).

## Goal

Git is the memory of **code** evolution (Research Graph remains reasoning memory).
Simple, code-only commits tied to experiments.

Ship:

1. **`GitTool` abstraction** with **GitPython** backend (CLI `execute` escape hatch)
2. **Branch per research session/experiment** — e.g. `research/S-001/E-042`
3. **Auto-commit before execution** — structured `CommitSnapshot` for agents
4. **Store commit hash on Experiment Artifact**

```json
{
  "experiment_id": "E-042",
  "git_commit": "a81f2c",
  "files_changed": ["pipeline/train.py"],
  "message": "experiment: add specaugment"
}
```

5. **Rollback CLI** — `research revert E-042` via GitTool checkout of code paths

Commit **code changes only** — not knowledge DB, artifacts store, or Research Graph.

## Acceptance

- [x] Creating/running an experiment can create `research/<session>/<experiment>` branch
- [x] Code changes are committed before/at experiment boundary; hash recorded on artifact
- [x] `research revert <experiment_id>` restores workspace code to that commit
- [x] Knowledge / artifact stores are not committed by this path
- [x] Tests cover commit hash persistence + revert smoke (temp git repo fixture)
