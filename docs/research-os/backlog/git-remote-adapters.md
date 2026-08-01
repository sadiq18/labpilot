# Backlog — Remote Git adapters (GitHub / GitLab)

**Status:** Backlog (post-M5 GitTool). Local GitPython first.

## Problem

M5 ships a local `GitTool` over GitPython for branch/commit/revert. Cross-machine
collaboration and PR-based review need remotes later.

## M5 already ships

```text
Agent → GitTool → GitPython → local .git
```

## Proposed later work

```text
GitTool
    ├── GitPython (local)
    ├── GitHub adapter
    └── GitLab adapter
```

- Push/pull research branches
- Open PRs for human review of agent commits
- Keep Conductor as authority for which branch/commit is active

## Out of scope here

Wrapping every git feature; use `GitTool.execute(...)` for rare CLI ops.
