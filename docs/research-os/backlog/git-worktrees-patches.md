# Backlog — Git worktrees + patch manager

**Status:** Backlog (after thin parallel; pairs with parallel research branches).

## Problem

Thin parallel workers share one worktree. True parallel experiments need isolated
checkouts; agent review benefits from patch objects rather than only commits.

## Proposed later work

### Worktrees

```text
main
 |
 +-- experiment-A (worktree)
 +-- experiment-B (worktree)
 +-- experiment-C (worktree)
```

- One worktree per parallel experiment under `GitTool`
- Merge/compare still Conductor-owned (see [parallel-research-branches](parallel-research-branches.md))

### PatchManager

```text
GitTool + PatchManager
```

- Generate/apply reviewable patches for Implementation specialists
- Optional path before commit for Critic/human approval

## Pickup trigger

Parallel experiment isolation or patch-based review becomes a product need —
not when thin `run_parallel_sync` is enough.
