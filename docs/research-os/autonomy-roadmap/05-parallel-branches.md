# M11 — Parallel research branches

**Status:** implementation shipped 2026-08-11/12 (PRs #122, #126, #127, #132,
#135, #136, #138) — parallel branches, worktree isolation and disk accounting,
compute budget, k-way fan-out · **exit criteria not yet demonstrated:** all three
require a campaign that actually fans out, and none has been run ·
**Design:** [design/05-parallel-branches.md](design/05-parallel-branches.md)

---

## Purpose

Iteration speed. A campaign currently tests one hypothesis at a time, and each
cycle costs a full train. With ~12 hypotheses queued and a target far away,
sequential testing is the difference between hours and days.

M5 already shipped the primitives — `agents/parallel.py`, thin workers with max
concurrency and a shared budget, and the Blinker event bus. They are unused: the
campaign runs strictly sequential.

## Goal

Fan out N hypotheses concurrently, compare, keep the winner, reflect on the
losers.

## Why this is deliberately late

**Parallelism before M7 multiplies nothing.** Every branch would render the same
template and return the same score. Running five identical experiments
concurrently is five times the compute for zero information — and it would look
like progress, which is worse.

This is the same mistake the project already made once: M5 shipped parallel
agents before the sequential loop could run a single real experiment.

## Approach

1. Conductor selects the top K untested hypotheses (K from the concurrency
   budget, not a constant).
2. Each branch gets its own plan, execution and workspace scratch area. Git
   worktrees are the natural isolation boundary and `git_evolution.py` already
   knows about experiment branches.
3. Branches publish `ExperimentCompleted` on the existing bus.
4. A comparison step ranks results, promotes the winner, and files reflections
   for the losers — losing results are evidence and must not be discarded.

## Exit criteria

1. Three hypotheses tested in one campaign step, wall-clock materially less than
   three sequential runs.
2. All three results present in the experiment graph, with the winner promoted.
3. A branch failure does not abort the others.

## Traps

- **Shared SQLite writers.** `knowledge.db`, the hypothesis store and the
  conductor store are all SQLite. Concurrent writers need WAL mode and a
  serialised write path, or campaigns will fail intermittently and
  unreproducibly.
- **Hypothesis dedupe across branches.** Two branches must not claim the same
  hypothesis; `mark_testing_if_proposed` exists and should be the claim
  mechanism.
- **Budget is shared, not per-branch.** The LLM ledger (M10) is global; five
  branches will exhaust a free tier five times faster. Concurrency must be
  derived from the budget, not configured independently of it.

## Related code

- `src/labpilot/research_engine/agents/parallel.py` — thin workers, max concurrency, shared budget
- `src/labpilot/research_engine/agents/events.py`, `subscribers.py` — the bus
- `src/labpilot/research_engine/agents/git_evolution.py` — experiment branch naming
- `docs/research-os/backlog/parallel-research-branches.md`, `git-worktrees-patches.md`
