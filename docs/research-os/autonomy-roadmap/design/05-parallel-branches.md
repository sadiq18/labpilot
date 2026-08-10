# Design — M11: parallel research branches

**Plan:** [../05-parallel-branches.md](../05-parallel-branches.md) · **Status:** design ·
**Owner:** unassigned · **Builds on:** M5 thin parallel workers (shipped, unused) ·
**Backlog it resolves:** [../../backlog/parallel-research-branches.md](../../backlog/parallel-research-branches.md)

---

## 1. Background

M5 shipped `run_parallel_async`/`run_parallel_sync` in
[`agents/parallel.py`](../../../../src/labpilot/research_engine/agents/parallel.py) —
max-workers via `anyio.CapacityLimiter`, a shared `ParallelBudget` guarded by an
`anyio.Lock`, per-item failure isolation (each `_run_one` catches its own
exception into a `ParallelResult`, never escaping the task group). It has never
been called from the campaign loop. `conductor/loop.py`'s `run_until_stop` picks
exactly one hypothesis per step (`_next_hypothesis_id`) and dispatches one
`OsTask` through `Scheduler.dispatch` — "does not chain tools," per its own
docstring. M11 is wiring the two together, not building new concurrency
primitives.

## 2. Problem

Three things are missing between "M5's workers exist" and "a campaign step
tests three hypotheses at once safely":

1. **No branch isolation.** `git_evolution.py` knows experiment branch *naming*
   (`research/<session>/<experiment>`) but `GitTool.create_branch` does an
   in-place `repo.create_head` + `branch.checkout()` — a real checkout on the
   single working tree. Two concurrent branches today clobber each other's
   checked-out state. The plan doc's assumption that "worktrees are already the
   isolation boundary" is not true of the current code; there is no `worktree`
   call anywhere under `research_engine/git/`.
2. **No safe concurrent writes.** `SqliteClient` (used by `knowledge.db`, the
   hypothesis-store DB mirror, and `ConductorStore`) opens
   `sqlite3.connect(..., check_same_thread=not allow_cross_thread)` with no
   `PRAGMA journal_mode=WAL` anywhere in `src/`, and its own docstring says the
   client "punts serialisation to callers" — a contract only `BudgetLedger`
   (`fitroute/budget.py`, `threading.RLock`) currently honors. Concurrent
   branches writing experiment results would hit this unlocked.
3. **No claim, comparison, or promotion.** `mark_testing_if_proposed` is a
   read-then-write (`get()` → `update_status()`) with no lock or transaction —
   a TOCTOU race that lets two branches claim the same hypothesis. And once
   branches finish, nothing picks a winner: `ExperimentGraph.best_path()` can
   *identify* the best-scoring path by a metric, but no field marks an
   `Experiment` promoted, and no code sets one.

None of this is exercised today because the loop is sequential, so these gaps
are latent, not yet observed in production incidents.

## 3. Requirements

**Functional**
- Conductor selects the top K untested hypotheses in one step (K bounded by
  the concurrency budget, not a constant).
- Each hypothesis's experiment runs in its own git worktree, so concurrent
  branches don't share a working tree.
- A hypothesis can be claimed by exactly one branch (fix the TOCTOU race in
  `mark_testing_if_proposed`), and a claim that fails to reach a running
  branch (setup failure after claim) is released rather than left stuck.
- Writes to `knowledge.db`, the hypothesis DB mirror, and `ConductorStore`
  during a parallel step are serialized (WAL + single-writer lock, or
  equivalent) so concurrent branches don't corrupt or silently drop rows.
- After all branches finish (or fail, or the process crashes), every worktree
  created for the step is torn down — no orphaned worktrees left on disk.
- After all branches finish, a comparison step ranks results by the
  competition's metric, promotes the winner, and every result (including
  losers) is visible in the experiment graph with a reflection filed for each
  loser.
- The shared LLM budget (M10's `BudgetLedger`) is respected across concurrent
  branches, including the reflection calls losers trigger — no branch or
  reflection should be able to starve the others of the entire step's budget.

**Non-functional** (from the plan's exit criteria)
- Three hypotheses tested in one campaign step, wall-clock materially less
  than three sequential runs.
- A single branch failure does not abort the other branches in the same step
  (already true inside `run_parallel_async`; must hold end-to-end through the
  campaign loop, not just the worker layer).
- Disk usage from K concurrent worktrees is bounded and understood — see §8.

**Blocking pre-requisite:** the budget-scoping decision in §5 must be made
(with M10 sign-off) before implementation starts — §7's design depends on it.

## 4. Scope

**In scope**
- Wiring `run_parallel_async` into the campaign loop for the experiment step.
- Worktree-based branch isolation, including crash-safe teardown (§6, §8).
- Atomic hypothesis claim, with release-on-setup-failure (§8).
- Write serialization for the three SQLite consumers above.
- A comparison/promotion subscriber on the existing Blinker bus.
- Reflection filing for losing branches (reuse existing reflection code,
  triggered per loser).

**Out of scope** (per the backlog doc this plan resolves, and M5's own
docstring — do not redo)
- Max-workers enforcement, shared budget accounting primitive, and
  `asyncio.gather`-style fan-out — M5 already ships these in
  `agents/parallel.py`.
- Distributed / multi-machine orchestration.
- Arbitrary branch-merge policy or conflict resolution beyond "pick one
  winner, keep the losers as evidence."

## 5. Blocking decision — budget scoping

`BudgetLedger` is one file per workspace (`config.budget_path`, no
per-branch/session dimension) and its `RLock` only guards intra-process
concurrency — the code comment notes real safety today depends on
`server._GATEWAY_LOCK` serializing calls at the proxy layer. Three concurrent
branches sharing one ledger means the first branch to call can exhaust the
whole step's budget before the other two get a turn — and reflection calls for
losers (§4) are a second, easy-to-miss consumer of the same ledger, stacking
on top of the 3 experiment runs. This has to be settled before implementation
starts, not discovered during it. Two options:

- Pre-split the step's budget K-ways before fan-out (each `ParallelWorkItem`
  gets `cost` pre-allocated from a per-step sub-budget), reusing M5's existing
  `ParallelBudget` for this rather than touching `BudgetLedger` at all.
  Extend the same pre-split to cover each loser's reflection call.
- Leave `BudgetLedger` as the hard backstop (cooldowns/RPM/TPM) and let
  `ParallelBudget` be the soft per-step allocator — the two are already
  different budgets (LLM calls vs. experiment cost) and don't need to merge.

Leaning toward the second — it needs zero changes to `fitroute` — but this is
**not implementable until M10's owner confirms it**, since `BudgetLedger` is
their component.

**Action:** file this as an explicit question to M10's owner before opening
any implementation PR for this milestone; do not start the `fitroute/budget.py`
row in §7 or the budget parts of §8 until the answer is recorded back into
this section.

## 6. Design

```text
Conductor step
   │
   ├─ select top K untested hypotheses (K ≤ concurrency budget)
   ├─ atomically claim each (fixed mark_testing_if_proposed)
   │     └─ on setup failure below: release the claim, do not leave it stuck
   │
   ├─ per hypothesis: create git worktree on research/<session>/<experiment>
   │
   ▼
run_parallel_async([ParallelWorkItem(id=hyp_id, agent=ExperimentAgent, task=..., cost=...)])
   │  (existing M5 code — max_workers, shared ParallelBudget, per-item isolation)
   ▼
list[ParallelResult]  →  each ok branch publishes ExperimentCompleted (existing bus)
   │
   ▼
new subscriber: branch comparison
   ├─ rank by competition metric (reuse ExperimentGraph.best_path)
   ├─ mark winner promoted
   └─ file reflection for each loser (existing reflection path)
   │
   ▼
teardown worktrees — always, including on crash (§8)
   │
   ▼
startup reconciliation — prune any worktree from a step that never reached
teardown (§8)
```

The campaign loop still decides *whether* to fan out (K > 1) or stay
single-step (K = 1, current behavior) — this is additive, not a replacement
of the sequential path.

## 7. Components & Responsibility

| Component | Change | Notes |
|---|---|---|
| `conductor/loop.py` | New: build K `ParallelWorkItem`s instead of one `OsTask`, call `run_parallel_sync` | Existing single-hypothesis path stays as the K=1 case |
| `agents/parallel.py` | None | Reused as-is; M5 primitives are sufficient |
| `git_evolution.py` / `git/python_backend.py` | New: worktree create/teardown per branch, crash-safe | `create_branch` today mutates the single working tree; needs a worktree-based sibling, not a modification of the existing checkout path |
| *(new)* reconciliation check | New: startup-time `git worktree prune` + orphan sweep | Closes the crash gap — see §8. This repo already has stray worktrees today (`.claude/worktrees/code-review-pr-104-daf104`, a detached-HEAD leftover), so the failure mode is real, not speculative |
| `hypothesis.py::mark_testing_if_proposed` | Fix: atomic claim + release-on-setup-failure | See §8 |
| `accessor/sqlite/client.py`, `conductor/store.py` | New: WAL + serialized writer | See §8 |
| `agents/events.py` / `agents/subscribers.py` | New subscriber: comparison + promotion | Same pattern as the existing evidence-refresh and experience-memory subscribers on `ExperimentCompleted` |
| `shared/experiments/models.py::Experiment` | New: a promoted/winner marker | See §8 |
| `fitroute/budget.py` | Possibly: per-step budget scoping | Blocked on §5 |

## 8. Implementation notes

**Atomic claim, with rollback.** Replace the read-then-write in
`mark_testing_if_proposed` with a single conditional write: a conditional
`UPDATE ... WHERE status = 'proposed'` against the hypothesis DB mirror,
treating `rowcount == 0` as "already claimed." Preferable over a new file-lock
primitive since the mirror already exists. If worktree setup then fails for a
claimed hypothesis (disk full, name collision), release the claim back to
`proposed` before surfacing the failure — a hypothesis must never end a step
stuck in `testing` with no branch behind it.

**Crash-safe worktree teardown.** Teardown runs in a `finally`-equivalent
around the fan-out, so it executes on branch failure as well as success. That
still doesn't cover a hard process crash (killed process, host restart) — for
that, reconciliation is a startup check: `git worktree list`, drop any entry
whose branch is not referenced by a currently-running campaign step (tracked
in `ConductorStore`), and run `git worktree prune`. This mirrors an existing
condition already observed in this repo's own worktree list, so it is treated
as a required path, not a nice-to-have.

**Write serialization.** Turn on `PRAGMA journal_mode=WAL` in `SqliteClient.__init__`
(cheap, and read concurrency during a parallel step matters). WAL alone doesn't
serialize writers, so wrap `ConductorStore` and the hypothesis DB mirror's
write paths in the same kind of `threading.RLock` `BudgetLedger` already uses —
consistent with the existing convention rather than a new one.

**Promotion.** Add a `promoted: bool = False` field to `Experiment`
(`shared/experiments/models.py`) rather than a separate promotion table —
`ExperimentGraph` already rebuilds from `Experiment` records on disk each call,
so a field on the model is visible immediately with no new persistence layer.
The comparison subscriber sets it via the existing experiment-record write path
after ranking finishes. Ties on the metric are broken by earliest
`ExperimentCompleted` timestamp — the first branch to finish wins — so
promotion is deterministic even when scores are numerically equal.

**Disk usage.** K worktrees means K full checkouts of the tracked tree. This is
a required pre-build check, same as §5's budget question, but it does not need
external sign-off — it's answerable by inspecting this repo's own workspace
layout, not a cross-team decision. Confirm whether large inputs (`.cache/`,
`runs/`) are shared across worktrees (they can be, via a symlink into a
common directory outside the git-tracked tree) or would otherwise be
duplicated K times — if the latter, K needs a disk-aware ceiling in addition
to the concurrency-budget ceiling already in §3. Record the answer here before
implementation starts.

**Budget scoping (blocked on §5).** Not implementable until M10's owner
confirms an option. Once decided: if pre-split is chosen, allocate each
`ParallelWorkItem`'s `cost` from a per-step `ParallelBudget` sized to K
branches plus their reflection calls, before fan-out starts; if `BudgetLedger`
stays the sole backstop, no code changes are needed here beyond the branches
sharing it as they already would. Fill in the actual implementation once §5
is resolved — this paragraph is the landing spot for it.

## 9. Tradeoffs

| Decision | Options | Choice | Why |
|---|---|---|---|
| Branch isolation | worktree vs. sequential checkout-lock (mutex around `create_branch`) | worktree | Checkout-lock defeats the purpose — branches would serialize on the working tree, which is what M11 exists to remove |
| Claim mechanism | file lock vs. conditional DB update | conditional DB update | No new lock primitive; mirror already exists; matches "SQLite owns its own serialization" direction of §8 |
| Write serialization | WAL only vs. WAL + explicit writer lock | both | WAL alone permits concurrent writers to still race on read-modify-write app logic; the lock is still needed on top |
| Promotion storage | field on `Experiment` vs. new table | field | `ExperimentGraph` already treats `Experiment` records as the source of truth; a new table would need its own read path |
| Worktree cleanup | teardown-only vs. teardown + startup reconciliation | both | Teardown alone doesn't survive a hard crash; reconciliation is what actually closes the orphan risk already visible in this repo |

## 10. Testing

- **Perf**: 3 concurrent hypotheses in one step, assert wall-clock < 1.5×
  measured single-hypothesis baseline. Exit criterion 1 says "materially less
  than three sequential runs" — a 3× bound would pass on almost no real
  parallelism, so the test needs to be close to single-run time, not just
  under the sequential sum.
- **Isolation**: kill one branch's experiment mid-run (raise inside its
  `Agent.execute`), assert the other two still complete and their results land
  in the experiment graph (exit criterion 3; the worker-level isolation is
  already covered by `test_parallel_workers.py` — this test is about the
  campaign-loop layer, not the worker layer).
- **Mid-run failure teardown**: same kill-mid-run setup as above, additionally
  assert the failed branch's own worktree is torn down — teardown claims to
  cover branch failure, not just setup failure or a full process crash, and
  that path had no dedicated test before this pass.
- **Claim race**: fire `mark_testing_if_proposed` concurrently (threads or
  `anyio` tasks) against the same hypothesis id, assert exactly one caller
  transitions it to `TESTING`.
- **Claim rollback**: force worktree setup to fail after a successful claim,
  assert the hypothesis returns to `proposed` rather than sticking in
  `testing`.
- **Write serialization stress**: N threads/tasks writing to `ConductorStore`
  and the hypothesis DB mirror concurrently under WAL + lock, assert no lost
  writes and no corruption — as important as the claim race test since it's
  the same class of bug across three call sites, not one.
- **Crash reconciliation**: create a worktree, simulate a crash before
  teardown runs, assert the startup reconciliation check removes it.
- **Promotion**: three branches with distinct scores, assert exactly one
  `Experiment.promoted == True` and it's the best-scoring one, and the other
  two have reflections filed.
- **Promotion tie-break**: two branches with equal scores, assert the
  earlier-completed one is promoted deterministically (not random, not both).
