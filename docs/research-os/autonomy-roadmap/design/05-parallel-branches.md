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

Four things are missing between "M5's workers exist" and "a campaign step
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
4. **No compute isolation.** Nothing in the training path sets thread or core
   limits — a repo-wide search for `n_jobs`/`num_threads`/`nthread` in
   `src/labpilot` returns zero hits. Generated training code gets library
   defaults, which for LightGBM/XGBoost/scikit-learn is typically "use every
   core." K concurrent branches under those defaults oversubscribe the same
   physical cores instead of getting K× the throughput — the parallelism this
   milestone builds toward can end up *slower* than sequential, not faster.

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
- Each branch's generated training code respects a per-branch compute budget
  (thread/core cap = available cores ÷ K) instead of the library defaults
  that assume sole use of the machine — nothing sets this today (§2 item 4).

**Non-functional**
- Three hypotheses tested in one campaign step, wall-clock materially less
  than three sequential runs (plan exit criterion 1).
- A single branch failure does not abort the other branches in the same step
  (plan exit criterion 3; already true inside `run_parallel_async`; must hold
  end-to-end through the campaign loop, not just the worker layer).
- Disk usage from K concurrent worktrees is bounded and understood — see §8
  (this design's own addition, not one of the plan's three exit criteria).
- Each branch's execution is still auditable — whether it produces its own
  `DecisionRecord`, updates checkpoints, and counts toward the consecutive-
  failure circuit breaker the same way sequential dispatch does today is not
  yet designed (see §8).

**Blocking pre-requisites, neither resolved yet:**
1. The budget-scoping decision in §5, needing M10 sign-off.
2. How `DecisionRecord`, checkpointing, and the circuit breaker apply to
   K-way fan-out (§8) — `run_parallel_async` bypasses `Scheduler.dispatch`
   entirely today, and this design doesn't yet say whether that's acceptable.

## 4. Scope

**In scope**
- Wiring `run_parallel_async` into the campaign loop for the experiment step.
- Worktree-based branch isolation, including crash-safe teardown (§6, §8).
- Atomic hypothesis claim, with release-on-setup-failure (§8).
- Write serialization for the three SQLite consumers above.
- A comparison/promotion subscriber on the existing Blinker bus.
- Reflection filing for losing branches (reuse existing reflection code,
  triggered per loser).
- Per-branch compute (CPU thread/core) budgeting for generated training code,
  so K-way fan-out doesn't oversubscribe the same cores (§8).

**Out of scope** (per the backlog doc this plan resolves, and M5's own
docstring — do not redo)
- Max-workers enforcement, shared budget accounting primitive, and
  `asyncio.gather`-style fan-out — M5 already ships these in
  `agents/parallel.py`.
- Distributed / multi-machine orchestration, including remote execution
  backends (Kaggle, Colab, cloud). The Runtime abstraction for this already
  shipped
  ([research-engineer/plan-7-runtime.md](../../../research-pipeline/milestones/research-engineer/plan-7-runtime.md),
  `execution/runtimes/`, `execution/capabilities/runtime/`); actual dispatch/
  poll/artifact-sync execution against remote backends is a separately
  tracked, already-deferred item
  ([TODO.md](../../../research-pipeline/milestones/TODO.md), "P2 remote
  execution"), not part of this milestone. `ParallelWorkItem` still gets a
  `runtime` field defaulting to `"local"` (§7) so that work can build on this
  fan-out later without retrofitting it.
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
   ├─ per hypothesis: allocate compute budget (cores ÷ K), injected into
   │     the generated training code — not left to library defaults (§8)
   │
   ▼
run_parallel_async([ParallelWorkItem(id=hyp_id, agent=ExperimentAgent, task=..., cost=...)])
   │  (existing M5 code — max_workers, shared ParallelBudget, per-item isolation)
   ▼
list[ParallelResult]  →  each ok branch publishes ExperimentCompleted (existing bus)
   │
   ▼
new subscriber: branch comparison
   ├─ rank by competition metric (`_pick_best`, not `best_path` — see §8)
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
| `conductor/loop.py` | New: build K `ParallelWorkItem`s instead of one `OsTask`, call `run_parallel_sync` | Existing single-hypothesis path stays as the K=1 case; K>1 bypasses `Scheduler.dispatch`/`DecisionRecord`/checkpointing/breaker accounting — open question, see §8 |
| `agents/parallel.py::ParallelWorkItem` | Minor: add `runtime: str = "local"` field (unread this milestone, forward-compat only) | Otherwise reused as-is; M5's concurrency primitives are sufficient |
| `agents/coding.py` (M19 delta-codegen path) | New: inject a per-branch thread cap into generated training code | See §8. Exact mechanism (prompt instruction vs. env-var wrapper) not yet decided |
| `git_evolution.py` / `git/python_backend.py` | New: worktree create/teardown per branch, crash-safe | `create_branch` today mutates the single working tree; needs a worktree-based sibling, not a modification of the existing checkout path |
| *(new)* reconciliation check | New: startup-time `git worktree prune` + orphan sweep | Closes the crash gap — see §8. A worktree whose creating process dies before teardown runs is a standard git-worktree failure mode, not something this design can assume away |
| `hypothesis.py::mark_testing_if_proposed` | Fix: atomic claim on the JSON file (not the DB mirror) + release-on-setup-failure | See §8 |
| `accessor/sqlite/client.py`, `conductor/store.py` | New: WAL + module-level serialized writer | See §8 |
| `agents/events.py` / `agents/subscribers.py` | New subscriber: comparison + promotion | Same pattern as the existing evidence-refresh and experience-memory subscribers on `ExperimentCompleted` |
| `agents/experiment.py::event_payload` | New: `completed_at` field | Needed for deterministic tie-break, see §8 |
| `shared/experiments/graph.py::assemble_experiment` + manifest write path | New: read/write a `promoted` flag via `manifest.metadata`, not a new `Experiment` field | See §8 |
| `fitroute/budget.py` | Possibly: per-step budget scoping | Blocked on §5 |

## 8. Implementation notes

**Auditability & breaker accounting — open, not designed.**
`run_parallel_async` calls `item.agent.execute(...)` directly; it does not go
through `Scheduler.dispatch`, `store.enqueue`/`OsTask`, `DecisionRecord`, or
`_record_experiment_outcome`'s consecutive-failure circuit breaker
(`conductor/loop.py`) — all of which wrap every other tool dispatch today.
That file's own comments describe past silent-degradation bugs this machinery
exists to catch. Before implementation starts: decide whether each of the K
branches produces its own `DecisionRecord` and checkpoint, and whether a
branch failure counts toward the breaker the same way a sequential failure
does. Skipping this silently reopens exactly the class of bug
`_record_experiment_outcome`'s comments describe.

**Atomic claim, with rollback.** The hypothesis DB mirror is not a valid claim
target — it's a best-effort dual-write cache (`_mirror_many_to_db`'s own
comment calls it a backfill for file-only hypotheses); `get()`, `list()`, and
`rank_hypotheses()` all read the **JSON file**, which is where the actual
TOCTOU race from §2 lives. A conditional `UPDATE` against the mirror would
leave that race exactly as it is today. The claim has to lock the JSON file
itself — a per-hypothesis file lock (`fcntl`/`filelock`) held across the
existing `get()` → `update_status()` sequence, so two callers can't both
observe `proposed` before either writes. If worktree setup then fails for a
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
serialize writers, and an instance-level lock like `BudgetLedger`'s
`threading.RLock` only works there because one `BudgetLedger` is constructed
once and shared for the process's lifetime. `ConductorStore` and
`KnowledgeStore` are not used that way — both are freshly constructed at each
call site (6 and 37 call sites respectively), so a `self._lock` added the same
way would be a new, uncontended lock every call and serialize nothing. The
lock has to be **module-level** — one lock object shared by every
`ConductorStore`/`KnowledgeStore` instantiation in the process, not an
instance attribute.

**Promotion.** Rank the step's K results with the module-level
`_pick_best(candidates, metric_key, maximize)` (`shared/experiments/graph.py`),
not `ExperimentGraph.best_path()` — `best_path()` walks a single lineage
starting from `self.roots`, descending only through the child it already
picked as best, so it never compares K siblings hanging off one shared parent
that may not sit on the overall best-scoring path. `_pick_best` is the actual
all-candidates comparison and is already what `best_path()` calls internally.

`Experiment` is explicitly a read-side aggregate, not a persisted record — its
own docstring: "Not a new file written to `runs/<id>/`... there is exactly one
writer per field elsewhere in the pipeline." A `promoted` field on the
pydantic model has nowhere to round-trip to. Write it into `manifest.metadata`
instead (`save_manifest` already has a writer for that file) and add a
corresponding read in `assemble_experiment()` so it survives the next
`build_graph()` call.

**Tie-break.** `ExperimentCompleted`'s payload (`agents/experiment.py`) has no
timestamp field today — add a `completed_at` field to `event_payload` at
publish time. Ties on the metric are then broken by earliest `completed_at`,
the first branch to finish wins, and promotion stays deterministic.

**Disk usage.** K worktrees means K full checkouts of the tracked tree. This is
a required pre-build check, same as §5's budget question, but it does not need
external sign-off — it's answerable by inspecting this repo's own workspace
layout, not a cross-team decision. Confirm whether large inputs (`.cache/`,
`runs/`) are shared across worktrees (they can be, via a symlink into a
common directory outside the git-tracked tree) or would otherwise be
duplicated K times — if the latter, K needs a disk-aware ceiling in addition
to the concurrency-budget ceiling already in §3. Record the answer here before
implementation starts.

**Compute budget (CPU/threads) — real, unaddressed gap.** Nothing in the
training path sets thread or core limits today — confirmed via a repo-wide
search: zero hits for `n_jobs`/`num_threads`/`nthread` in `src/labpilot`.
Generated training code gets library defaults, which for LightGBM/XGBoost/
scikit-learn is typically "use every core." K concurrent branches under those
defaults oversubscribe the same physical cores instead of getting K× the
throughput — cache thrashing and context-switch overhead can make wall-clock
*worse* than sequential, directly undermining exit criterion 1, the actual
justification for this milestone. Unlike §5, this needs no external sign-off
— it's fully implementable inside M11: compute a per-branch cap
(`available_cores // K`, or a configured ceiling) before fan-out and inject it
into the generated training code via `agents/coding.py` (M19's delta-codegen
path). Prefer wrapping execution with `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/
`OPENBLAS_NUM_THREADS` env vars over a prompt instruction telling the LLM to
set `n_jobs` — env vars are enforced regardless of what the generated code
does; a prompt instruction depends on the LLM actually complying.

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
| Claim mechanism | file lock vs. conditional DB update against the mirror | file lock | The mirror is a best-effort cache, not the source of truth `get()`/`list()` actually read — a DB-side conditional update wouldn't touch the real race |
| Write serialization | WAL only vs. WAL + module-level writer lock | both | WAL alone permits concurrent writers to still race on read-modify-write app logic; an instance-level lock (à la `BudgetLedger`) doesn't work here since `ConductorStore`/`KnowledgeStore` are constructed fresh per call — the lock must be module-level |
| Promotion storage | field on `Experiment` vs. `manifest.metadata` | `manifest.metadata` | `Experiment` is assembled fresh on every call, not persisted — a pydantic field has nowhere to round-trip; `manifest.metadata` already has a writer (`save_manifest`) |
| Worktree cleanup | teardown-only vs. teardown + startup reconciliation | both | Teardown alone doesn't survive a hard crash; reconciliation is what actually closes the orphan risk already visible in this repo |
| Compute isolation | leave library defaults vs. inject per-branch thread cap | inject cap, via env vars | Library defaults (all-cores) oversubscribe under K-way fan-out and can make wall-clock worse than sequential; env vars don't depend on the LLM cooperating the way a prompt instruction would |

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
- **Compute contention**: run K=3 branches under a fixed, small core count
  (e.g. 2 cores via `taskset`/cgroup in the test), assert each branch's
  training process is capped near `cores // K` rather than the library
  default, and that the Perf test's wall-clock bound isn't violated by
  oversubscription.
