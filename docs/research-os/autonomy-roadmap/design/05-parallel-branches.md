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
   `src/labpilot` returns zero hits. LightGBM/XGBoost default to using every
   core, and generated code reaching for the common `n_jobs=-1` idiom does the
   same on scikit-learn. K concurrent branches under that oversubscribe the
   same physical cores instead of getting K× the throughput — the parallelism
   this milestone builds toward can end up *slower* than sequential, not
   faster (see §8 for exactly what does and doesn't fix this).

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
  branches, including the reflection calls losers trigger — already true via
  the gateway's existing pace-and-retry behavior (§5); no new code needed.
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
- Each branch's execution is fully auditable: its own `DecisionRecord`, its
  own checkpoint, and its own contribution to the consecutive-failure circuit
  breaker — full parity with sequential dispatch, decided below (see §8).

**Both design decisions below are now resolved** — §5 (budget scoping) needed
no new code once the existing gateway behavior was checked against the
threading model; §8 (auditability) is decided as full parity, not deferred.

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
  `runtime` field defaulting to `"local"` (§7, §8) so that work can build on
  this fan-out later without retrofitting it — and, until it does, so a
  remote-bound item is refused rather than run locally under a label nobody
  reads.
- Arbitrary branch-merge policy or conflict resolution beyond "pick one
  winner, keep the losers as evidence."

## 5. Budget scoping — resolved, no `fitroute` changes

Original concern: K branches sharing one `BudgetLedger` could starve each
other, and a hard K-way pre-split (`ParallelBudget` sized per branch) looked
like the fix. That was wrong on its own terms before it was wrong on
mechanism — forks don't need equal LLM budget, so pre-allocating equal shares
either starves a branch that needs more or wastes budget reserved for one
that needs less.

**The pacing mechanism this needs already exists.**
`RoleBoundClient.complete(..., allow_wait=True)` — the default —
(`src/fitroute/gateway.py`) does not fail on exhaustion; when `select_route`
reports no provider available, it sleeps `decision.wait_seconds` (bounded by
`spec.max_wait_seconds`, 900s default — `fitroute/catalog.py`) and retries,
rather than raising. `wait_seconds` comes from `BudgetLedger.availability()`,
which is already `threading.RLock`-guarded for concurrent callers
(`fitroute/budget.py`).

Each of the K branches runs on its own OS thread today
(`ExperimentAgent.execute` via `anyio.to_thread.run_sync` — confirmed in §1's
background research). A branch that hits a rate limit sleeps on **its own**
thread — it does not block the other branches, which keep making calls
against the same, correctly-locked ledger. That is exactly the "wait for the
next-retry time and resume" behavior the budget question needed, and it costs
M11 nothing — `ExperimentSpecialist` already calls through this gateway.

**No pre-split, no `fitroute` change, no M10 sign-off needed.**

**Measured limit — a branch gets exactly one wait** (task 8,
`tests/unit/test_branches_pace_independently.py`). The residual above was first
written as "each independently waits and retries — not perfectly efficient (a
small thundering-herd retry burst)", which reads as a throughput note. It is not.

`_complete_once` sleeps `wait_seconds`, re-selects once, and raises
`RoleUnavailable` if the provider is still spent; the outer `complete` retry
loop passes `allow_wait=allow_wait and attempt == 1`, so nothing waits a second
time. One window's worth of branches is served after that single wait and the
rest **fail** — they do not run slower.

The consequence is accounting, not efficiency: each excess branch is recorded as
a *failed experiment* and counts against the circuit breaker
(`max_consecutive_failures`, default 3). A wide fan-out against a slow provider
spends K worktrees to get a window's worth of results and a run of breaker
strikes — for a rate limit, not for anything the science did.

So **K is bounded by the provider's rpm, not only by cores**: keep K within a
small multiple of the narrowest provider the role can reach, or give the role a
second provider so exhaustion degrades sideways instead of waiting.

*An earlier version of this section tabulated exact served/failed counts per
(K, rpm) as `served = min(K, 2 × rpm)`. That table was wrong to state as
behaviour: `select_route` and the ledger `record` are not atomic, so how many
branches slip through before the first one records is a property of thread
scheduling. Identical code served 2-of-3 in 40/40 trials on a ten-core machine
and 3-of-3 on a two-core CI runner. The rule above — one wait per branch — is
what the code guarantees; the counts were the scheduler.*

## 6. Design

```text
Conductor step
   │
   ├─ select top K untested hypotheses (K = min(desired fan-out, max_workers)
   │     — K and max_workers must agree, or the compute-budget math below is
   │     wrong, see §8)
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
| `conductor/loop.py` | New: build K `ParallelWorkItem`s instead of one `OsTask`, call `run_parallel_sync`; emit a `DecisionRecord`/checkpoint per branch and feed each branch's outcome to the circuit breaker | Existing single-hypothesis path stays as the K=1 case; full audit parity with sequential dispatch, decided in §8 |
| `agents/parallel.py::ParallelWorkItem` | Minor: add `runtime: str = LOCAL_RUNTIME` field, validated before the batch starts (not unread — see §8) | Otherwise reused as-is; M5's concurrency primitives are sufficient |
| `execution/training/environment.py::child_environment` | New: inject a per-branch thread-cap env vars into the training subprocess's environment | See §8. **Not** `agents/coding.py` — that only generates code, it never executes it; `execution/training/runner.py::TrainingRunner.run()` is the actual `subprocess.run(...)` call, and `child_environment()` already builds the env dict it uses |
| `git_evolution.py` / `git/python_backend.py` | New: worktree create/teardown per branch, crash-safe | `create_branch` today mutates the single working tree; needs a worktree-based sibling, not a modification of the existing checkout path |
| *(new)* reconciliation check | New: startup-time `git worktree prune` + orphan sweep | Closes the crash gap — see §8. A worktree whose creating process dies before teardown runs is a standard git-worktree failure mode, not something this design can assume away |
| `hypothesis.py::mark_testing_if_proposed` | Fix: atomic claim on the JSON file (not the DB mirror) + release-on-setup-failure | See §8 |
| `accessor/sqlite/client.py`, `conductor/store.py` | New: WAL + module-level serialized writer | See §8 |
| `agents/events.py` / `agents/subscribers.py` | New subscriber: comparison + promotion | Same pattern as the existing evidence-refresh and experience-memory subscribers on `ExperimentCompleted` |
| `agents/experiment.py::event_payload` | New: `completed_at` field | Needed for deterministic tie-break, see §8 |
| `shared/experiments/graph.py::assemble_experiment` + manifest write path | New: read/write a `promoted` flag via `manifest.metadata`, not a new `Experiment` field | See §8 |
| `fitroute/budget.py` | None | Resolved in §5 — the existing gateway pacing already covers K concurrent callers, no change needed |

## 8. Implementation notes

**Auditability & breaker accounting — decided: full parity.**
`run_parallel_async` calls `item.agent.execute(...)` directly; it does not go
through `Scheduler.dispatch`, `store.enqueue`/`OsTask`, `DecisionRecord`, or
`_record_experiment_outcome`'s consecutive-failure circuit breaker
(`conductor/loop.py`) — all of which wrap every other tool dispatch today.
That file's own comments describe past silent-degradation bugs this machinery
exists to catch, so K-way fan-out gets the same treatment, not a lighter one:
each of the K branches produces its own `DecisionRecord` and checkpoint after
its `ParallelResult` comes back, and each branch's outcome (ok/failed) feeds
`_record_experiment_outcome`'s breaker the same way a sequential failure
would — a step with 2 of 3 branches failing counts as 2 failures toward the
breaker, not 1 step. This is more wiring in `conductor/loop.py` than a
step-level shortcut would be, but it doesn't reopen the silent-degradation
class of bug `_record_experiment_outcome`'s comments describe.

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

**Path safety in the worktree module is carried by a type, not by checks.**
Every path `agents/git_worktree.py` deletes is resolved *and* strictly inside
`.worktrees/`; both properties live in `_SafeTarget`, and
`_force_unregister` — the only function there that deletes anything — accepts
nothing else. A new destructive operation cannot skip either property because
it cannot construct the argument.

That shape was arrived at the expensive way, and the reasoning is worth
keeping because the alternative looks cheaper every time. Five review rounds
on the module produced five variants of **one** mistake — a path property
applied at some call sites and not others:

| Round | What was missing | Consequence |
|---|---|---|
| 1 | containment in `create`/`remove` | `rmtree` deleted `knowledge/` (with `knowledge.db`) before git rejected the refname |
| 2 | containment at its own boundary — a dead clause permitted the root | `rmtree` on `.worktrees/` destroyed *every* concurrent branch |
| 3 | containment in `reconcile`'s separate copy of the rule | same, from the unattended startup sweep |
| 4 | normalization between returning and reporting | `wt.path in result.removed` False on any symlinked workspace |
| 5 | the dirname duplicated as a literal outside the module | a rename would silently un-ignore K full checkouts |

Rounds 1–3 were each fixed by centralising the *check*, which kept working and
kept not working: a check is still a discipline every call site must remember,
so the omission simply moved — and round 4 proved it by appearing in a
different *property*. Round 5 then found the duplication reappearing just
outside the boundary the type had drawn.

The rule for anyone extending this: normalising an *input* at a public entry
point is fine; a *decision* about whether a path is safe belongs only in
`_SafeTarget._check`, and the directory name has exactly one definition
(`labpilot.workspace.WORKTREE_DIRNAME`, which `REQUIRED_IGNORES` is built
from).

**Atomic writes — a second reader-vs-writer race, found by review, not
designed for up front.** Locking the mutators (`HypothesisStore`,
`EvidenceCardStore`) closes writer-vs-writer races, but `get()`/`list()`
callers never took that lock — deliberately, to keep reads lock-free — which
only works if a reader can never observe a half-written file. It could:
`path.write_text()` truncates then writes as two separate steps, so a reader
landing in between saw a 0-byte file. Fixed once, generically
(`accessor/common/atomic_write.py::atomic_write_text` — write to a temp file,
`os.replace` onto the real path, atomic on POSIX), and applied to **both**
file-backed stores — `EvidenceCardStore.save()` got the identical fix
`HypothesisStore._write_json` did, not just the id-allocation lock it already
had. Verified empirically both times: reverting to the old write reliably
produced 1000+ torn reads under a synchronized reader/writer stress test;
the fix produces zero.

**The same torn-read race a third time, in git's store rather than ours — so
creation retries instead.** `git worktree add` registers itself by writing
`.git/worktrees/<name>/gitdir` and *then* `commondir`, neither atomically.
Every git command that enumerates worktrees calls `get_worktrees()` first —
`add` and `remove` included, not just `list` — and a registration whose
`gitdir` exists while `commondir` is still zero bytes is **fatal**, not
skipped:

```
fatal: failed to read .git/worktrees/exp-3/commondir: Undefined error: 0
```

(errno 0 because the read hits EOF on an empty file rather than failing.)
This is precisely the `write_text` truncate-then-write window above, in a
store we do not own and cannot make atomic.

It is not theoretical: it flaked
`test_concurrent_creation_gives_each_thread_its_own_tree` once during a
full-suite run and then passed three reruns, which is what sent this to
investigation. Measured on git 2.47.1 by running the test's own fan-out in a
loop with per-thread exceptions captured: **9 failing rounds in 250 at K=6,
and 9 in 250 at K=8** — ~3.6% of fan-outs, one single failure mode, no others.
Under task 7 that is a campaign branch aborting a few percent of the time.

The window is microseconds wide and closes on its own, so
`create_experiment_worktree` retries — four attempts, exponential backoff from
50 ms. Three details are deliberate:

- **It retries any failure, not a match on git's message.** That message goes
  through gettext (`fatal: failed to read %s`), so a non-English git would
  walk straight past a string match. Retrying blind is safe here only because
  every repeated operation is idempotent by construction: `_force_unregister`
  tolerates absence and `add -B` force-resets. A permanent failure costs the
  backoff budget and is then re-raised **as git wrote it** — unwrapped, so the
  real cause is what reaches the caller.
- **The unregister and `mkdir` are inside the loop, not hoisted above it.** A
  concurrent `add` breaks *this* call's `worktree remove` the same way, and
  `_force_unregister` swallows that by design — so an attempt can fail with
  "already registered" purely because its own cleanup was the casualty.
  Repeating the cleanup is what makes the next attempt a retry rather than a
  rerun of the same broken state.
- **Path validation stays outside the loop.** A branch resolving out of
  `.worktrees/` is wrong on every attempt; retrying it would only re-enter the
  deleting path against a target already judged unsafe.

Verified the same way as the atomic-write fix: 800 further rounds (400 at K=6,
400 at K=8) with the retry in place, on the same machine and harness that
produced the 18 baseline failures.

**Not fixed, and deliberately so:** `list_registered_worktrees` and
`_force_unregister` are victims of the identical window. Both are reached from
`reconcile_worktrees`, which task 7 runs at campaign *start*, before any
fan-out — no concurrency within a campaign. Two campaigns sharing one repo
would race, but a retry would not make that safe: reconcile removes any
worktree not in *this* campaign's `live_branches`, so it would happily delete
a worktree the other campaign just created. That is a scoping question for
multi-campaign support, not a retry.

**Locking one store's own methods isn't enough — audit every entry point
into its data.** `HypothesisStore`'s five mutators are all locked, but
`evidence/apply.py::apply_card_to_hypothesis` reached past the public API
into `store._save()` directly for a second, unlocked read-modify-write right
after a properly-locked `update_outcome()` call — found by review, not by
design. Fixed by extending `update_outcome` with an optional `confidence`
parameter so the whole mutation is one locked call, not two. The lesson this
leaves for future stores: "every mutator is locked" isn't the same claim as
"every write path is locked" — grep for `_save`/`_write_json`-style private
writes from *outside* the class, not just inside it.

**Write serialization — implemented, and more precise than first designed.**
Two different problems turned out to need two different fixes, not one:

- A single atomic SQL statement (`increment_metric`'s
  `SET field = field + ?`) is safe under concurrent writers regardless of any
  application lock — SQLite guarantees it. Verified with a 20-thread
  barrier-synchronized run: zero lost updates, with or without WAL. `WAL` +
  an explicit `busy_timeout` were still added to `SqliteClient.__init__` (read
  concurrency during a parallel step matters, and WAL removes "readers block
  behind an in-flight writer"), but **not** because the un-patched default
  raises `database is locked` — `sqlite3.connect`'s own implicit 5s timeout
  already covers that case here, so that specific risk this paragraph
  originally named was not real.
- Allocating an id then inserting a row that uses it (`new_decision_id` +
  `append_decision`) is a genuine multi-statement TOCTOU race no `busy_timeout`
  closes — confirmed: 6 of 20 concurrent, unlocked attempts raised
  `IntegrityError: UNIQUE constraint failed` (two callers read the same "next
  id" before either wrote). This is the one that needed the module-level lock
  — an instance-level lock like `BudgetLedger`'s `threading.RLock` doesn't
  work here since `ConductorStore` is freshly constructed at each call site,
  not shared for the process's lifetime the way `BudgetLedger` is. Fixed by
  `write_lock_for(db_path)` (`accessor/sqlite/client.py`) — keyed per database
  file, not one process-wide lock, so unrelated competitions don't serialize
  on each other. Callers can't forget it: `ConductorStore.append_new_decision`
  encapsulates the whole allocate-then-insert sequence under the lock, so K-way
  fan-out (task 7) calls that method, not the raw two-call pattern. The same
  fix (`write_lock_for` for SQL stores, or the equivalent file-lock helper —
  `accessor/common/file_lock.py` — for JSON-backed ones) was applied to every
  store sharing this exact allocate-then-insert shape that M11 actually
  touches (`HypothesisStore`, `EvidenceCardStore`, `ExperienceStore`); stores
  M11 doesn't touch (`PlanStore`, execution's and reflection's id-allocators)
  were left as-is — same latent shape, genuinely out of this milestone's
  scope, `write_lock_for`/`locked()` are there for whoever picks them up.

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
timestamp field today — add a `completed_at` field to `event_payload`. Ties on
the metric are then broken by earliest `completed_at`, the first branch to
finish wins, and promotion stays deterministic.

*Implemented at run completion, not at publish time as this paragraph
first specified.* Publish is separated from the run by `_load_metrics` and
`write_experiment_git_record`, and the latter's cost scales with
`files_changed`. Stamping after them would fold record-writing time into the
comparison, so a branch that finished first but wrote a large record could
lose a tie-break to one that finished later and wrote less — the ranking would
partly measure record size. The stamp is therefore taken as soon as `run_plan`
returns, and attached to the payload only on the success path: the same dict
is the `ModelFailed` payload, and a `completed_at` on a run that died would
assert the completion that `ExperimentSpecialist.execute`'s early return —
and `tests/unit/test_failed_run_is_not_completed.py` — exist to deny.

**The `runtime` field.** `ParallelWorkItem` gains `runtime`, and it
is *validated* rather than carried unread as this document originally
proposed. An unread field would have been dead code by definition, and worse
than absent: an item asking for Kaggle would run locally, finish, report a
metric, and leave nothing downstream able to tell the answer came from the
wrong machine. `run_parallel_async` therefore refuses any non-local runtime
before the batch starts — up front, because the mistake is knowable without
running anything and refusing late means having already spent budget and
compute on siblings that were never going to add up to the fan-out asked for.

The value is not a new vocabulary. It is the `provider` discriminator from
`execution/runtimes/models.py`, where `LocalRuntime.provider` is `"local"`
and the siblings are `"kaggle_kernel"`, `"google_colab"` and `"other"` — the
Runtime abstraction this document's non-goals already point at.
`parallel.py` keeps `LOCAL_RUNTIME` as a literal rather than importing those
models: `execution.runtimes` is *not* otherwise imported by the agents
package (checked, not assumed), so importing it would add a dependency for
one string. `test_parallel_workers.py` asserts the copy equals
`LocalRuntime(id=...).provider`, so the two cannot drift unnoticed.

**Branch bookkeeping runs off the event loop.** `ExperimentSpecialist.execute`
offloaded `snapshot_before_experiment` and `run_plan` but left the metrics
read and the git-record write (cost scales with `files_changed`) inline. All K
branches share one loop, so each branch's bookkeeping stalled every sibling:
the fan-out would serialise on exactly the work §5 argues is cheap. Both now
go through `anyio.to_thread.run_sync`, matching the two calls that already
did. The stat that decided the metrics `ArtifactRef` folded into the read —
a successful `read_text` already proves the file exists, so it is only needed
to tell an absent file from a corrupt one. This had to land before the
budget-pacing test, which measures whether fan-out beats sequential and would
otherwise have measured this instead.

Two consequences worth naming. **The per-branch worktree is now load-bearing
for write safety, not just for checkout isolation.** `write_experiment_git_record`
writes one `experiment/record.json` per workspace root under no lock; while it
ran on the loop, two branches could never be inside it at once, so distinct
roots were a convenience. In threads they genuinely overlap, and two branches
sharing a root would tear that file. **The read is ordered before the
write** so that no `await`, and therefore no cancellation point, separates the
record landing on disk from the `ExperimentCompleted` that announces it —
otherwise a cancelled branch could leave a record no subscriber ever saw.

**Disk usage — answered, and the disk was the smaller half.** K worktrees means
K checkouts of the *tracked* tree; `git worktree add` never copies ignored
files, so `data/`, `.cache/` and `models/` are safe wherever `.gitignore`
actually carries the line — true of a freshly-scaffolded workspace, not of one
that predates it or had the line hand-edited away. `LARGE_INPUT_IGNORES`
closes that gap the same way `SHARED_STATE_IGNORES` does below: reconciled
into an *existing* workspace by `ensure_required_ignores`, not only written
into the fresh-scaffold template. What was not safe at all, template or not,
is that `init_git_repo` runs `git add -A` once at scaffold, so everything else
present at that moment became tracked — while every later commit goes through
`snapshot_before_experiment`, which only touches
`CODE_PATHS = ("pipeline", "src", "configs", "tests")`. `knowledge.db` and
`runs/` were therefore tracked by accident, committed once, never updated
again, and copied into every branch.

Measured on a 734 MB workspace (300 MB `data/`, 200 MB `.cache/`, 100 MB
`models/`, 60 MB `runs/`, 35 MB `knowledge.db`, incompressible bytes):

| | tracked as before | `knowledge.db` + `runs/` ignored |
|---|---|---|
| after `git init` | 839 MB | 729 MB |
| per worktree | **+104.9 MB** | **+0.06 MB** |
| K=5 total | 1364 MB | 729 MB |

So K needs no disk-aware ceiling. The fix is `SHARED_STATE_IGNORES` in
`workspace.py`: `/knowledge/**/knowledge.db` rather than all of `knowledge/`,
because the hypothesis JSONs beside it are small and deliberately tracked;
anchored under the workspace root rather than bare `**/`, because an
unanchored pattern also matches a same-named path nested under tracked code —
a pipeline script that writes its own output to `pipeline/runs/`, for
instance, would have that output silently untracked by a pattern meant only
for the workspace-level `runs/`.

**The anchor itself needed a second pass — anchoring is not the same as
anchoring at the right depth.** The first anchored form, `/knowledge/knowledge.db`,
matches nothing real: `ResearchPaths.db_path` (`intelligence/paths.py`) puts
the database at `knowledge/research/knowledge.db` for a client workspace and
`knowledge/<competition>/research/knowledge.db` for legacy — one or two
directories past where a bare anchor reaches. That reopened the exact bug
this section exists to close, silently, because the unanchored `**/knowledge.db`
this replaced had matched the real path fine; the anchoring fix broke the
primary case while fixing the false-positive one. Found by review, and by the
same mechanism as before: the test asserting the pattern matches real artifact
names planted its fixture at the convenient top-level path rather than
resolving it through `ResearchPaths` the way every actual store does.
`/knowledge/**/knowledge.db` — anchored at the workspace root, open depth
below it — matches all three real shapes and still excludes `pipeline/knowledge.db`.

Reconciling a group whose *header* already exists but has gained a new
pattern needed one more check than reconciling patterns alone: the loop
originally appended a fresh header on every group with a missing pattern, so
re-running it after `SHARED_STATE_IGNORES` grew a second entry duplicated the
header already written for the first. `ensure_required_ignores` now checks
whether a group's header line is already present before writing it again —
found by review, not by the tests that shipped with the first version, which
only ever exercised a single pass over a `.gitignore` missing an entire group.

**The real reason is staleness and drift, not bytes.** A per-branch
`knowledge/` is not merely a wasteful copy — it is a *fork*. Each branch would
read a snapshot frozen at campaign start, so nothing a sibling learns mid-step
is visible to the others, and each would write its findings into a private file
that teardown deletes. K branches would end the step having diverged from each
other and from the shared base, with every claim, hypothesis transition and
evidence card written into a copy nobody reads. That also silently voids §8's
atomic claim and write-serialisation work above: both exist to make *shared*
concurrent writes safe, and there is nothing to serialise once each branch owns
its own database.

Isolation is wanted for the *working tree* — the code a branch edits and would
otherwise clobber. It is not wanted for research state, which is the one thing
every branch must agree on.

**Ignoring is only half of it.** Keeping these paths untracked stops them being
*copied*; it does not make a branch read the shared ones. That needs the
`Workspace` facade to split "where the code is" from "where the research state
is". Without it, a branch simply recreates an empty `knowledge/` inside its
worktree and the drift returns.

**Resolved — `Workspace.for_branch(code_root)`.** Of the two options this
section offered (overridable fields, or symlinking `data/` and `.cache/` into
each worktree), the fields won. Symlinks looked cheaper — read-only data, no
copy, relative paths unchanged — but they put a second source of truth on
disk for every branch, and `git worktree remove` refuses to delete a tree
whose contents it does not recognise, so teardown would have had to unpick
them in the right order. Fields keep the whole question in one process.

`for_branch` points `root` at the worktree, so `pipeline_dir` and
`artifacts_dir` follow it — that is the isolation — and *pins* the shared
locations: `data_dir` and `cache_dir` through the new
`data_dir_override`/`cache_dir_override` fields, `effective_runs_dir` through
the `runs_dir` field that already existed. Pins resolve before the copy, so
branching a branch keeps the original locations rather than compounding.

The path properties had to change to make this work at all. Each one
short-circuited to `self._client` first, so under the client layout they
returned the shared workspace's directory no matter what `root` said — asking
for a worktree root changed nothing. They now resolve the layout's *relative*
name against the current `root`, which also keeps a workspace with custom
directory names working on a branch.

One consequence worth stating: `ensure_roots()` skips
`ensure_required_ignores` on a branch. `.gitignore` is tracked, so appending
to the copy inside a worktree would dirty the branch before its experiment
starts and fold an unrelated edit into the snapshot commit and
`files_changed`.

**Session liveness — a status cannot answer it.** The startup sweep preserves
worktrees belonging to `running` or `paused` sessions, because sweeping on
"unknown means dead" would delete a concurrent campaign's checkouts
mid-experiment. But every transition to a terminal status runs *inside* the
loop, so a process killed by SIGKILL, OOM or power loss leaves its session
`running` for good — and the sweep then preserved exactly the worktrees it was
written to reclaim, permanently, with `create_experiment_worktree` failing on
those experiment keys forever after.

Campaigns therefore stamp `metadata.owner = {pid, host}` at start
(`claim_session_ownership`), and the sweep treats a session as dead only when it
can *prove* it: same host, and `os.kill(pid, 0)` raising `ProcessLookupError`.
No stamp, a different host, or an unreadable value all count as live — guessing
wrong deletes a running experiment, while over-keeping costs one stale worktree
that the next provable case clears. A heartbeat threshold was the alternative
and needs a number nobody can pick: a training step can legitimately run for
hours without touching the row.

**Fan-out width is bounded by cores too.** `resolve_k` caps K at
`available_cpus()` and logs when it does. Nothing else bounded it — `-k 64` on a
ten-core box checked out 64 worktrees and started 64 training runs, with
`cpu_share` handing each a 1-core share and no sign the request was degenerate.
This is the disk/CPU half of the ceiling; §5 has the LLM half
(`served = min(K, 2 × rpm)`).

**Migration gap — closed in `_untrack_shared_state`.** A `.gitignore` pattern
does not untrack a file that is already tracked, so workspaces scaffolded
before this change keep copying `knowledge.db` and `runs/` into every
worktree. `ensure_required_ignores` fixes the pattern but cannot fix the
index. The conductor now untracks them (`git rm -r --cached`) beside
`reconcile_worktrees` at campaign start, and only when `branches > 1` — it
rewrites a user's index, so it belongs with the feature that depends on it
rather than in a helper every command runs.

Two details that bit: `SHARED_STATE_IGNORES` holds *gitignore patterns*, and a
leading `/` anchors those to the repository root but means an absolute path to
a git pathspec — `git ls-files -- '/runs/'` dies with `fatal: Invalid path
'/runs'`. `_as_pathspecs` strips it, deriving from the constant so a pattern
added there cannot go unhandled. And `git rm` aborts the whole invocation when
any one pathspec matches nothing, so a workspace with `runs/` tracked but no
`knowledge.db` would have untracked neither without `--ignore-unmatch`.

**Compute budget (CPU/threads) — real, unaddressed gap.** Nothing sets thread
or core limits today — confirmed via a repo-wide search: zero hits for
`n_jobs`/`num_threads`/`nthread` in `src/labpilot`. Note scikit-learn
estimators default to `n_jobs=None` (serial) — the actual risk isn't "library
defaults use every core," it's **generated code explicitly passing
`n_jobs=-1`** (a common LightGBM/XGBoost/sklearn idiom the LLM is likely to
reach for), which spawns joblib/loky workers sized off `loky.cpu_count()`.
`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` don't govern
joblib/loky — `LOKY_MAX_CPU_COUNT` does, and needs to be set alongside them.
Even then, a hard-coded explicit count (`n_jobs=8`, not `-1`) in generated
code is a residual risk env vars don't fully close — flagging this as
accepted, not solved, pending an OS-level fallback (cgroup/`taskset` around
the `TrainingRunner.run()` subprocess) if it proves necessary in practice.

The injection point is `execution/training/environment.py::child_environment()`
(consumed by `execution/training/runner.py::TrainingRunner.run()`'s
`subprocess.run(..., env=child_environment())`) — **not** `agents/coding.py`,
which only generates `train.py` and never executes it.

**Resolved: environment variables only, and here is why the alternative was
rejected.** The open question was env vars versus env vars plus OS-level
enforcement (a cgroup or `taskset` around the training subprocess). Settled by
checking rather than preferring: `taskset`, `cgexec` and `systemd-run` are all
Linux-only and absent on macOS, which is where this is developed and run. An
enforcement layer that does not exist on the development platform is not
enforcement, so it cannot be the primary mechanism. The residual gap it would
have closed — generated code with a **hard-coded** `n_jobs=8` rather than
`-1`, which no environment variable governs — is therefore accepted and named
rather than solved.

Implementation notes that follow from the above, all in
`execution/training/compute_budget.py`:

- The variable set spans OpenMP, the three BLAS families, numexpr, loky and
  polars/rayon. `VECLIB_MAXIMUM_THREADS` is included because Apple Accelerate
  backs numpy on the common dev platform, and `LOKY_MAX_CPU_COUNT` because it
  is what `n_jobs=-1` actually consults. **The list cannot be complete**:
  generated code declares its own dependencies via PEP 723, so a library whose
  pool is governed by an unlisted variable runs uncapped. That is the same
  open-world objection this codebase already used to reject a package
  allowlist, and it applies here — the difference being that an unknown
  package fails loudly at import while an unknown thread variable just means
  one library quietly ignores the budget. Worth keeping accurate; not worth
  presenting as exhaustive.
- The share travels in a `ContextVar`, not `os.environ`: branches are
  concurrent threads in one process, so an environment variable is shared
  between them and the last writer would set the value for all. A context
  value propagates into `anyio.to_thread.run_sync` and stays per-task.
- `available_cpus()` prefers `os.process_cpu_count()` (3.13+) and
  `sched_getaffinity` over `os.cpu_count()`, since only those respect affinity
  and cgroup limits — in a container pinned to 2 cores, `os.cpu_count()`
  reports the host's count and every branch would be licensed to oversubscribe
  the very limit the container imposed.
- `cpu_share()` returns `None`, never `0`, for "cannot determine", and floors
  at 1 otherwise (`2 // 3` is `0`). These variables read `0` as *unset, use
  every core*, so the no-cap meaning must never be carried by a number that
  means its opposite.

## 9. Tradeoffs

| Decision | Options | Choice | Why |
|---|---|---|---|
| Branch isolation | worktree vs. sequential checkout-lock (mutex around `create_branch`) | worktree | Checkout-lock defeats the purpose — branches would serialize on the working tree, which is what M11 exists to remove |
| Claim mechanism | file lock vs. conditional DB update against the mirror | file lock | The mirror is a best-effort cache, not the source of truth `get()`/`list()` actually read — a DB-side conditional update wouldn't touch the real race |
| Write serialization | WAL only vs. WAL + module-level writer lock | both, for different reasons | WAL is for read/write concurrency, not writer-vs-writer safety — a single atomic UPDATE is already safe without any lock (verified). The lock is for multi-statement allocate-then-insert sequences specifically (verified: 6/20 unlocked concurrent id-allocations collided); an instance-level lock (à la `BudgetLedger`) doesn't work there since `ConductorStore` is constructed fresh per call |
| Promotion storage | field on `Experiment` vs. `manifest.metadata` | `manifest.metadata` | `Experiment` is assembled fresh on every call, not persisted — a pydantic field has nowhere to round-trip; `manifest.metadata` already has a writer (`save_manifest`) |
| Worktree cleanup | teardown-only vs. teardown + startup reconciliation | both | Teardown alone doesn't survive a hard crash; reconciliation is what actually closes the orphan risk already visible in this repo |
| Compute isolation | no cap vs. env-var cap vs. env-var cap + OS-level enforcement | **env-var cap** (resolved) | OS-level enforcement cannot be primary: `taskset`, `cgexec` and `systemd-run` are Linux-only and absent on macOS, where this is developed and run. `OMP_NUM_THREADS`+`LOKY_MAX_CPU_COUNT` cover the realistic failure (`n_jobs=-1`); a hard-coded count stays uncovered and is accepted rather than solved — see §8 |
| LLM budget under fan-out | pre-split `ParallelBudget` K-ways vs. reuse the gateway's existing pace-and-retry | reuse existing gateway behavior | A fixed split is wrong on its own terms (forks need unequal budget); `RoleBoundClient.complete(allow_wait=True)` already sleeps-and-retries per caller thread, which is sufficient once each branch runs on its own thread — zero new code, see §5 |
| Auditability under fan-out | step-level `DecisionRecord`/breaker vs. per-branch parity | per-branch parity | A step-level shortcut is less code but a branch failing wouldn't show up in the audit trail the way a sequential failure does today — explicitly rejected in favor of not reopening the silent-degradation bug class `conductor/loop.py` already guards against |

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
- **Concurrency tests must name their own cause.**
  `test_concurrent_creation_gives_each_thread_its_own_tree` originally ran
  `create_experiment_worktree` in a bare `threading.Thread` with no exception
  capture, so the git race in §8 surfaced only as `assert len(made) == n` — a
  count that named neither the failing thread nor git's error, on a test that
  then passed three reruns. Each worker now captures its own traceback and the
  assertion reports them, cause before symptom. The rule this generalises to:
  a thread whose exception dies inside `Thread.run` converts every concurrency
  bug into the same uninformative arithmetic failure.
- **Race fixes get a deterministic test, not a stressier one.** The retry is
  pinned by a `GitTool` wrapper that fails `worktree add` a set number of times
  with git's verbatim message — covering the retry, the bounded exhaustion
  re-raising git's own error, and validation *not* being retried. Reproducing
  the real window needs two threads inside a microsecond, which is exactly why
  the bug reached main; the stress loop measures the rate, the fake pins the
  behaviour, and only the fake belongs in the suite.
- **Claim rollback**: force worktree setup to fail after a successful claim,
  assert the hypothesis returns to `proposed` rather than sticking in
  `testing`.
- **Write serialization stress**: implemented as
  `test_conductor_store_concurrency.py`. Two cases, not one — a single atomic
  UPDATE (`increment_metric`) needs no lock and is asserted safe under 8
  concurrent single-connection-per-thread writers; allocate-then-insert is
  fixed by routing through `append_new_decision()` (and the equivalent
  `append_new_feedback`/`append_new_suggestion`/`append_new_capability_decision`),
  which hold `write_lock_for(db_path)` across the whole sequence — not the raw
  `new_decision_id()` + `append_decision()` two-call pattern, and not the
  process-local `write_lock` this doc originally named (replaced with the
  cross-process `write_lock_for`, since branches may end up as separate OS
  processes, not just threads). Verified against a real failure: 6/20
  unlocked concurrent attempts raised `IntegrityError` before the fix.
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
- **Budget pacing across branches**: force two branches to exhaust the same
  provider's rate limit concurrently, assert both sleep-and-retry
  independently (one branch's wait does not block or fail the other) and
  both eventually complete rather than raising `RoleUnavailable` — the
  regression guard for §5's resolution.
- **Breaker parity**: run a step where 2 of 3 branches fail, assert the
  circuit breaker's consecutive-failure count increments by 2, not 1 — the
  regression guard for §8's full-parity decision.
