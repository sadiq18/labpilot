# Design — M16: the evidence routine as a background producer

**Plan:** [../11-background-routine.md](../11-background-routine.md) ·
**Status:** built behind `--gather-background`; measured 2026-08-20 (§3, §3.1), all exit criteria met, two follow-ups open (§11) ·
**Depends on:** M7, M11, M14 (all shipped) ·
**Ledger priority (§8):** built — `availability(..., reserve=)`

---

## 1. Problem

`should_gather_evidence` ships, and it is a **brake on a sequential loop**, not a
second worker. When the gate says yes, the campaign still stops testing and
sweeps kernels for ~15 minutes on the same thread — the observation that opened
the plan, unchanged.

The plan calls the remainder "mostly a scheduling change". Five things say
otherwise.

| # | Gap | Where |
|---|---|---|
| 1 | **No callable entry point.** The gate lives *inside* `available_tools`, which returns tool names; the invocation is an `OsTask` through `Scheduler.dispatch`. Nothing can ask "gather now, if you should" without the policy step this milestone bypasses | `conductor/policy.py`, `conductor/scheduler.py` |
| 2 | **No runner.** `_run_until_stop_inner` is `for step in range(max_steps)`, one dispatch per step. M11's fan-out is bounded to a single step and joins before the loop advances | `conductor/loop.py:1116` |
| 3 | **Content dedupe is unsafe under two writers.** `persist_recommendations` (the producer's own output) does not dedupe at all; `_already_covered_by_proposed` scans the proposed pool *outside* the lock `create()` takes. `create()` holds `.alloc.lock` across allocate-and-write, so two writers cannot collide on an **id** — only on an **idea** | `intelligence/hypothesis/persist.py`, `execution/outcome.py` |
| 4 | ~~**One weak claim call.**~~ **Checked and withdrawn.** `evaluator.mark_testing` does use `mark_testing_if_proposed`, and that is correct: the producer never claims — it proposes — so M16 adds no claimer, and its one caller (`execution/engineer.py`) usually runs *inside* a claim `prepare_branches` already made. An exclusive claim there would report "lost" on every branch of a healthy fan-out. Documented at the method instead | `reflection/hypotheses/evaluator.py:57` |
| 5 | **The LLM ledger has no priority.** `availability()` answers identically for every caller, so nothing implements the plan's "producer yields to consumer". Hypothesis generation is a `reasoning`-role call | `fitroute/budget.py` |

Gaps 1, 2 and 4 are latent-by-design. Gap 3 is latent only because there is
currently exactly one writer, which is what this milestone changes.

**One correction to the plan.** Step 4 ("feed reflection back into the
producer") is largely built — the consumer already mints from its own outcomes
at five sites (four `maybe_mint_*` in `execution/outcome.py`, plus
`mint_stagnation_hypothesis`) into the same store the producer writes to. What
the plan misses is the coupling: those mints raise the **viable count**, which
is the producer's own brake, so a campaign can hold gathering shut with
hypotheses it generated itself. That is the 46-row ratchet in a new costume, and
the reason the stagnant clause must stay independent of the count.

## 2. Requirements

**Functional** — 1–4 are the plan's exit criteria; 5–7 are what the design adds.

1. A campaign step never blocks on evidence gathering.
2. The producer re-evaluates the gate each tick and no-ops with a logged reason.
3. A thin pool refills without the consumer stalling.
4. Producer and consumer never claim the same hypothesis.
5. Producer and consumer minting the same idea concurrently produce **one** row.
6. Exactly one component gathers — the consumer's allowlist loses
   `analyze_competition` unconditionally, not via the gate.
7. Producer state is visible in the observe bundle.
8. **The producer owns no source policy.** It is handed *what to gather*; it
   decides only *whether* and *when*. No Kaggle constant is readable from
   `conductor/producer.py` (§5.4).

**Non-functional**

| Constraint | Number |
|---|---|
| Skip-path tick cost | **< 50ms** — one `MAX(created_at)` read, one pool scan, one score summary. It runs on a timer; the skip must be ~10⁴× cheaper than the sweep |
| Campaign shutdown delay | **≤ 1 pipeline stage** (~1 analyzer), not the full sweep |
| Producer crash impact on campaign | **zero** — every tick exception-isolated, same shape as `_maybe_mint_on_stagnation` |
| Behaviour change when off | **zero** — off is the default (§11) |

## 3. Success metrics — measured 2026-08-20

A paired run on one workspace, producer off then on. Sandbox clones of
`rogii-wellbore-geology-prediction` (never the operator's copy), identical
environment on both sides, `--max-steps 8 --yes --max-submissions 0`.

| | Producer **off** | Producer **on** |
|---|---|---|
| Wall clock | 847s | 1047s |
| Steps dispatched | 5 | 4 |
| Tools chosen | implement, run_experiment ×2, generate_plan, run_experiment | run_experiment ×2, generate_plan, run_plan |
| `analyze_competition` dispatched **by the campaign** | **0** | **0** |
| Research artifacts | 239 → **239** (+0) | 239 → **257** (+18) |
| Evidence age at end | **158.1h** | **0.02h** |
| Hypothesis pool (viable/proposed) | 10/136 → 10/136 | 10/136 → 10/136 |
| Producer ticks | — | 1 (gathering; still sweeping at shutdown) |

### What this shows, and what it does not

**The consumer never waited.** With the producer running, the campaign
dispatched four steps and `analyze_competition` zero times, while the sweep ran
beside it. Requirement 6 and exit criterion 1 hold as *observed* behaviour, not
just as a property of the allowlist.

**Evidence refilled; the hypothesis queue did not — in the 8-step run.** +18
artifacts and staleness collapsing from 158 hours to one minute is the producer
doing its job, but the pool stayed at 10 viable because the sweep was still
running when the campaign stopped. A third run settles it; see §3.1.

**The premise of criterion 1 did not reproduce.** The baseline never blocked on
gathering — not because it was prevented, but because it *declined*: the gate
reported "Evidence gathering available: only 10 viable hypotheses queued" at all
five steps and the LLM policy chose testing every time. The ~15-minute blocking
sweep in the plan's opening observation came from an earlier policy regime. So
**steps per hour is not the discriminating measure on this workspace** — the
two runs chose different tools and ran different experiments, and the 847s/1047s
difference measures that, not the producer. The measure that discriminates is
evidence freshness: 158h versus 0.02h, for the same campaign work.

That is a correction to §3 as originally written, and it moves what this
milestone is *for*. The failure it removes here is not a campaign stalled behind
a sweep; it is a campaign testing hard for fourteen minutes against evidence
five days stale, with a gate that says "go and look" every single step and a
policy that correctly refuses because it has work to do. Nothing else in the
system resolves that standoff.

**The provenance fix is confirmed in the field.** 29 invocations recorded from
the producer thread during the run — `RepositoryAnalyzerAgent` ×22,
`ConceptNormalizerAgent` ×5, `CompetitionPageAnalyzerAgent`,
`RepoQueryPlannerAgent` — none of which the campaign could have produced, since
it dispatched no `analyze_competition`. Before `contextvars.copy_context()` in
`start()` every one of those rows was dropped.

**Shutdown behaved as designed.** "Evidence producer still sweeping at shutdown;
leaving it to the process exit" — the bounded join (§7.4), taken rather than
made the operator wait out a multi-minute sweep.

### 3.1 The long run — criterion 3, met

A sweep on this workspace takes **~20 minutes**. An 8-step campaign does not
last that long, so the first two attempts measured the wrong thing. What bounds
the campaign is not `--max-steps` and not the failure breaker:

```
stop:failing — 2 consecutive failed execution(s), 8 step(s) since the last success
```

That is `max_barren_steps=8`. On a workspace whose experiments never succeed, it
fires long before anything else, and it has no CLI flag — raising `--max-steps`
and even `max_consecutive_failures` changed nothing across two runs. With
`max_barren_steps=40` set on the session, the third run gave the producer room:

| | Long producer-on run |
|---|---|
| Wall clock | **3083s** (51 min), 19 decisions, 18 dispatches |
| Consumer work | run_experiment ×10, generate_plan ×4, run_plan ×2, query_memory ×1 |
| `analyze_competition` dispatched **by the campaign** | **0** |
| Producer sweeps | **2 complete** (1344.6s, 1043.9s), a third in flight at shutdown |
| Hypotheses minted | **10 + 10** |
| Proposed pool | 136 → **153** (18 created during the run) |
| Research artifacts | 239 → **274** (+35) |
| Evidence age | 158h → **0.01h** |
| Provenance rows this run | 274, of which 20 `HypothesisGeneratorAgent` |

**Exit criterion 3 is met.** The queue refilled — 18 new proposals — while the
consumer dispatched 18 steps and never once waited on a sweep.

**And the run found something the design did not anticipate.** The producer
swept *continuously*: two full sweeps and a third started, ~40 minutes of
reasoning-role LLM work inside a 51-minute campaign. The reason is visible in
its own log —

```
Evidence producer: analyze_competition finished in 1344.6s, 10 hypothesis(es) added
Evidence producer: gathering — only 10 viable hypotheses queued
```

— ten hypotheses added, and the next tick still reads *only 10 viable*. Measured
at the end: 18 rows created during the run, `proposed` up by 17, and
`viable_hypothesis_count` **unchanged at 10 throughout**.

`viable_hypothesis_count` excludes rows the selector has passed over
`STALE_AFTER_SELECTIONS` (2) times, and every `generate_plan` ages every
row it does not pick. The run made six selections, so the producer's own output
aged out about as fast as it arrived. **The producer cannot satisfy the gate it
gates on**, and so it never stops.

That is not the M21 ratchet — the pool is not holding gathering *shut* — but it
is the same shape inverted, and it is worse for a background worker than for a
campaign step: a sequential campaign that re-swept would at least be visible as
a step it chose. Options, none of them free:

1. **Count freshly minted rows as viable for a grace period.** Smallest change;
   makes "viable" mean "not yet judged" rather than "not yet passed over".
2. **Give the producer its own signal** — e.g. gate the tick on artifact
   freshness alone once a sweep has landed, since staleness *is* moved by its
   own work (158h → 0.01h) while viability is not.
3. **Leave it, and rely on `_MIN_RESWEEP_HOURS`.** The default 0.5h floor would
   have allowed one re-sweep per 30 minutes rather than continuous ones; this
   run deliberately lowered it to 0.02h. That makes the default a load-bearing
   safety limit rather than the rate limit §5.2 calls it.

(2) is the honest one: the gate's three clauses were written for a consumer
deciding whether to spend *its own* step, and one of them does not survive being
asked by a worker whose job is to change the answer.

### Conditions, so the numbers are readable

* `evaluation_metric` was `null` in the workspace contract and the campaign
  refuses to run without one. Set in the sandbox to `rmse`/minimize — what this
  workspace's own `metrics.json` already records every prior run against, not a
  metric chosen for this exercise. **The research these runs produced is not
  meaningful; only the loop's mechanics and timings are.**
* `LABPILOT_VIABLE_HYPOTHESIS_TARGET=25` on both sides, so the 10-viable pool
  reads as thin — criterion 3's condition. `LABPILOT_MIN_RESWEEP_HOURS=0.02` so
  a producer could sweep more than once inside a bounded run; it did not get
  the chance to.
* One pair, one workspace, free-tier providers with observed 429s and failovers.
  This is an existence proof about mechanism, not a performance measurement.
* **The long run is not a healthy campaign.** Its experiments failed throughout
  ("completed without writing metrics"; `smoke_gate timed out after 120s`), and
  it was kept alive by raising `max_barren_steps` purely to buy the producer
  wall clock. That is a fair test of *"does the queue refill while the consumer
  keeps working"* and a poor one of anything about research quality.

## 4. Scope

**In:** `gather_once` as a callable unit; a thread runner; the allowlist change;
store-level dedupe across check-then-create; the `claim_if_proposed` migration;
a producer-side quota reserve; producer state in observe.

**Out:** a separate long-lived process (`research gather --watch`) — deferred,
not rejected (§5.2); any change to *what* gathering does; hypothesis **quality**
— the plan's fourth trap is M21's viability filter plus the stagnant clause,
both shipped; and **removing the Kaggle coupling that already exists** in
`ANALYZE_ARGS` and the `analyze_competition` tool name, which is
[M12](../06-beyond-kaggle.md)'s job. This milestone's obligation is not to
deepen it (§5.4).

## 5. Design

```
campaign thread (consumer)              producer thread
  step 1  claim → plan → run              tick: should_gather_evidence()?
  step 2  claim → plan → run                └─ no: log reason, wait
  step 3  claim → plan → run              tick: yes
  step 4  claim → plan → run                ├─ analyzers
  step 5  claim → plan → run                ├─ fetch → ingest
    ▲                                       ├─ hypothesize ──┐
    │                                       └─ brief         │
    └───────────── shared HypothesisStore ◄──────────────────┘
                   (flock per id; .alloc.lock for create)
```

### 5.1 The unit

```python
# conductor/producer.py
@dataclass(frozen=True)
class GatherPlan:
    tool: str                     # which gathering tool to invoke
    args: dict[str, Any]          # what to ask it for

@dataclass(frozen=True)
class GatherOutcome:
    gathered: bool
    reason: str                   # the gate's reason, verbatim, either way
    hypotheses_created: int = 0
    duration_s: float = 0.0

def gather_once(workspace, registry, plan: GatherPlan, *,
                llm_client=None, budgets=None) -> GatherOutcome: ...
```

**The plan is an argument, not a constant** (§5.4). `gather_once` evaluates the
gate and invokes `plan.tool` through the registry with `plan.args`; it contains
no source names, no fetch plans and no competition vocabulary. Resolution lives
in one function beside it — `default_gather_plan(workspace)`, returning today's
`("analyze_competition", ANALYZE_ARGS)` — which is the only place in this
milestone that may name a source.

Through the registry, not straight to `AnalyzeOrchestrator`, so the
`verify_ai_artifact` gate and the report write stay in the path; `verify_auto`
defaults to `True`, so nothing prompts on a background thread.

`llm_client` is **required, not optional**. `Scheduler._with_llm_client` injects
it by signature and the producer does not go through the Scheduler — that
omission is the mechanism behind twelve identical MSE 194.80 runs.

### 5.2 The runner

| Option | Verdict |
|---|---|
| Blinker subscriber on `ExperimentCompleted` | **Rejected.** `EventBus.publish` is a synchronous `signal.send` — the handler runs on the publisher's thread, blocking the consumer at exactly the moment this milestone exists to unblock it |
| `anyio` task beside the campaign | **Rejected.** `_run_until_stop_inner` is synchronous throughout; an event loop for one background task buys nothing a thread does not |
| Separate OS process | **Deferred.** Correct eventually; outliving the campaign means orphan detection, its own config and LLM resolution, and sweeps against a workspace nobody is using |
| **Daemon thread in `_run_until_stop_inner`** | **Chosen.** Workspace, registry, `llm_client` and budgets are already resolved there; it dies with the campaign; M11 made every store it touches safe for a second thread |

Ticks on `LABPILOT_GATHER_TICK_S` (default 300), waiting on a `threading.Event`
so idle shutdown is immediate. Each tick opens its own `ConductorStore`, re-reads
the session, `load_budget_pair`s it, resolves the `GatherPlan`, and calls
`gather_once` — keeping the unit a pure gate-then-pipeline a test can call with a
hand-built pair and a fake plan.

The interval is a **rate limit, not a cadence assumption**. A domain whose
evidence moves weekly rather than hourly does not need a different runner; it
gets more no-ops, and a no-op costs < 50ms.

### 5.3 The consumer loses the tool

`available_tools` sets `analyze_competition: False` unconditionally while the
producer runs — the shape `search_papers` already has. Leaving it gated on the
predicate lets both components pass the same gate in the same second and sweep
twice. It also makes criterion 1 checkable: the tool is not on the table, so
"never blocks on gathering" is a property of the allowlist rather than a hope
about scheduling. The gate is unchanged; it moves from deciding the consumer's
allowlist to deciding the producer's tick.

### 5.4 Domain coupling: what this milestone may and may not assume

[M12](../06-beyond-kaggle.md) is blocked by exactly one thing going wrong here —
"Kaggle assumptions already leaking" into the control plane. The producer sits
*in* the control plane, so every element is audited:

| Element | Domain-coupled? | Why |
|---|---|---|
| `should_gather_evidence` | **No** | Reads `MAX(created_at)` from `research_artifacts`, a viable-hypothesis count, and a score series. None of the three knows what a Kaggle kernel is |
| `GatherPlan` | **No, by construction** | The one field that could name a source is data passed in |
| `default_gather_plan(workspace)` | **Yes, deliberately** | The single quarantined site. M12 changes this function and nothing else in `producer.py` |
| Tick interval, reserve, thresholds | **No** | Knobs with defaults, not branches on domain |
| `create_unless_covered(covered_by=…)` | **No** | The duplicate predicate is caller-supplied, so a domain-specific notion of "same idea" never lands in the store |
| `claim_if_proposed`, the runner, shutdown | **No** | Concurrency mechanism |
| Tool name `analyze_competition`, `workspace.competition` | **Inherited** | Repo-wide naming that predates this milestone. The producer names the tool once, through `GatherPlan`. It does not deepen the coupling and does not fix it |

**The failure this prevents is concrete.** `ANALYZE_ARGS` — the campaign's
existing gathering args — is Kaggle-shaped twice over: `fetch_kaggle=True` with
`kaggle_fetch_plan="best_score"`, and `exclude=["papers"]`, whose stated
rationale is *"on a Kaggle competition the kernels are the better-grounded source
anyway: they ran against this dataset."* Off Kaggle that rationale inverts —
papers and repositories may be the only sources there are.

Had the producer hardcoded those args, a non-Kaggle workspace would get: a
Kaggle fetch that soft-fails to a log warning (`_fetch_kaggle_run` catches and
notes), papers excluded, and therefore near-zero new evidence — while the gate,
seeing a pool that never fills, keeps answering *gather*. The result is a sweep
every `_MIN_RESWEEP_HOURS` (30 min), forever, producing nothing. Sequentially
that is one bad step; as a background producer it is a permanent one, and this
milestone is precisely what turns the first into the second.

Not in scope, per M12's own trap ("do not build a plugin system first"): a
source-provider registry, or making `analyze_competition` domain-neutral. One
dataclass and one resolver is the whole abstraction, and it is there so the
producer is not the thing standing in M12's way.

## 6. Components

| Component | Responsibility | Change |
|---|---|---|
| `conductor/producer.py` | `gather_once` + thread runner + `default_gather_plan` (the one quarantined domain site) | **new** |
| `conductor/policy.py` | gate unchanged; `available_tools` gains the producer-owned case | edit |
| `conductor/loop.py` | start/stop the runner | edit |
| `shared/experiments/hypothesis.py` | `create_unless_covered` | **new method** |
| `intelligence/hypothesis/persist.py` | route through it | edit |
| `execution/outcome.py` | route four `maybe_mint_*` through it; drop the unlocked pre-check | edit |
| `reflection/hypotheses/evaluator.py` | docstring: why this is a status marker and not a claim | edit |
| `fitroute/budget.py` | `availability(..., reserve=)` | edit |
| `cli/conduct.py` | `--gather-background` | edit |

## 7. Implementation details

### 7.1 Nothing live is shared across the thread boundary

`SqliteClient` opens `check_same_thread=True` and `ConductorStore` does not opt
out — its docstring records the cost of opening one instead: **~1.7ms warm**. The
producer opens and closes its own handles per tick. Any patch handing it the
campaign's `store` is the bug.

The same applies to `budgets`. `BudgetState` is mutated by the campaign on every
recorded experiment, and the producer's stagnant clause reads its score series;
sharing the instance yields a verdict assembled from two campaign states — the
defect `availability`'s own docstring records having had once. The producer
re-loads the pair each tick. One tick of staleness is acceptable: "has this
campaign stopped improving" does not change meaningfully inside 5 minutes.

The three SQLite stores it touches are WAL with a 5s `busy_timeout`;
multi-statement writes take `write_lock_for`, a **file** lock that keeps holding
if the producer ever becomes a process.

### 7.2 Dedupe: one lock across check-and-create

```python
def create_unless_covered(self, *, covered_by, **fields) -> Hypothesis | None:
    """Create, unless `covered_by(existing_proposed)` says one already is.

    The predicate runs inside `.alloc.lock`, so a concurrent writer cannot
    slip a near-duplicate in between the check and the write.
    """
```

`covered_by` comes from the caller — token overlap for the outcome mints, a
stricter identity for the producer's recommendations — so this method arbitrates
*when* the check runs, not *what counts as a duplicate*.

**`.alloc.lock` is not reentrant.** `locked()` is `fcntl.flock(LOCK_EX)`, and a
second acquisition on a new descriptor for the same file blocks, including from
the same thread. So `covered_by` must not call `create()`, and the implementation
cannot be "take the lock, then call the existing `create`" — the
allocate-and-write body has to be factored out and shared.

### 7.3 The claim (no change, and why)

`claim_if_proposed` already exists and `fanout.py` already uses it. The design's
first draft called for migrating `evaluator.mark_testing` to it as well; reading
the call path retired that.

The producer **proposes**; it never claims. So M16 introduces no second claimer,
and the only concurrent claiming remains M11's fan-out, which claims in
`prepare_branches` before the branch reaches the engineer. `mark_testing` then
runs against a hypothesis that is already `testing` and legitimately owned — an
exclusive claim there would answer `None` on every branch of a healthy fan-out
and mean nothing by it.

What was actually missing was the rule written down. `mark_testing` now says it
is a status marker, that `None` means "no such hypothesis" and never "someone
else has it", and that a caller acting on *ownership* wants
`HypothesisStore.claim_if_proposed` with `fanout.py` as the worked example.

Exit criterion 4 is unchanged and still worth its test — it pins the property,
not the migration.

### 7.4 Shutdown

The campaign sets the stop event and joins with a bounded timeout. A tick
mid-sweep checks the event at each pipeline stage boundary —
`apply_side_effects` already runs four labelled steps through an `on_progress`
hook — and returns early. Each stage writes as it completes, so an abandoned tick
loses work, not consistency. Past the timeout the thread is daemon and the
process exits.

## 8. Tradeoffs

| Decision | Chosen | Alternative | Cost of the choice |
|---|---|---|---|
| Runner lifetime | Thread owned by the campaign | Long-lived process | No gathering between campaigns — the cadence a producer ideally wants. Accepted: a producer with no consumer grows a store for nobody |
| Trigger | Timer | Bus event | Up to one tick of latency before a drained pool is noticed. The bus is synchronous, so event-driven costs the consumer the very block being removed |
| Gate | Reuse `should_gather_evidence` unchanged | Producer-specific policy | Inherits its blind spots exactly. Deliberate: one predicate, one place, already tested |
| Budget priority | `reserve=<fraction>` on `availability()` | Priority queue in `fitroute` | Coarse — does not distinguish a cheap producer call from an expensive one. The queue is the better design and needs a scheduler and fairness policy to serve one background caller; the reserve is four lines for the same intent |
| Dedupe | In the store, under `.alloc.lock` | Per-caller checks | Every minting caller pays a `list(PROPOSED)` scan under a lock — milliseconds, against a sweep measured in minutes |

The reserve, concretely: `availability(provider, *, rpm, rpd, tpm,
reserve=0.0)` withholds a **fraction** of each configured window — request or
token — from the caller. A fraction rather than a call count because `rpm`/`rpd`
are free-tier shapes; a paid, token-metered provider binds on `tpm`, where "5
calls" means nothing. The consumer passes 0 and sees the real limit; the producer
passes `LABPILOT_GATHER_RESERVE` (default `0.2`) and runs out first, by that
margin, against whichever window binds. The ledger already holds its own `RLock`
across `availability`'s read-compare and opens `check_same_thread=False`, so a
producer thread needs no new synchronisation.

## 9. Observability

Three signals, all of which must exist before the paired campaign run in §3 is
worth doing:

| Signal | Where | Why |
|---|---|---|
| Per-tick line: decision, gate reason, duration, rows created | log | Criterion 2 is literally "no-ops with a logged reason" |
| `evidence_producer` — last tick, last decision, last reason | observe bundle | Otherwise the policy watches the pool grow between steps with no account of why |
| Steps per hour | campaign log | The only number that settles criterion 1 |

`evidence_producer` is deliberately *description*, not control. The gate decides;
a number in the prompt has changed no decision in nine campaigns and this one is
not expected to either.

## 10. Testing

| # | Scenario | Passes only if |
|---|---|---|
| 1 | Producer on, campaign runs N steps | The gathering tool (`analyze_competition` today) is absent from every step's allowlist |
| 2 | Full pool, 3 ticks | Three logged no-ops naming the gate's reason; zero `research_artifacts` rows |
| 3 | Thin pool, producer ticks | Pool refills; consumer's step count over the same wall-clock matches the producer-off run |
| 4 | Producer and consumer race one `proposed` hypothesis | Exactly one `claim_if_proposed` returns non-`None` |
| 5 | Producer and consumer mint the same idea concurrently | One row. **Must fail before §7.2 lands** — if it passes against today's `create`, the fixture is not expressing the race and the test is worthless |
| 6 | Producer tick raises | Campaign completes its remaining steps; exception in the log |
| 7 | `gather_once` driven with a `GatherPlan` naming a stub tool and empty args | It gates, invokes, and reports normally. **This is the check that keeps §5.4 true** — it fails the moment someone reaches for `ANALYZE_ARGS` inside the producer |

Write #5 first and watch it fail. The rest of this design is mechanism whose
absence is visible; the duplicate is mechanism whose absence looks like success.

Concurrency tests use the existing fake clock and the **real** `flock` — a
`threading.Lock` substitute passes while proving nothing, the mistake
`file_lock.py`'s docstring already records.

## 11. Rollout

Off by default. `research conduct run --gather-background` (or
`LABPILOT_GATHER_BACKGROUND=1`) starts the producer; without it the loop is
byte-for-byte today's behaviour, including `analyze_competition` staying gated on
the predicate in the consumer's allowlist. **Rollback is dropping the flag.**

Ship order, each step independently useful — **all four landed**:

1. ~~§7.2 dedupe~~ — correctness under two writers, no producer yet (§7.3 turned
   out to be documentation, not a change).
2. ~~§5.1 `gather_once`~~ — callable, invoked from nowhere.
3. ~~§5.2 runner + §5.3 allowlist + §9 observability~~, behind the flag.
4. ~~§8 reserve~~ — `availability(..., reserve=)`, `select_route(..., reserve=)`,
   `LLMGateway.preview(role, reserve=)`, and a pre-flight check in the producer.

What the tests cover, and what they do not:

| Criterion | State |
|---|---|
| 1 — a step never blocks on gathering | **Unit-proven at the allowlist**: with a producer running, `analyze_competition` leaves the consumer's allowlist even when the gate says *gather*. Not yet shown on a campaign log |
| 2 — full backlog ticks and no-ops with a reason | **Covered** |
| 3 — a thin backlog refills without the consumer stalling | **Met, measured 2026-08-20** (§3.1). 18 hypotheses minted across two completed sweeps while the consumer dispatched 18 steps and never waited. Needed `max_barren_steps` raised for the campaign to outlast a ~20-minute sweep |
| 4 — producer and consumer never claim the same hypothesis | **Covered**, and structurally: the producer proposes and never claims (§7.3) |
| 5 — one idea, one row, under two writers | **Covered and mutation-checked** — moving the predicate outside `.alloc.lock` makes eight racing writers produce eight rows, and the test catches it |
| 6 — a tick that raises does not take the campaign with it | **Covered** |
| 7 — the plan decides what is gathered, not the producer | **Covered** — driven with a stub tool and non-Kaggle args |

All five exit criteria are now met (§3.1). Two things the runs surfaced are
**not** fixed and should be booked before this is called done:

1. **The producer cannot satisfy its own gate** (§3.1). Its output does not move
   `viable_hypothesis_count`, so it re-sweeps indefinitely — ~40 minutes of
   reasoning-role LLM work in a 51-minute campaign. `_MIN_RESWEEP_HOURS` is
   currently the only thing standing between that and a hot loop, which makes
   the "rate limit, not a cadence assumption" framing in §5.2 too casual.
2. **A sweep costs ~20 minutes**, most of it `RepositoryAnalyzerAgent`. Any
   campaign shorter than that gets fresh evidence and no new hypotheses — a
   perfectly reasonable outcome the plan never names.

**It does not fix what a campaign is short of.** The plan's first trap stands,
now aimed at M22–M26: a faster supply of hypotheses tested against a target
nobody verified is a faster way to learn nothing. This makes the supply cheap,
not worth having.

**Handoff to M12.** One function — `default_gather_plan(workspace)` — is
everything that must change for a non-Kaggle domain to gather. If a second
domain arrives and any *other* file in this milestone needs editing, §5.4 was
wrong and the review that let it through is the thing to fix.
