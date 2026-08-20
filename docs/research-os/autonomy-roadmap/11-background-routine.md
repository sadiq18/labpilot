# M16 — Evidence routine as a background producer

**Status:** routine shipped behind `--gather-background`; campaign runs
2026-08-20 — all exit criteria met, two follow-ups open ·
**Design:** [design/11-background-routine.md](design/11-background-routine.md) ·
**Blockers cleared:** M11 (concurrency) shipped 2026-08-11/12 and M14 completed
2026-08-07, so this is unblocked and waiting on a decision, not on other work

---

## Purpose

Evidence gathering and hypothesis testing have opposite cost profiles and
opposite cadences, but today they share one sequential loop:

| | Gathering | Testing |
|---|---|---|
| Cost | minutes of network + LLM | minutes of compute |
| Cadence | occasional — evidence goes stale slowly | continuous — the bottleneck |
| Blocks the other? | **yes, today** | yes |

Observed: a campaign spent ~15 minutes re-sweeping kernels, discussions, papers
and repositories while **ten untested hypotheses sat idle**. That is the most
expensive possible way to make no progress.

## What already ships (the skip condition)

Evidence gathering is gated on **three independent clauses under one floor**.
Any single clause is sufficient to gather; the floor is a rate limit, not a
fourth condition:

```python
def should_gather_evidence(workspace, budgets=None) -> tuple[bool, str]:
    age_hours = hours_since_last_artifact(workspace)
    if age_hours is not None and age_hours < _MIN_RESWEEP_HOURS:
        return False, f"evidence gathered {age_hours * 60:.0f} minutes ago"   # floor
    if viable_hypothesis_count(...) < _VIABLE_TARGET:
        return True, f"only {viable} viable hypotheses queued"                # thin
    if steps_since_improvement >= plateau_window:
        return True, f"{stagnant_for} experiments with no improvement"        # stagnant
    if age_hours is None or age_hours >= _EVIDENCE_COOLDOWN_HOURS:
        return True, f"evidence is {age_hours:.1f}h old"                      # stale
    return False, f"{viable} viable hypotheses queued and evidence gathered ..."
```

`analyze_competition` is removed from the policy's allowlist when this returns
False, so the tool is never even offered. `search_papers` is **not** gated on
this predicate — it is hard-`False` in `available_tools` for an unrelated
reason: every conductor path forces `offline=True`, so a campaign's
`search_papers` writes `count: 0` and returns, and literature is reached
through `analyze_competition` instead.

The skip reason is logged and the signals (`viable_hypotheses`,
`untested_hypotheses`, `hours_since_last_artifact`) appear in the observe bundle
so the policy reasons *with* the constraint rather than against it.

The clauses are **independent** — any one is sufficient to gather. The first
two were originally ANDed, which made each a veto on the other: with 46 proposed
hypotheses queued the staleness clause was never evaluated at all and gathering
was disabled permanently, so the pool blocked the only thing that could refresh
it. The floor sits under all three as a rate limit, not as another gate — making
the clauses independent introduced the opposite failure, a pool that stays thin
sweeping on every step.

| Env var | Default | Meaning |
|---|---|---|
| `LABPILOT_VIABLE_HYPOTHESIS_TARGET` | `5` | Gather while fewer than this many *viable* hypotheses are queued |
| `LABPILOT_EVIDENCE_COOLDOWN_HOURS` | `24.0` | Gather once the newest artifact is at least this old |
| `LABPILOT_MIN_RESWEEP_HOURS` | `0.5` | Floor under all three: never re-sweep sooner than this |
| — (`plateau_window`, `--plateau-window`) | `3` | Stagnant clause: gather once this many experiments pass with no improvement. Shares the campaign's window rather than adding a knob |

Viable, not merely queued: a hypothesis the selector has passed over twice
stops voting on whether the campaign may look for something better. See
[`intelligence/hypothesis/viability.py`](../../../src/labpilot/research_engine/intelligence/hypothesis/viability.py).

`LABPILOT_HYPOTHESIS_BACKLOG_TARGET` was the name of the first of these and is
**no longer read anywhere**. Setting it has no effect.

Verified live:

```
Skipping evidence gathering: 12 untested hypotheses already queued
step 1/14: chose generate_plan
```

— the ~15-minute sweep skipped, straight to testing queued work. That line was
recorded before the viability change; the reason string now names the *viable*
count (`3 viable hypotheses queued and evidence gathered 2.1h ago`), and the
stagnant clause can reopen gathering with any number of them queued.

**Tuning note.** The clauses are OR-ed, so the cooldown is a *ceiling on
staleness* rather than a brake: it guarantees a sweep eventually, no matter how
large the pool grows. The pool size and the plateau do the day-to-day work.
Shorter cooldown = fresher evidence, larger store, more tokens; longer = leaner,
staler. Default is 24h, which keeps the artifact store manageable across long
campaigns. Lower it (e.g. `LABPILOT_EVIDENCE_COOLDOWN_HOURS=1`) when a
competition's discussions are moving fast and staleness costs more than storage.

## Goal

Gathering runs as a **background producer** on its own cadence, never blocking
the tester.

```
producer (routine)                    consumer (campaign)
  ├─ should_gather_evidence()?          ├─ claim next hypothesis
  ├─ fetch kernels / papers / repos     ├─ plan against it
  ├─ ingest → concepts → techniques     ├─ run experiment
  └─ propose hypotheses ────────────────┤  reflect
        ▲                               └─ reflection → new hypothesis ──┐
        └───────────────────────────────────────────────────────────────┘
```

The consumer never waits for the producer; it works the queue. The producer
tops the queue up when the skip condition allows.

## Approach

Full mechanism in [design/11-background-routine.md](design/11-background-routine.md).

1. **Extract the gathering pipeline** into a routine callable independently of
   the campaign loop. `apply_side_effects` is the *second* half (fetch → ingest
   → hypothesize → brief); the analyzers run in `analyze_without_side_effects`
   and the `verify_ai_artifact` gate sits between them, so the unit to extract
   is the `analyze_competition` handler, not `apply_side_effects` alone. The
   gate is likewise not callable on its own today — it lives inside
   `available_tools`, which returns a set of tool names.
2. **Run it on a timer or as a spawned worker**, re-evaluating
   `should_gather_evidence()` each tick and no-op'ing when the condition fails.
   The predicate is already the whole decision. A bus subscriber is *not* an
   option: `EventBus.publish` is a synchronous `signal.send`, so the handler
   runs on the publisher's thread and would block the consumer at exactly the
   moment this milestone exists to unblock it.
3. **Claim hypotheses atomically.** Already shipped with M11:
   `claim_if_proposed` returns `None` to the loser, backed by a cross-process
   `fcntl.flock` per hypothesis id, and `fanout.py` uses it. The remaining
   caller of `mark_testing_if_proposed` — `reflection/hypotheses/evaluator.py`
   — is correct as it stands: the producer proposes and never claims, and that
   caller usually runs inside a claim fan-out already made, where an exclusive
   claim would report "lost" on every healthy branch. It now says so.
   **Claiming is not the only race, and the other one is open.** Neither minting path dedupes safely under
   two writers: `persist_recommendations` (the producer's own output) does not
   dedupe at all, and `_already_covered_by_proposed` lists the proposed pool
   *outside* the lock `create()` takes. Two writers produce two rows for one
   idea. Latent today because there is one writer.
4. **Feed reflection back into the producer.** Largely built: the consumer
   already mints from its own outcomes at five sites (`maybe_mint_improvement_
   hypothesis`, `maybe_mint_stacked_from_success`,
   `maybe_mint_ablation_from_combo_win`, `maybe_mint_combo_from_success`,
   `mint_stagnation_hypothesis`) into the same store the producer writes to —
   the edge from [M8](02-objective-loop.md) exists. What remains is the coupling
   it creates: those mints raise the viable count, which is the producer's own
   brake, so a campaign can hold gathering shut with hypotheses it generated
   itself. That is the 46-row ratchet in a new costume, and the reason the
   stagnant clause must stay independent of the count.

## Exit criteria

1. A campaign step never blocks on evidence gathering.
2. With a full backlog, the producer ticks and no-ops with a logged reason.
3. With an empty backlog, the queue refills without the consumer stalling.
4. Producer and consumer never claim the same hypothesis.
5. *(added by the design, §2.5 / §10.)* Producer and consumer minting the same idea
   concurrently produce **one** row. Claiming was the only race this plan
   named; creating is the other one.

**Measured 2026-08-20** on sandbox clones of rogii (design §3, §3.1). The
consumer never stalled and evidence went from **158h stale to 0.01h**, against
**+0 and 158h** on the baseline. A sweep takes **~20 minutes**, so the first two
attempts ended before it finished; a 51-minute run produced **two complete
sweeps, 18 hypotheses minted, 18 consumer steps, and zero `analyze_competition`
dispatched by the campaign**. Criterion 3 met.

**What bounds the campaign is `max_barren_steps`, not `--max-steps`.** Both
short runs died on *"8 step(s) since the last success"* — on a workspace whose
experiments fail, that fires long before anything else, and it has no CLI flag.

**The producer could not satisfy the gate it gates on** — now fixed (design §7.5). Ten hypotheses
minted, and the next tick still read *"only 10 viable hypotheses queued"*:
`viable_hypothesis_count` excludes rows the selector has passed over twice, and
every `generate_plan` ages every row it did not pick, so the producer's output
aged out as fast as it arrived. It swept continuously — ~40 minutes of
reasoning-role LLM work in a 51-minute campaign — with `_MIN_RESWEEP_HOURS` the
only brake. Not the M21 ratchet (nothing is held shut) but the same shape
inverted. The producer now latches that clause when a completed sweep leaves the
count where it found it, and lifts the latch when the count moves.

**Left open, deliberately:** at a 153-row pool with one pick per selection,
"passed over twice" is arithmetic, not a quality signal — 8 of 18 fresh rows
were stale within one campaign. Whether `viable_hypothesis_count` is the right
gate for *anyone* is a bigger question than M16: it decides the consumer's
allowlist too, and loosening it is how [M21](16-hypothesis-selection.md)'s
ratchet re-opens.

**Criterion 1's premise did not reproduce, and that matters more than the
result.** The baseline never blocked on gathering. The gate reported
*"Evidence gathering available: only 10 viable hypotheses queued"* at all five
steps and the LLM policy chose testing every single time. So **steps per hour
does not discriminate here** — the failure this milestone actually removes is
not a campaign stalled behind a sweep, it is a campaign testing hard against
five-day-old evidence while the gate says "go and look" and the policy
correctly refuses because it has work to do. Nothing else in the system
resolves that standoff.

## Traps

- **Do not build this before [M7](01-technique-to-model.md).** ~~Blocking~~ —
  M7 done 2026-08-07. The *reason* stands and is now aimed at
  [M22–M26](README.md): a faster supply of hypotheses tested against a target
  nobody verified is a faster way to learn nothing. This milestone makes the
  supply cheap; it does not make it worth having.
- ~~**Shared SQLite writers.**~~ **Cleared by M11.** `knowledge.db`, the
  hypothesis DB mirror and the conductor store all open through `SqliteClient`
  (WAL + 5s `busy_timeout`), multi-statement writes take `write_lock_for`, and
  both are file locks that keep holding across processes. What M11 did *not*
  give is content-level dedupe — see Approach 3.
- ~~**The producer competes for the same LLM budget.**~~ **Built.**
  `availability(..., reserve=)` withholds a fraction of each window from the
  caller that asks for it — a fraction, not a call count, so a token-metered
  provider is covered too. It threads through `select_route` and
  `LLMGateway.preview`, and the producer asks before a sweep rather than
  per call inside one: a sweep abandoned halfway has already spent the quota it
  was meant to protect. `LABPILOT_GATHER_RESERVE` defaults to `0.2`.
- **A backlog is not a good backlog.** **Partly answered.** The count is now of
  *viable* hypotheses ([M21](16-hypothesis-selection.md)), so rows the selector
  has passed over twice stop voting, and the stagnant clause reopens gathering
  on "N experiments with no improvement" regardless of pool size — the
  score-based reopen this trap asked for. Still missing the outcome-shaped
  version: "the last N tested all came back **inconclusive**" is a different
  signal from "no improvement", and nothing reads it.

## Related code

- `src/labpilot/research_engine/conductor/policy.py` — `should_gather_evidence`, `untested_hypothesis_count`, `hours_since_last_artifact`, `available_tools`
- `src/labpilot/research_engine/intelligence/orchestrator.py` — `analyze_without_side_effects` (the analyzers) and `apply_side_effects` (fetch → ingest → hypothesize → brief); the two halves of the pipeline
- `src/labpilot/research_engine/tools/handlers/analyze.py` — the `analyze_competition` handler, which is both halves plus the verify gate
- `src/labpilot/research_engine/shared/experiments/hypothesis.py` — `claim_if_proposed` / `release_claim` (the claim primitives; `mark_testing_if_proposed` cannot tell winner from loser), `create` and its `.alloc.lock`
- `src/labpilot/research_engine/agents/parallel.py` — worker primitives
- `src/labpilot/research_engine/intelligence/hypothesis/persist.py` — `persist_recommendations`, the producer's write path (no dedupe)
- `src/labpilot/research_engine/execution/outcome.py` — the four `maybe_mint_*` functions and `_already_covered_by_proposed`
- `src/labpilot/accessor/sqlite/client.py` — WAL, `busy_timeout`, `write_lock_for`
- `src/fitroute/budget.py` — `BudgetLedger.availability`, where producer priority would go
