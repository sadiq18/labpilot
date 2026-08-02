# M16 — Evidence routine as a background producer

**Status:** gating shipped, routine not started · **Blocked by:** M11
(concurrency), M14 (a trustworthy LLM path)

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

Evidence gathering is now gated on **two independent brakes**, both of which
must pass. Either alone lets the store bloat:

```python
def should_gather_evidence(workspace) -> tuple[bool, str]:
    if untested_hypothesis_count(workspace) >= _HYPOTHESIS_BACKLOG_TARGET:
        return False, f"{backlog} untested hypotheses already queued"
    if hours_since_last_artifact(workspace) < _EVIDENCE_COOLDOWN_HOURS:
        return False, f"evidence gathered {age:.1f}h ago (cooldown)"
    return True, ...
```

`analyze_competition` and `search_papers` are removed from the policy's
allowlist when this returns False, so the tool is never even offered. The skip
reason is logged and both signals (`untested_hypotheses`,
`hours_since_last_artifact`) appear in the observe bundle so the policy reasons
*with* the constraint rather than against it.

Both thresholds are configurable:

| Env var | Default | Meaning |
|---|---|---|
| `LABPILOT_HYPOTHESIS_BACKLOG_TARGET` | `3` | Queue depth above which gathering stops |
| `LABPILOT_EVIDENCE_COOLDOWN_HOURS` | `6.0` | Minimum age of newest artifact before refetching |

Verified live:

```
Skipping evidence gathering: 12 untested hypotheses already queued
step 1/14: chose generate_plan
```

— the ~15-minute sweep skipped, straight to testing queued work.

**Tuning note.** The two brakes are AND-ed, so the backlog does most of the
work and the cooldown is a floor against thrashing. Shorter cooldown = fresher
evidence, larger store, more tokens; longer = leaner, staler. Default is 6h,
which keeps the artifact store manageable across long campaigns. Lower it (e.g.
`LABPILOT_EVIDENCE_COOLDOWN_HOURS=1`) when a competition's discussions are
moving fast and staleness costs more than storage.

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

1. **Extract the gathering pipeline** into a routine callable independently of
   the campaign loop — `analyze → ingest → hypothesize` already exists as
   `apply_side_effects`, so this is mostly a scheduling change.
2. **Run it on a timer or as a spawned worker**, re-evaluating
   `should_gather_evidence()` each tick and no-op'ing when the condition fails.
   The predicate is already the whole decision.
3. **Claim hypotheses atomically.** `mark_testing_if_proposed` exists and must
   become the claim mechanism so producer and consumer cannot collide.
4. **Feed reflection back into the producer.** The consumer's reflections become
   hypothesis candidates alongside the mined ones — this is the edge from
   [M8](02-objective-loop.md) and the only path where the system learns from
   itself.

## Exit criteria

1. A campaign step never blocks on evidence gathering.
2. With a full backlog, the producer ticks and no-ops with a logged reason.
3. With an empty backlog, the queue refills without the consumer stalling.
4. Producer and consumer never claim the same hypothesis.

## Traps

- **Do not build this before [M7](01-technique-to-model.md).** A faster supply of
  hypotheses that all produce MSE 194.80 is a faster way to learn nothing.
- **Shared SQLite writers.** `knowledge.db`, the hypothesis store and the
  conductor store are all SQLite; two writers need WAL and a serialised write
  path (same constraint as [M11](05-parallel-branches.md)).
- **The producer competes for the same LLM budget.** Hypothesis generation is a
  `reasoning` role — a background producer running hot will exhaust the free
  tier the consumer needs. Producer work should be the *lower* priority claim on
  the ledger from [M10](04-llm-tiering.md).
- **A backlog is not a good backlog.** The current condition counts hypotheses;
  it cannot tell three strong ideas from three weak ones. Once reflection
  outcomes exist, "the last N tested all came back inconclusive" should reopen
  gathering regardless of count.

## Related code

- `src/labpilot/research_engine/conductor/policy.py` — `should_gather_evidence`, `untested_hypothesis_count`, `hours_since_last_artifact`, `available_tools`
- `src/labpilot/research_engine/intelligence/orchestrator.py` — `apply_side_effects` (the gathering pipeline)
- `src/labpilot/research_engine/shared/experiments/hypothesis.py` — `mark_testing_if_proposed` (the claim primitive)
- `src/labpilot/research_engine/agents/parallel.py` — worker primitives
