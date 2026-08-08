# M21 — A hypothesis earns its turn

**Status:** in progress · **Blocks:** [M19](14-experiments-as-deltas.md) step 2 ·
**Completes:** M16's known gap (*"a backlog is not a good backlog"*)

---

## Purpose

Four campaigns ran with `codegen.strategy: delta` on 2026-08-09. **None produced
a delta experiment**, and the editor was right to refuse every time:

> *"The code currently: 1 Trains a LightGBM model … 3 Averages predictions from
> both models … Since no modifications are required, there are no
> SEARCH/REPLACE blocks to output."*

The hypothesis asked for an ensemble that `train.py` already had. Nothing marks
a hypothesis as already implemented, so it stayed `proposed` and was selected
again on the next step, and the next.

That is not a codegen defect. It is the selector choosing work already done.

## The ratchet

The failure compounds, which is what makes it worth a milestone rather than a
patch.

```
46 proposed hypotheses
   → should_gather_evidence() shuts at backlog >= 3
      → analyze_competition and search_papers leave the allowlist
         → no new evidence, no better hypotheses
            → the 46 stay, and the gate stays shut
```

Measured on rogii: **46 `proposed`**, of which 43 were generated 2026-08-07 and
never tested, including `3D garment modeling` and `Breath Focus practice` for a
wellbore-geology regression. The only thing that would refresh the pool is
disabled *by* the pool.

M16 already named this and left it: *"The condition counts hypotheses; it cannot
tell three strong ideas from three weak ones."* It is now load-bearing, because
M19 step 2 needs a measured delta failure rate and cannot get one.

## What is actually wrong

Four distinct defects, discovered together and worth separating:

| # | Defect | Consequence |
|---|---|---|
| 1 | A redundant hypothesis is never marked | Re-selected forever; `aider_no_edit` counted as an adapter failure |
| 2 | A failed hypothesis has no terminal/retryable distinction | A 429 and a dead end are recorded identically |
| 3 | The backlog gate counts *rows*, not *viable* ideas | 46 junk entries hold the gate shut as firmly as 46 good ones |
| 4 | `confidence` is a prior that evidence never updates | `hyp:H-010` sat at 0.99 after the runs that disproved it |

Selection already sorts by `(confidence, id)` — it is not picking by numbering,
as it first appears. The weakness is #4: nothing turns a prior into a posterior,
so ranking is stable in exactly the way it should not be.

`RankingConfig` — `expected_gain`, `implementation_cost`, `risk`, `novelty` —
exists in config, is used by candidate generation, and **is ignored by the
conductor's actual pick.** Two rankers, and the one that runs is the simpler.

## The insight

**The critic that matters most is not a model.**

"Is this already implemented?" is a mechanical question, and 1c already built
both halves of the answer: `DeltaBriefAgent` emits `added` / `kept` / `combined`
as *code identifiers*, and `consistency.py` parses the parent's AST. If every
symbol in `added` already appears in the parent, the hypothesis is redundant —
deterministically, for free, and with no judgement to be wrong about.

That is `check_addition` run **before** the experiment instead of after. It
would have ended the P-021 loop on its first step.

An LLM critic earns its place afterwards, on the softer question — *is this
promising given what we know* — where mechanism runs out. One condition: **a
score must cite its evidence**, the card or the symbol that justifies it, never
a bare number. A model ranking hypotheses is exactly where "plausible but wrong"
lives, and this project has paid for that: `vit` was `confirmed` at 0.99
confidence on a tabular regression.

## The steps

| step | what | state |
|---|---|---|
| 1 | Mechanical redundancy check, before the experiment | in progress |
| 2 | Terminal vs retryable hypothesis outcomes, reusing `failure_kind` | in progress |
| 3 | Viability-aware backlog + fetch on `< 10` **or** `> 24h` | in progress |
| 4 | `confidence` updated from evidence — a posterior, not a prior | in progress |
| 5 | LLM critic + ranking, *if* 1–4 have not already fixed selection | not started |

Ordered so each is measurable on its own. The expectation is that 1–4 resolve
most of it and step 5 turns out to be a narrower question than it looks today —
which is itself the reason not to build the critic first.

**Fusion is deliberately absent.** Combining hypotheses multiplies the surface
before a single experiment runs end to end, and the roadmap already names that
trap: *"M5 shipped parallel agents before the sequential loop could run a single
real experiment. Breadth before depth is exactly how a beautiful control plane
ended up driving a one-motion data plane."* Revisit when a campaign reliably
produces evidence cards.

## Retryable is the same rule the LLM path already follows

`BaseMicroAgent.run` retries transient failures and records a failure **only
when attempts are exhausted** — a call that succeeds on attempt 2 is a success,
not a failure with a caveat. Hypotheses need the same distinction one layer up,
and the vocabulary already exists: `failure_kind` separates transient
(`rate_limit`, `unavailable`, `timeout`) from terminal (`schema`, `json_shape`).

A hypothesis whose experiment died on a 429 is retryable. One whose change is
already implemented is a dead end. Recording both as "failed" loses the only
distinction that matters for what to do next.

## Exit criteria

1. A redundant hypothesis is retired **the first time** it is detected, with the
   symbol that proves it, and is never selected again.
2. `aider_no_edit` no longer covers redundancy — the two are separately
   countable, so M19 step 2 measures the adapter rather than the backlog.
3. Evidence gathering reopens on staleness alone, so no backlog size can hold it
   shut indefinitely.
4. A hypothesis disproved by measurement ranks below one that has never been
   tested. Today it does not.
5. A campaign on rogii selects a hypothesis that is **not** already implemented,
   without anyone editing the workspace by hand.

## Traps

**Do not fix this by raising the backlog threshold.** `< 10` instead of `>= 3`
still shuts on 46 junk rows. The staleness clause is what guarantees recovery;
the count is only meaningful once it counts viable ideas.

**Do not let the critic invent a curated list of "good" techniques.** That is
the pattern rejected four times — `KNOWN_TECHNIQUES`, the package allowlist, the
template pack, the technique→symbol map. Redundancy is decided against *this
workspace's code*, and promise against *this competition's evidence*.

**Do not mark a hypothesis dead on a transient failure.** The system already
lost a real improvement once by recording `SWA` as harmful; retiring ideas on
infrastructure noise would be the same mistake with a different mechanism.
