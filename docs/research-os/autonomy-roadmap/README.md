# Autonomy Roadmap

Plans produced from a full-day validation session (2026-08-02) that drove
`research conduct` against `rogii-wellbore-geology-prediction` until the
autonomous loop's real limits surfaced.

**Read [00-diagnosis.md](00-diagnosis.md) first.** Everything else follows from
one finding, and the plans are worthless without it.

---

## The finding

> Every milestone shipped its **structure** but not its **function**.

The stores, registries, event bus, memory layers, agent runtime and task queue
all exist and are well built. Almost none of them *change an outcome*. Memory
has four layers and nothing reads them to make a decision that alters a model.
The event bus exists and no agent reacts to `ExperimentCompleted`. M5 shipped
parallel workers and the campaign runs strictly sequential.

The sharpest instance: the Conductor can decide "try a CNN", and nothing
downstream can produce one. Twelve hypotheses were proposed, planned, run and
reflected on — and every experiment scored **MSE 194.80, identically**.

You built the brain and the nervous system. You did not build hands that can do
more than one motion.

---

## Build order

Milestone numbers reflect when each plan was written, not the order to build
them. Order below is derived from the `Blocked by` line of each plan, not
asserted.

| Phase | Work | Why here |
|-------|------|----------|
| **0** | **M14 phase 1** — stamp rule-engine results as degraded | Nearly free, and it makes every later phase *observable*. This alone would have exposed the silent-degradation class on day one |
| **1** | **M10** wiring **⇄ thin M7 slice** | Mutual, not sequential — see below |
| **2** | **M7** full · adopt **M15**'s contract test | M15's "different input ⇒ different artifact" *is* M7's exit criterion generalised, so the practice starts here |
| **3** | **M8 + M17** together | Both need the same missing writer: `metric_history` / `last_metric` are read in four places and written in none |
| **4** | **M13** (needs M7+M8) · **M11** (needs M7 only) | M11 does **not** need M8 and can start as soon as M7 lands |
| **5** | **M16** (needs M11 + M14 full) · **M12** (needs M7 + M8) | Last |

Standing throughout: **M9** (verification-first), **M15** (capability audit),
**M14** phases 2–3 once the test migration is budgeted.

### The one cycle: M10 ⇄ M7

These are **mutually dependent** and cannot be ordered linearly:

- M7 needs a competent model — its *path efficacy* metric compares registry
  recipes against LLM implementations and is unmeasurable without one.
- M10 needs M7 to prove itself — its real purpose is "codegen gets a model that
  can write training code", and the only proof is generating training code.

Resolution: **M10 wiring first, then a thin M7 slice (one technique, one
dataset) validates it, then M7 in full.** Shipping M10 on unit tests alone would
repeat the mistake review already caught, where `select_route` was tested,
unwired, and called done.

### Corrections to an earlier statement of this order

The chain "M10 → M7 → M8 → M13" was accurate but incomplete. It hid the M10⇄M7
cycle, placed M14 phase 1 vaguely "alongside" when it belongs first, separated
M17 from M8 despite their shared wiring, implied M11 waits for M8 when it only
needs M7, and omitted M11/M12/M16/M17 entirely.

| # | Plan | Unlocks | Status |
|---|------|---------|--------|
| **M10** | [LLM tiering & routing](04-llm-tiering.md) | **A trustworthy reasoning substrate. Everything downstream inherits its quality.** | Decision layer only — not wired |
| **M7** | [Technique → model](01-technique-to-model.md) | Anything at all. Without it there is nothing to optimise over. | [Designed](design/01-technique-to-model.md) — blocked on M10 |
| **M8** | [Objective feedback loop](02-objective-loop.md) | The system noticing it is making no progress | Not started |
| **M9** | [Verification-first execution](03-verification-first.md) | Trusting any result | Partly done |
| **M11** | [Parallel branches](05-parallel-branches.md) | Iteration speed | Not started |
| **M12** | [Beyond Kaggle](06-beyond-kaggle.md) | The actual product thesis | Not started |
| **M13** | [Policy reasons about state](08-policy-reasoning.md) | Decisions instead of keyword matches | Not started |
| **M14** | [LLM required; delete rule engines](09-llm-required.md) | Failure becomes impossible to miss | Decided |
| **M15** | [Capability audit](10-capability-audit.md) | Stops the control plane outrunning the tools again | Not started |
| **M16** | [Evidence routine as background producer](11-background-routine.md) | Gathering stops blocking testing | Gating shipped |
| **M17** | [Run until plateau or goal](12-run-until-done.md) | Campaigns end on the objective, not a step counter | Not started |
| — | [Interaction modes](07-interaction-modes.md) | Auto / accept-edits / plan UX | Not started |

### The principle that sets the order

> **Model capability is a product tier, not an architectural constraint.**

An earlier version of this roadmap was shaped by the machine it was validated on
— a 14B local model — and let that become the design's ceiling. A paying
customer supplies a frontier model; the free tier is a development mode with
known limits, not the substrate to build for.

This is a research OS. A weak component does not merely run slower — it produces
**wrong research**, which propagates into beliefs, claims and memory where it is
expensive to remove. No component is left weak because the current dev
environment is.

### Ordering notes

- **M10 is now first, not M7.** Not for cost control — to make the reasoning
  substrate trustworthy before anything depends on it. M7's evaluation is also
  *incomplete* without it: path efficacy compares registry against LLM
  implementations and cannot be measured with a model that cannot write code.
- **M10 must be validated by a thin M7 slice.** Its real purpose is "codegen
  gets a model that can write training code", and the only proof is generating
  training code. Shipping it on unit tests alone would repeat the mistake review
  already caught once, where `select_route` was tested, unwired, and called
  done.
- **M14 phase 1 is nearly free and should go early.** Merely *stamping*
  rule-engine results as degraded would have made the entire silent-failure
  class visible on day one. Phases 2–3 need a test migration (see the plan).
- **M13 requires M7 and M8.** "Plateaued" cannot be detected while every
  experiment returns the same score.
- **M16's skip condition already ships.** Evidence gathering is gated on
  backlog **and** artifact freshness; only the background *routine* remains.
- **M17 shares M8's wiring.** Both need the metric fed into durable state;
  `metric_history` and `last_metric` are currently read in four places and
  written in none, so `metric_target` and `plateau` can never fire.
- **M15 is a standing practice, not a one-off.** Its contract test — *different
  input must produce a different artifact* — is the generalisation of M7's exit
  criterion and the check that would have caught the hollow layer immediately.

---

## How to use these documents

Each plan carries:

- **Purpose** — the concrete failure it removes, with evidence
- **Goal** — what becomes possible
- **Approach** — design decisions *and their rationale*
- **Exit criteria** — a check that cannot be satisfied by accident
- **Traps** — approaches already tried and rejected, so they are not retried

[evidence-log.md](evidence-log.md) records every defect found in the session,
how it was found, and what it cost. When a plan's rationale seems excessive,
the evidence log is why.

---

## The one habit to break

M5 shipped parallel agents before the sequential loop could run a single real
experiment. Breadth before depth is exactly how a beautiful control plane ended
up driving a one-motion data plane.

M7 is unglamorous — a recipe table and some plumbing — and it is worth more than
every remaining milestone combined. If you find yourself adding a seventh
provider adapter before a single technique has changed a CV score, that is the
same trap wearing a new hat.
