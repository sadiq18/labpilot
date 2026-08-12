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
| **Ground truth** | **M22** → **M23** → **M24**, then **M25** → **M26** | Everything below optimises *over* a dataset description nobody checked. Re-profiling rogii on 2026-08-13 inferred the target as `EGFDU`, a horizon depth, and a six-experiment campaign scored 91× worse than one line of code without noticing. These come first because a wrong answer here makes every phase under it a more sophisticated way of optimising the wrong problem |
| **0** | ~~M14~~ **complete 2026-08-07** — phases 1, 2a, 2b, 3 | Degradation is now visible *and* fatal on request; the 20 rule engines are gone |
| **1** | ~~M10 wiring ⇄ thin M7 slice~~ **both done 2026-08-07** | The cycle closed: a real codegen call produced working training code, and two hypotheses produced different scores. [Evidence](evidence-log-2026-08-07.md) |
| **2** | ~~**M7** full~~ **done** · ~~**M18**~~ **done** | M7's differ-table showed every `applied` technique changes the emitted code; M18 gave the vocabulary derived status |
| **3** | ~~**M19**~~ **done 2026-08-09** · ~~**M8**~~ **shipped 2026-08-11**, unmeasured · **M17** | The missing writer landed with M8's score writer (PR #125), which is what M17 also needed |
| **4** | **M13** (needs M7+M8) · ~~**M11**~~ **shipped 2026-08-11/12**, unmeasured | M11 did not need M8; both are now in and neither has been demonstrated on a campaign |
| **5** | **M16** (needs M11 + M14 full) · **M12** (needs M7 + M8) | Last |

Standing throughout: **M9** (verification-first), **M15** (capability audit),
**M20** (gates must be able to fail), **M14** phases 2–3 once the test
migration is budgeted.

~~**M20 is blocked by nothing and can start at any point.**~~ **Done
2026-08-12.** It was listed as standing rather than phased because its items are
independent of the research loop — and the cost of having deferred it was real:
on 2026-08-08, eight gates reporting `pass` on things that could not run took
nine campaign runs to find, and four of them had been read and approved.

**What is now unmeasured is the loop itself.** M8 and M11 both ship
implementations whose exit criteria can only be met by a campaign log, and no
campaign has been run since the template pack was retired. That is the next
piece of evidence this roadmap needs, and no amount of further mechanism work
substitutes for it.

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

**Resolved (2026-08-07).** The cycle closed. With a capable provider
configured, a real codegen call produced a `train.py` that ran and wrote
metrics, and a campaign produced **MSE 194.80 → 190.97** — the first time two
experiments in this system have differed. Both halves are validated by the same
run rather than by unit tests, which is what the cycle demanded.

Worth recording *how* it was unblocked: four separate wiring defects stood
between "routing configured" and "codegen runs", and all four were invisible
until M14 phase 2a made a missing LLM a refusal rather than a silent degrade.
The full account is in [evidence-log-2026-08-07.md](evidence-log-2026-08-07.md).

What is still unproven: the **recipe** path. Every template gate is built and
tested, and no campaign has applied one — the improvement came through LLM
codegen. M7's gate contract holds in tests only.

### Corrections to an earlier statement of this order

The chain "M10 → M7 → M8 → M13" was accurate but incomplete. It hid the M10⇄M7
cycle, placed M14 phase 1 vaguely "alongside" when it belongs first, separated
M17 from M8 despite their shared wiring, implied M11 waits for M8 when it only
needs M7, and omitted M11/M12/M16/M17 entirely.

| # | Plan | Unlocks | Status |
|---|------|---------|--------|
| **M10** | [LLM tiering & routing](04-llm-tiering.md) | **A trustworthy reasoning substrate. Everything downstream inherits its quality.** | **v0.1 shipped + exit criterion 3 met 2026-08-07** — a real codegen call produced a working `train.py` that ran and wrote metrics ([evidence](evidence-log-2026-08-07.md)) |
| **M7** | [Technique → model](01-technique-to-model.md) | Anything at all. Without it there is nothing to optimise over. | **Done 2026-08-07: MSE 194.80 → 190.97**, the first distinct scores, via the LLM path. A differ-table confirmed every `applied` technique changes the emitted code and every `not_applicable` one says why — so the real blocker was undeclared dependencies, not the technique path ([evidence](evidence-log-2026-08-07.md)) |
| **M8** | [Objective feedback loop](02-objective-loop.md) | The system noticing it is making no progress | **Implementation shipped 2026-08-11** (PRs #123, #125, #128, #131): score writer, score summary, stagnation mint. **Exit criteria not demonstrated** — all three ask for evidence from a campaign log, and no campaign has been run since |
| **M9** | [Verification-first execution](03-verification-first.md) | Trusting any result | Partly done |
| **M11** | [Parallel branches](05-parallel-branches.md) | Iteration speed | **Implementation shipped 2026-08-11/12** (PRs #122, #126, #127, #132, #135, #136, #138): parallel branches, worktree isolation and disk accounting, compute budget, k-way fan-out. **Exit criteria not demonstrated** — all three need a campaign that actually fans out |
| **M12** | [Beyond Kaggle](06-beyond-kaggle.md) | The actual product thesis | Not started |
| **M13** | [Policy reasons about state](08-policy-reasoning.md) | Decisions instead of keyword matches | Not started |
| **M14** | [LLM required; delete rule engines](09-llm-required.md) | Failure becomes impossible to miss | **Complete 2026-08-07.** 2b ships default-off against a measured 3.2% fallback rate; a 30-step campaign completed with strict mode on. Phase 3 retired all 20 rule engines (PR #104) behind the `structured_output` precondition (PR #98) that makes removal safe |
| **M18** | [Technique vocabulary earns its entries](13-technique-vocabulary.md) | Attention goes somewhere defensible | **shipped 2026-08-07** (PR #100 step 1, PR #101 step 2). Status derived from evidence; the junk (`the`, `Breath Focus practice`) no longer reaches the planner and `SWA` still does |
| **M19** | [An experiment is a change to its parent](14-experiments-as-deltas.md) | Validation discipline is preserved, and violations are visible | **All five exit criteria met 2026-08-09** (PRs #110–#118). `codegen.strategy` defaults to `delta`; the Jinja pack is deleted in the same commit. Measured over 18 attempts: 1 failure (5.6%), 0 template fallbacks since the codegen fixes. §5's fifth check (validation-region flagging) shipped the same day, with F7 leakage discipline alongside it — the region is derived from the validation plan the workspace already declares, so nothing is curated by hand |
| **M20** | [A gate that cannot fail is not a gate](15-gates-must-fail.md) | Verification you can believe — the checks stop reporting pass on things that cannot run | **All five exit criteria addressed 2026-08-12** (PRs #120, #121, #124, #130, #134, #137). Criterion 4 is *mostly* done and names its two gaps. Found and fixed along the way: 20 gates nobody had shown could say no, two `rejects` markers that had never been earned, three paths running model-written code with the operator's credentials, and seven unstamped derived views ([evidence](evidence-log-2026-08-08.md)) |
| **M21** | [A hypothesis earns its turn](16-hypothesis-selection.md) | The campaign stops re-testing work it has already done | **shipped 2026-08-09** (PR #116), which unblocked M19 step 2. Four delta campaigns had produced no experiment because the hypothesis was already implemented and nothing retired it; 46 stale `proposed` rows held evidence gathering shut permanently |
| **M22** | [An inference without evidence is a guess](17-dataset-understanding.md) | **Ground truth.** Every result above is a conclusion about whichever column the profiler happened to pick | Not started. Re-profiling rogii today infers the target as `EGFDU`, a horizon depth; it stayed invisible because `profile.json` was written 2026-08-02 and never re-derived |
| **M23** | [A model that loses to a constant is not a baseline](18-baseline-correctness.md) | Telling "improving" from "below chance" | Not started. rogii's pipeline scored RMSE 1380 where carrying one column forward scores 15.1 — and the campaign reported `1789 → 1409 → 1380` as progress for fourteen steps |
| **M24** | [Understanding is measured, not asserted](19-competition-benchmark.md) | The instrument M22 and M23 need, and the answer to *"the loop itself is unmeasured"* | Not started. `tests/integration/` has held only stale `.pyc` since `109745c` |
| **M25** | [A finding is a statistic, not a plot](20-eda-findings.md) | Understanding the signal, once the problem is framed correctly | Not started. Both promotion triggers in [future-specialists.md](../backlog/future-specialists.md) are now met |
| **M26** | [A feature is a claim its code must honour](21-feature-specs.md) | Creating signal, with attribution that survives | Not started. rogii produced 31 engineered features and nothing can say which of them helped |
| **M15** | [Capability audit](10-capability-audit.md) | Stops the control plane outrunning the tools again | In progress — PR #129 open |
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
  done. **Now the open item**: v0.1 is wired and unit-tested, and the slice that
  would validate it has not run.
- **Routing lives outside labpilot.** M10 v0.1 ships as `src/fitroute/`, which
  imports nothing from `labpilot` — a boundary enforced by a test, not by
  intention, so it can be extracted as an open-source package later. Its own
  roadmap is [docs/smart-router/DESIGN.md](../../smart-router/DESIGN.md);
  outcome learning, model discovery and streaming are designed there and
  deliberately unbuilt.
- **Role routing is opt-in.** `llm.routing.providers` is empty in
  `configs/default.yaml`, so an unconfigured workspace still takes the legacy
  provider-priority path. That keeps existing workspaces working, and it means
  "M10 shipped" does not imply "M10 is in effect here" — `research doctor`
  reports the resolved provider per role and is the check that tells you which.
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

Two evidence logs record every defect found, how it was found, and what it
cost. When a plan's rationale seems excessive, they are why.

- [evidence-log.md](evidence-log.md) — 2026-08-02, the day that produced this
  roadmap.
- [evidence-log-2026-08-07.md](evidence-log-2026-08-07.md) — taking M10 routing
  live and driving the loop to its **first distinct experiment scores**
  (194.80 → 190.97). Also records five wiring defects of one shape, three
  guards that looked protective and were not, and three corrections to
  diagnoses stated too confidently.
- [evidence-log-2026-08-08.md](evidence-log-2026-08-08.md) — restarting a
  workspace dead for a day: **108 consecutive failures, four sessions, no stop
  condition**. Nine campaign runs, fifteen defects, thirteen fixed. Eight share
  one shape — *a gate that does not test what it claims to* — which is why
  [M20](15-gates-must-fail.md) exists. Ends with the first honest **research**
  failure this workspace has produced, rather than an infrastructure one.

### Deferred — understood, not yet fixed

[`deferred/`](deferred/) holds work found while driving this roadmap that was
deliberately not fixed in the PR that found it, with the reason why. Each names
its site, what it costs, and how to measure the fix — so picking one up does not
start from scratch.

| Item | Why deferred |
|---|---|
| [baseline-plan-step-burner.md](deferred/baseline-plan-step-burner.md) | **Live defect.** [#7](evidence-log.md) fixed only the case where a hypothesis exists; with none proposed the idempotent baseline re-runs and burns steps. The fix is campaign policy, not arg resolution |
| [technique-vocabulary-cleanup.md](deferred/technique-vocabulary-cleanup.md) | M18's writers were guarded in PR #112; the rows earlier runs wrote remain. Cleaning them edits the user's workspace |

This is not the [post-M6 backlog](../backlog/README.md), which catalogues
deferred *features*. These are defects and repairs with a known cause.

---

## The one habit to break

M5 shipped parallel agents before the sequential loop could run a single real
experiment. Breadth before depth is exactly how a beautiful control plane ended
up driving a one-motion data plane.

M7 is unglamorous — a recipe table and some plumbing — and it is worth more than
every remaining milestone combined. If you find yourself adding a seventh
provider adapter before a single technique has changed a CV score, that is the
same trap wearing a new hat.

**M10 v0.1 sits close to that line, and it is worth saying so.** It shipped two
adapters and a router while no technique has yet changed a score. Two things
keep it on the right side: it fixed capabilities that already existed and did
nothing — one cache row across nine campaigns, a rate ledger with no callers,
`.env` keys the router could not see — and its exit criterion is a generated
`train.py`, not a passing unit test. Neither justification survives if the next
change is a third adapter instead of that slice.
