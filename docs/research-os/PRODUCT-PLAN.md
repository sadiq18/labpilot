# LabPilot — 12-Month Product and Startup Plan

**Document status:** Current direction  
**Audience:** maintainers, prospective design partners, and independent ML researchers  
**Updated:** 2026-08-19

This is LabPilot's canonical current-direction document. It states what the
project is building next, who it is for, how progress is measured, and which
work remains deliberately deferred. Historical milestone, design, and evidence
documents remain the record of why earlier decisions were made.

## Thesis and boundaries

LabPilot helps ML researchers choose, execute, and learn from experiments. Its
value is structured problem understanding, evidence-backed experiment decisions,
correct execution, and durable research memory.

Kaggle is the initial proving environment because it has public tasks,
repeatable inputs, and measurable feedback. It is not the product boundary. The
first users are independent ML researchers; small ML teams are the next segment
only after the single-researcher loop is trustworthy.

LabPilot is not a generic coding agent, AutoML service, graph-database product,
or autonomous submission bot. Coding tools execute approved changes; LabPilot
decides what deserves to be tried and retains the evidence.

## Current truth

M1–M6 infrastructure is implemented: artifacts, workspace, tools, Conductor,
campaigns, context, specialists/events, parallel execution, and transferable
experience records. This is control-plane infrastructure, not proof that the
system performs reliable research.

The immediate trust foundation is mandatory:

1. **M22 — Dataset understanding:** critical dataset inferences need evidence,
   confidence, alternatives, and an ask-or-block path.
2. **M23 — Baseline correctness:** open-ended research cannot start until a
   generic model is compared with a valid trivial floor.
3. **M24 — Competition benchmark:** a reproducible real-competition corpus and
   scorecard must measure the first two claims beyond hand-written tests.

Until these are complete, operators manually verify target, validation scheme,
metric, and baseline before trusting a campaign or submission.

## Roadmap

### Phase 1 — Trust foundation (months 1–3)

**Product outcome:** On a benchmark corpus, LabPilot states what it predicts,
how it is scored, and whether a baseline is meaningful—or blocks and asks
instead of guessing.

- Deliver M22, M23, and M24 in dependency order.
- Wire `ObjectiveSpec.blocks_launch` into the real preflight path.
- Publish a reproducible scorecard with a ratchet and known-failure ledger.
- Record real campaign failures as evidence, not product success.

**Customer validation:** Interview 8–10 independent ML researchers about their
baseline, validation, and experiment-history workflow. Show the benchmark
report and test whether the blocking behaviour prevents failures they recognize.

**Go/no-go:** Proceed only when the scorecard measures target/train-test/metric
understanding, dummy-baseline validity is 100% where defined, and generic
baseline performance is measured on full data. Otherwise stop feature expansion
and repair the benchmarked path.

### Phase 2 — Research efficacy (months 4–6)

**Product outcome:** Once the problem is trustworthy, LabPilot proposes and
tests smaller, evidence-backed changes that affect the next experiment.

- Deliver M25 deterministic EDA findings and M26 feature specifications.
- Finish M13 state-aware policy and M17 goal/plateau stopping.
- Close M8 and M11 with campaign evidence: score feedback changes decisions and
  fan-out provides measurable, independent throughput.
- Keep each hypothesis, feature group, and conclusion tied to a finding, parent
  comparison, and metric direction.

**Customer validation:** Recruit 3–5 researchers to replay a completed or live
competition. Measure whether the recommended next experiment is understandable,
evidence-cited, and preferable to their unaided shortlist.

**Go/no-go:** Continue only if campaigns show distinct valid outcomes and users
report that evidence changes or accelerates a decision. More runs alone are not
efficacy.

### Phase 3 — Private alpha (months 7–9)

**Product outcome:** A small set of independent researchers safely uses the full
loop in their own workspaces without maintainer intervention.

- Add plan, accept-edits, and auto interaction modes only after the trusted
  loop exists.
- Improve coding-tool adapters and measured specialist loops that block real
  users; do not add specialists because a role sounds useful.
- Complete capability registration and opt-in, redacted campaign telemetry.
- Ship guided onboarding, recovery, and a concise research-cycle demo.

**Customer validation:** Onboard 5–10 private-alpha researchers. Run weekly
sessions covering setup, first baseline, first decision, and repeat use.

**Go/no-go:** Proceed when a majority completes a valid first baseline without
maintainer repair, at least half return for a second cycle, and users can name
one decision made better by the evidence trail.

### Phase 4 — Product validation (months 10–12)

**Product outcome:** Validate repeatable value for 10–20 independent ML
researchers and decide whether to build for team workflows.

- Stabilize the workflow and publish supported capabilities, cost expectations,
  privacy posture, and failure modes.
- Run a paid-design-partner test with high-intent users; price demonstrated
  research-decision value, not hypothetical autonomy.
- Assess team requirements: shared workspaces, private-code adapters,
  collaboration, tenancy, governance, and support burden.

**Customer validation:** Track activation, repeated research cycles, decision
quality, willingness to pay, and retention—not sign-ups or benchmark scores
alone.

**Go/no-go:** Invest in team workflows only if independent-researcher use repeats
and design partners pay or make a concrete commitment. Otherwise improve the
single-researcher loop rather than building enterprise infrastructure.

## Scoreboard

| Measure | Product signal | Customer signal |
|---|---|---|
| Dataset/objective correctness | Benchmark pass rate, abstention correctness, scorecard ratchet | Trust in the explanation or block |
| Baseline validity | Valid floor; generic baseline beats it when defined | Time to believable first baseline |
| Research efficacy | Distinct experiments, valid comparisons, hypothesis/feature success rate | Evidence changes the next decision |
| Autonomy safety | Blocked/waived actions, recovery success, cost per valid experiment | First cycle without maintainer repair |
| Retention and value | Repeat cycles and run cost | Return use, willingness to pay, design-partner commitments |

Metrics never replace judgment: a result is invalid if its objective, validation,
or baseline gate is unresolved. Campaign throughput is not success unless its
experiments are valid and distinct.

## Deferred until evidence requires it

| Capability | Pull it forward only when… |
|---|---|
| Graph backend / Kuzu | SQL graph queries are a measured bottleneck or cannot express a needed query |
| Hybrid semantic retrieval | BM25/context metrics show persistent lexical-recall failures |
| Broad multi-tenancy | Repeated team users require shared state and governance |
| Distributed scheduler / async Conductor | Valid multi-campaign work is blocked by one-process scheduling |
| General scientific domains | The ML loop is benchmarked and a new domain has an executable measurable objective |
| New specialists | A measured skill loop consumes material campaign time or causes recurring errors |
| Automatic memory transfer | A sufficiently large experience corpus proves transfer quality and safety |

## Documentation lifecycle

Use one of these labels at the top of material documents:

- **Current direction** — active product strategy and sequencing.
- **Current contract** — behavior an operator or contributor can rely on now.
- **Historical record** — implemented rationale; preserve it, but do not use it
  to infer current priorities.
- **Proposed** — scoped future work that is not implemented.
- **Backlog** — unscheduled work with an evidence-based pull condition.
- **Superseded** — retained only for history and linked to its replacement.

The [autonomy roadmap](autonomy-roadmap/README.md) remains the detailed
engineering source for M7+; this document defines product sequencing and
validation. The [backlog](backlog/README.md) remains the deferred-work catalog.
