# Plan 3 — Event bus (Blinker)

Back to [README.md](README.md).

## Goal

In-process pub/sub on top of the M2 append-only decision/task log so specialists
and Conductor reactions are decoupled.

Publish examples (first cut):

- `ExperimentCompleted`
- `ImplementationFinished`
- `ModelFailed`
- `EvidenceUpdated` (when a subscriber updates evidence)

Subscribers react without hard-coded `Experiment → Reflection` call chains.
Reflection/Critic as a **specialist** stays deferred
([future specialists](../../backlog/future-specialists.md)); a thin subscriber that
updates evidence or enqueues a Conductor observe refresh is enough to prove the bus.

Bus builds on the durable log — does not replace it.

## Acceptance

- [ ] Blinker (or equivalent in-process) bus ships; unit tests for publish → subscribe
- [ ] Experiment completion publishes `ExperimentCompleted` (payload includes experiment id + artifact refs)
- [ ] At least one subscriber updates evidence or notifies Conductor without a hard call from Experiment
- [ ] M2 decision/task log remains source of truth for resume/explain
- [ ] No distributed NATS/Redis requirement in this plan
