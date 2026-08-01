# Research Graph — logical layer over SQLite

Canonical design for evidenced knowledge edges. Pair with
[evidence-card.md](evidence-card.md).

**Decision:** SQLite remains the system of record. This is a **logical graph**
(API + edge writes), not a graph database. Migrating to Neo4j/etc. is backlog
only if SQL becomes a bottleneck. Cross-competition shared knowledge is also
backlog; local `reusable_for` is the precursor.

---

## Feedback loop

```mermaid
flowchart TD
  artifacts[Artifacts techniques] --> hyps[Hypotheses]
  hyps --> run[Plan run treatment E]
  run --> metrics[Rich metrics]
  metrics --> compare[COMPARE vs control E]
  compare --> card[Evidence Card]
  card --> graph[Research Graph edges]
  card --> beliefs[Beliefs confidence]
  card --> claims[Claims confidence]
  graph --> query[Graph query for planner]
  beliefs --> artifacts
  query --> hyps
```

---

## Canonical path (every edge carries evidence)

Score-only rows are insufficient. Every future planner should walk evidenced
edges:

```text
Technique  --supports-->  Claim
    --used_in-->  Hypothesis
    --executed_as-->  Experiment/Execution
    --produced-->  Evidence Card (+0.007 CV, …)
    --updates-->  Belief
```

Example:

```text
Mixup
  │ supports
  ▼
"Regularization improves generalization"
  │ used_in
  ▼
H-014
  │ executed_as
  ▼
E-042
  │ produced
  ▼
EV-xxx  (observed.cv_gain=+0.007)
```

---

## Tables (existing SoR)

| Edge | Mechanism |
|------|-----------|
| technique ↔ experiment artifact | `artifact_techniques` (`supports` / `contradicts` / `mentions`) |
| generic targets | `evidence_links` (`used_in`, `executed_as`, `produced`, …) |
| claim ↔ evidence | `claim_evidence` |
| belief mutations | `belief_updates` audit |

Relation tables must be **refreshed on every Evidence Card** — not only during
`analyze` with sparse `mentions`.

---

## Query layer

`research_engine.intelligence.graph.query_techniques` filters local evidence:

```text
techniques where
  belief.confidence > 0.8
  AND reusable_for intersects modality
  AND median train_time_s < 2h
  AND mean observed.cv_gain > 0.003
```

Wired into hypothesize ranking (graph confidence bonus) and available for
retrieve/context builders. Prefer graph hits over paper-only recall when local
evidence exists.

---

## Non-goals (see backlog)

1. **Graph database** — only if SQL graph queries become a bottleneck.
2. **Cross-competition shared knowledge** — aggregate SpecAugment ✓ audio / ✗
   tabular style priors across competitions; Evidence Card `reusable_for` is the
   local hook.

See [milestones/backlog.md](../milestones/backlog.md).
