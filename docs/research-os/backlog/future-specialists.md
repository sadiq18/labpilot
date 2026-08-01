# Backlog — Future specialists

**Status:** Backlog (not M5 first cut). Promote when metrics show stable skill loops.

## Problem

M5 ships **Implementation + Experiment** only. The long-term Capability Specialist
set is larger; early splits create routing complexity without payoff.

## Long-term hierarchy (target)

```text
Research Conductor
        |
Capability Specialists
        +-- Literature / Paper
        +-- Data Understanding (EDA)
        +-- Feature Engineering
        +-- Implementation          ← M5
        +-- Experimentation         ← M5
        +-- Evaluation
        +-- Critic / Reflection
        +-- Submission
```

## Deferred specialists

| Specialist | Why deferred | Promote when |
|------------|--------------|--------------|
| Paper / Literature | Needs stronger research retrieval/context | Literature intake is a repeated bottleneck after M4+ retrieval matures |
| Reflection / Critic | Depends on better evidence + memory | Evidence cards + events are stable; critic loop is measurable |
| Evaluation | Overlaps Experiment; policy-ish | Compare/leaderboard workflows need a dedicated skill loop |
| Submission | Policy-heavy; external side effects | Always gated; promote only with clear approval UX |
| EDA (Data Understanding) | Domain task under Implementation today | ~30%+ of campaign time on EDA, or repeated leakage/distribution misses |
| Feature Engineering | Overlaps Implementation + Experiment | Feature decisions frequently wrong or thrash without a dedicated loop |

## Progression

```text
M5: Implementation + Experiment
  → add Critic/Reflection
  → add Literature / Data Understanding
  → full Research Agent ecosystem
```

## Out of scope here

Implementing these agents — M5 only registers Impl + Experiment.
