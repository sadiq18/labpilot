# Backlog — Capability registration

**Status:** Backlog (not M3). Pick up when the Campaign Engine needs a larger action space.

## Problem

M3 maps research actions onto the **existing** tool catalog only. When policy
repeatedly emits `no_capability` / suggestions, the OS needs a way to **register
new tools** so Conductor can expand without code forks.

## Proposed later work

- Registry API to add tools at runtime or via config plugins
- Versioned capability descriptors (name, schemas, cost hints)
- Conductor observe includes newly registered names automatically
- Guardrails: approval before enabling high-risk capabilities

## Signals to watch (M3 metrics)

- `no_capability` counts
- Suggestion text themes
- Repeated human interventions around the same missing step

## Out of scope here

Implementing registration itself — M3 only **emits** the gap signals.
