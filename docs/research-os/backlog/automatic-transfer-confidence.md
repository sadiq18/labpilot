# Backlog — Automatic transfer with confidence scoring

**Status:** Backlog (M7+). M6 ships human-visible warm-start only
(`research memory seed` / `inspect`) plus retrieve-always via Context Engine.

## Problem

Operators should not have to remember to seed every new competition. Automatic
transfer can warm-start priors — but silent seeding introduces hidden bias and
makes Conductor decisions harder to explain.

## Proposed later work

- Score candidate experiences (similarity, outcome, recency, modality match)
- Auto-attach priors only above a confidence threshold
- Always attach explainable provenance (which experiences, why, score)
- Opt-out / policy gate; never bypass Conductor approvals
- Metrics: transfer precision/recall vs human seed baseline

## Out of scope here

M6 experience store, extractor, context provider, and seed/inspect CLI
([06-transfer-memory](../milestones/06-transfer-memory/)).
