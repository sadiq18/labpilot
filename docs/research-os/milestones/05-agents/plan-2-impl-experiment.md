# Plan 2 — Implementation + Experiment specialists

Back to [README.md](README.md).

## Goal

Land the first two specialists and wire Conductor routing to them:

**Implementation Specialist**

- Write / update code, tests, fixes via `CodingTool`
- Read **existing** repo code and patch in place (not always regenerate a monolith `train.py`)
- Prefer clearer layout: train vs inference (and related modules) over dumping everything into one file
- Treat EDA and feature work as **code tasks under Implementation** — not separate agents

**Experiment Specialist**

- Run experiments (sandbox / existing execute paths)
- Collect metrics and compare results
- Produce / update experiment artifacts and evidence hooks
- Emit completion signals for the event bus (plan 3 can subscribe)

Conductor selects Impl vs Experiment via registry (capability + budget + context). Strategy
and approval policy stay with Conductor.

## Acceptance

- [x] Implementation + Experiment registered and routable from Conductor
- [x] Implementation updates existing code when present; does not always rewrite from scratch
- [x] Generated/updated layout keeps train and inference concerns separable
- [x] Experiment runs produce metrics on experiment artifacts
- [x] No peer agent→agent calls; path is Conductor → registry → specialist → tools
- [x] Submit remains gated; no ungated live Kaggle in this plan
