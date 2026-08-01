# Plan 2 — Budgets and stop conditions

Back to [README.md](README.md).

## Goal

Automatic stops: submission count, wall-time, $/LLM cost, LB/metric target,
plateau (N experiments, no gain). Operator pause remains a manual control.

## Acceptance

- [x] Each stop condition is unit-tested with fixtures
- [x] Loop exits with clear stop reason on budget/target/plateau
