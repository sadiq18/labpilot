# Plan 1 — Artifact contracts

Back to [Foundation](README.md). Design: [artifacts.md](../../design/01-artifacts.md).

## Goal

Introduce stable typed artifact boundaries and adapters so stages no longer depend
on each other’s implementations to sequence work.

## Depends on

- Design pass (`research-os-design`) merged or contracts agreed
- V1 pipeline kernel available

## Work (when implementing)

- Define / document adapter APIs for: analysis report, ResearchPlan, execution
  outcomes, Evidence Card refs, reflection outputs, submission records
- Wire producers to write through adapters (prefer thin wrappers over existing stores)
- Add schema_id / version where formats will churn
- Forbid new cross-stage “call execute next” imports (checklist + tests where easy)

## Acceptance

- [ ] Adapters documented and importable
- [ ] Round-trip unit tests for ≥3 artifacts
- [ ] Analyze/plan/run/reflect paths use adapters for primary outputs

## Non-goals

- OS Task queue (M2)
- Tool registry (plan 2a)
- CLI rewrite (plan 3)
