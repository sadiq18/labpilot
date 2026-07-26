# Plan 8 — Journal CLI + recommend

Back to [Research Reflection](README.md). Design: [cli.md](cli.md).

**Status:** Ready. **Depends on:** Plans 2–7 (MVP can ship with 2–5 + stubs).

---

## Goal

Ship `research reflect` and `research journal`; next-experiment suggestion.

## In scope

- CLI commands per [cli.md](cli.md)
- `journal/projector.py` — strength buckets, open questions, beliefs, claims
- `recommendation/next_experiment.py` (LLM + rule_engine)
- Wire into existing CLI module layout under `cli/`

## Out of scope

- Deleting `labpilot.report` / relocating dashboard templates (Plan 9)
- Optional HTML skin for Journal (follow-on after Plan 9)
- Deleting other legacy packages (Plan 9)

## Acceptance criteria

- [ ] `research journal --competition <slug>` prints evidence tiers + next step
- [ ] `research reflect --execution E-xxx` re-runs pipeline idempotently enough
- [ ] Offline path documented and tested
