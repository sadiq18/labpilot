# Plan 6 — Capstone

Back to [README.md](README.md).

## Goal

End-to-end offline campaign: multi-step action, budget stop, resume after pause,
unmappable suggestion. Checklist for M4.

## Acceptance

- [x] Integration tests green (`tests/unit/test_campaigns.py`)
- [x] M4 handoff checklist written

## M4 handoff checklist

- [x] Context engine consumes Conductor observe + artifact refs (no new tool invent)
- [ ] Autonomy level 2 design (budget/policy-change pauses) — not enabled in M3
- [x] Asyncio/AnyIO boundaries for context retrieve (see M4/M5 READMEs)
- [x] Keep submit family gated; do not ungated live Kaggle in M4
- [ ] Capability registration remains backlog until `no_capability` volume justifies it
- [ ] Campaign metrics / suggestions feed context ranking experiments
- [ ] Telemetry export (OTel + Phoenix/Langfuse) and S3 suggestions remain backlog
- [ ] Shared user/team/org store across competitions remains backlog
