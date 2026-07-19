# Spike — Kaggle discussion access

Back to [Milestone 3](README.md). Design: README §6.7, §15 Spike.

**Status:** Not started. **Depends on:** nothing (parallel). **Unlocks:** Plan F go/no-go for Kaggle provider.

---

## Goal

Produce a short feasibility + Terms of Service note on accessing Kaggle competition
discussions (official API vs authenticated HTML). **Not production code.** Does not block
Plans 1–11.

## In scope

- Document available endpoints / undocumented surfaces
- ToS / robots / auth constraints
- Caching implications if ever allowed
- Go / no-go / “GitHub Issues first” recommendation

## Out of scope

- Shipping `DiscussionAnalyzer` or scrapers
- Changing Phase 1 scope

## Deliverable

`docs/milestones/milestone-3/spike-kaggle-discussions-notes.md` (or section in this file
filled after investigation) with explicit recommendation.

## Acceptance criteria

- Written go/no-go with evidence links/dates.
- Clear statement whether Plan F Kaggle provider may proceed.

## Review notes

- Spike must not introduce production HTML scrape into main tree.
