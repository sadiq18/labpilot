# Spike — Kaggle discussion access

Back to [Milestone 3](README.md). Design: README §6.7, §15 Spike.

**Status:** Not started. **Depends on:** nothing (parallel). **Unlocks:** Plan F go/no-go for Kaggle provider;
also go/no-go for related-competition search + winning-solution providers.

---

## Goal

Produce a short feasibility + Terms of Service note on accessing Kaggle competition
discussions (official API vs authenticated HTML), **and** evaluate ToS-safe access for:

1. Related / previous competitions via Kaggle search (excluding the same competition id)
2. Winning writeups / code links for those competitions
3. Technique-oriented discussion threads

**Not production code.** Does not block Plans 1–11. Plan 5 keeps
`NullWinningSolutionProvider` until this spike clears a provider swap.

## In scope

- Document available endpoints / undocumented surfaces (discussions, search)
- ToS / robots / auth constraints
- Caching implications if ever allowed
- Go / no-go / “GitHub Issues first” recommendation
- Recommendation for winning-solution / related-comp search providers (API vs HTML)

## Out of scope

- Shipping `DiscussionAnalyzer` or production scrapers
- Changing Phase 1 scope / rewriting `CompetitionAnalyzer` before go

## Deliverable

`docs/milestones/milestone-3/spike-kaggle-discussions-notes.md` (or section in this file
filled after investigation) with explicit recommendation.

## Acceptance criteria

- Written go/no-go with evidence links/dates.
- Clear statement whether Plan F Kaggle provider may proceed.
- Clear statement whether a `SearchWinningSolutionProvider` / related-comp search provider
  may proceed (and under what constraints).

## Review notes

- Spike must not introduce production HTML scrape into main tree before go.
- Plan 5 overview/rules enrichment is separate (competition contract pages only —
  via official `list_competition_pages`, not public SPA HTML).
