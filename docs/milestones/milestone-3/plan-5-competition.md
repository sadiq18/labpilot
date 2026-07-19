# Plan 5 — CompetitionAnalyzer

Back to [Milestone 3](README.md). Design: README §3.5 Competition Intelligence.

**Status:** Not started. **Depends on:** Plan 1. **Unlocks:** Plan 8.

---

## Goal

Ship `CompetitionAnalyzer` as the Kaggle expert brief: competition profile (task, metric,
dataset, constraints), related competitions via official API, and winning-solutions capability
as **ok | unavailable** (`NullWinningSolutionProvider` — no HTML scrape in M3).

## Why this matters

Retrieval Intent and Hypothesis Assistant need a structured competition profile. Related comps
seed Suggested techniques without claiming local belief.

## In scope

- `CompetitionAnalyzer` + `CompetitionProfile`
- Related-competition provider (official API)
- `WinningSolutionProvider` protocol + Null provider (`status: unavailable`)
- External data / inference limits as structured fields
- Persist artifacts under store when Plan 2 available; always contribute to analyze merge

## Out of scope

- HTML scraping of writeups (Future + ToS spike)
- Forum discussions (Spike + Plan F)
- LLM summarization of competition pages

## Design summary

- Deterministic parse only (§2.4). Capability provider pattern for solutions.

## Implementation checklist

| Path | Work |
|------|------|
| `intelligence/analyzers/competition.py` | Analyzer |
| Providers for related comps / solutions | Official API + Null |
| Tests | Mocked Kaggle client fixtures |

## Acceptance criteria

- `research analyze competition <slug>` yields profile artifact(s).
- Winning solutions section reports unavailable without crashing when Null provider used.
- No HTML scrape code paths in M3.

## Test plan

- Unit: mock API → profile fields populated.
- Unit: NullWinningSolution → explicit unavailable status in payload/notes.
- Network tests optional, marked, not required for CI green.

## Review notes

- No special-case `if kaggle: scrape_html()` inside analyzer.
