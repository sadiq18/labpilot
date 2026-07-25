# Plan 5 — CompetitionAnalyzer

Back to [Milestone 3](README.md). Design: README §3.5 Competition Intelligence.

**Status:** Implemented (Plan 5 + 5b overview/rules enrichment). **Depends on:** Plan 1, Plan 3.
**Unlocks:** Plan 8.

---

## Goal

Ship `CompetitionAnalyzer` as the Kaggle expert brief: competition profile (task, metric,
dataset, constraints), related competitions via official API, overview/rules enrichment
(Micro Agent + `rule_engine` fallback), and winning-solutions capability as **ok |
unavailable** (`NullWinningSolutionProvider` — no writeup/search scrape in M3).

## Why this matters

Retrieval Intent and Hypothesis Assistant need a structured competition profile. Related comps
seed Suggested techniques without claiming local belief. API gaps (external data, inference
limits, evaluation formula, submission format) are filled from overview/rules pages.

## In scope

- `CompetitionAnalyzer` + `CompetitionProfile`
- Related-competition provider (official API / series heuristics)
- `WinningSolutionProvider` protocol + Null provider (`status: unavailable`)
- Overview + rules page fetch/cache via official `competition_list_pages` API +
  `CompetitionPageAnalyzerAgent` (typed extract; LLM optional; same schema via
  `rule_engine`)
- External data / inference limits / evaluation / submission enrich fields
- Persist artifacts under store when Plan 2 available; always contribute to analyze merge

## Out of scope

- HTML scraping of **writeups** / winning-solution search (spike + Future provider)
- Forum discussions (Spike + Plan F)
- Headless browser / JS rendering of SPA shells (public HTML is an empty shell;
  enrichment uses the authenticated pages API instead)
- Free-form LLM page dumps as system of record

## Design summary

- Deterministic API parse (§2.4). Capability provider pattern for solutions.
- Page enrichment: Micro Agent fills typed `CompetitionPageExtract`; LLM upgrades quality
  only — storage shape is identical with `rule_engine`.

## Implementation checklist

| Path | Work |
|------|------|
| `intelligence/analyzers/competition.py` | Analyzer + page enrich merge |
| `competition/page_fetch.py` | Pages API (Description/Evaluation/Rules/…) + RawStore cache; HTTP SPA fallback |
| `micro_agents/competition_page_analyzer/` | Agent + skill.md |
| Providers for related comps / solutions | Official API + Null |
| Tests | Mocked Kaggle + page fixtures |

## Acceptance criteria

- `research analyze competition <slug>` yields profile artifact(s).
- Overview/rules enrich `external_data` / `inference_limits` / evaluation / submission when
  page text is available (LLM or `rule_engine`).
- Missing credentials / empty API pages / HTTP SPA shells → explicit `unavailable`.
- Winning solutions section reports unavailable without crashing when Null provider used.
- No writeup/search scrape code paths in Plan 5.

## Test plan

- Unit: mock API → profile fields populated.
- Unit: NullWinningSolution → explicit unavailable status in payload/notes.
- Unit: page agent rule_engine + fake LLM → same `CompetitionPageExtract` schema.
- Network tests optional, marked, not required for CI green.

## Review notes

- No special-case `if kaggle: scrape_html()` for winning solutions inside analyzer.
- Overview/rules fetch lives in `competition/page_fetch.py`, not in the Null provider.
