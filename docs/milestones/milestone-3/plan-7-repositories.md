# Plan 7 — Repository / GitHub Intelligence

Back to [Milestone 3](README.md). Design: README §5.

**Status:** Not started. **Depends on:** Plans 1, 3 (Plan 2 for persist). **Unlocks:** Plan 8.

---

## Goal

Ship `RepositoryAnalyzer` + GitHub `RepositoryProvider`, `RepositoryAnalyzerAgent` structured
repo cards (`RepoKnowledge`), and `RepoDiffer` vs local code for `TransferOpportunity`
(effort + expected gain). Targeted file fetch + cache only — no wholesale clone of search hits.

## Why this matters

Winning / strong repos are the transfer surface. Diff-vs-local is the product moment; README
dumps are not.

## In scope

- Search/fetch via official GitHub API; raw under `research/raw/repositories/`
- Extract architecture / loss / aug / tricks / files / deps
- Diff vs local profile → transfer opportunities (suggest only; no auto-edit `train.py`)
- Soft-fail per repo

## Out of scope

- HTML scraping GitHub
- Auto-applying diffs to local templates
- Forum issues as DiscussionAnalyzer (Plan F may reuse GitHub Issues provider)

## Implementation checklist

| Path | Work |
|------|------|
| `analyzers/repositories/*` | Analyzer, provider, differ, local_profile |
| `micro_agents/repository_analyzer/` | Agent + skill.md |
| Tests | Fixture README + file set |

## Acceptance criteria

- `research analyze repositories <slug>` yields repo artifacts + optional transfer notes.
- Differ produces effort/gain fields without modifying workspace training code.
- No full-repo summarization product output.

## Test plan

- Unit: extract from fixture files via rule_engine/Agent stub.
- Unit: differ against fake local profile.
- Optional live GitHub tests marked.

## Review notes

- Network only in provider; Agent sees cached text only.
