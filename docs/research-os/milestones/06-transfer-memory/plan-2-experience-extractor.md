# Plan 2 — ExperienceExtractor

Back to [README.md](README.md).

**Status:** Done.

## Goal

Deterministic **ExperienceExtractor**: map a completed experiment (+ reflection when
available) into an Experience Record and persist via ExperienceStore.

Sources (read, do not rewrite SoR):

- Experiment / execution artifacts (metrics, hypothesis link, parent/compare)
- Reflection / critic outputs (lessons, belief updates — as text/links)
- Code `git_commit` on experiment artifact when M5 git evolution is present
- Competition slug + lightweight tag heuristics (modality, technique keywords)

Tagging is **heuristic facets only** — not a curated taxonomy product. Missing
optional fields stay empty rather than inventing category wikis.

## Acceptance

- [x] Given experiment (+ optional reflection), extractor produces a valid Experience Record
- [x] Idempotent: re-extract same experiment upserts the same record
- [x] `git_commit` copied into artifacts when present on experiment
- [x] Tags populated from available signals without requiring LLM
- [x] Unit tests with fixture experiment/reflection payloads
- [x] No first-class prompt/HP/paper pattern tables

## Implementation notes

- `ExperienceExtractor.extract(...)` — callable for Plan 5 write hooks
- Sources: `Experiment` model, agent `experiment/record.json`, optional reflection/plan/hypothesis
- Outcome from comparison verdict / metric delta / status (rule-based)
- Facets are **rule hints with confidence + evidence** (Stage 1). Further stages:
  [experience-facet-extraction backlog](../../../backlog/experience-facet-extraction.md)

## Out of scope

- Context Engine wiring (plan 3)
- CLI (plan 4)
- Automatic subscription to event bus (plan 5 — this plan may expose a callable)
- LLM-based narrative rewriting as a hard dependency
- Artifact-aware / embedding / LLM / facet-graph extraction (backlog stages 2–5)
