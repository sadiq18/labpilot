# Plan 2 — ExperienceExtractor

Back to [README.md](README.md).

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

- [ ] Given experiment (+ optional reflection), extractor produces a valid Experience Record
- [ ] Idempotent: re-extract same experiment upserts the same record
- [ ] `git_commit` copied into artifacts when present on experiment
- [ ] Tags populated from available signals without requiring LLM
- [ ] Unit tests with fixture experiment/reflection payloads
- [ ] No first-class prompt/HP/paper pattern tables

## Out of scope

- Context Engine wiring (plan 3)
- CLI (plan 4)
- Automatic subscription to event bus (plan 5 — this plan may expose a callable)
- LLM-based narrative rewriting as a hard dependency
