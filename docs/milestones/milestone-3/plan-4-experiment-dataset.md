# Plan 4 — ExperimentAnalyzer + DatasetAnalyzer

Back to [Milestone 3](README.md). Design: README §3.4, Experiment / Dataset readers.

**Status:** Not started. **Depends on:** Plan 1 (Plan 2 recommended for persist). **Unlocks:** Plan 8.

---

## Goal

Ship local-only analyzers that turn M2 experiment graph / run metrics and dataset profile
signals into `ResearchArtifact` batches (and persist via store when Plan 2 is present).

## Why this matters

Hypothesis Assistant and retrieval need local experiments and failures without network. This
plan delivers standalone value: `research analyze experiments|dataset <slug>`.

## In scope

- `ExperimentAnalyzer` — runs, metrics, pipeline technique tags, failures
- `DatasetAnalyzer` — deterministic profiling (Pandas/NumPy; **no LLM**)
- Emit `ResearchArtifact` type=experiment / dataset
- Soft-fail when no runs / missing data

## Out of scope

- Paper/repo/forum fetch
- Ranking / Hypothesis Assistant (Plans 9–10)
- Writing established beliefs (Plan 8 / §12.4)

## Design summary

- Reuse `research_engine.execution.experiments` read-only.
- Deterministic Engine only for dataset stats (§2.4 Hard No).

## Implementation checklist

| Path | Work |
|------|------|
| `intelligence/analyzers/experiments.py` | Analyzer |
| `intelligence/analyzers/dataset.py` | Analyzer |
| Registry registration | default_enabled |
| Tests | Fixture competition with fake runs |

## Acceptance criteria

- `research analyze experiments <slug>` and `dataset` produce artifacts / notes.
- No LLM calls; no external network.
- Failures surfaced as artifacts or notes when regressions exist in fixtures.

## Test plan

- Unit: assemble artifacts from M2-like run fixtures.
- Unit: empty runs → soft-fail notes, exit 0.

## Review notes

- Does not auto-write M2 `knowledge_base.json` as Established.
