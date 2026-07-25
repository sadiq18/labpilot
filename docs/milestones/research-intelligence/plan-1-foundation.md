# Plan 1 — Foundation (registry, orchestrator, CLI)

Back to [Milestone 3](README.md). Design: [README.md](README.md) §3, §11, §13 · [knowledge-system.md](knowledge-system.md).

**Status:** Not started. **Depends on:** Milestone 2 shipped. **Unlocks:** Plans 2–7, F.

---

## Goal

Stand up `research_engine/intelligence/` as a deployable package skeleton: shared
`ResearchArtifact` models, Analyzer Protocol, plugin registry, thin orchestrator, and CLI
`research analyze [analyzer] <slug>` that writes a stub `reports/analyze.json`. Soft-fail
friendly; no real source analyzers required yet.

## Why this matters

Everything else plugs into registry + orchestrator. Without a stable envelope and CLI
contract, Plans 4–7 cannot land as independent PRs.

## In scope

- Package layout under `src/labpilot/research_engine/intelligence/` (models, context,
  registry, orchestrator, renderers stub, analyzers/base)
- `ResearchArtifact` / `ResearchArtifacts` / `ResearchArtifactType` (§3.1)
- `Analyzer` Protocol + registry (`--include` / `--exclude`)
- Orchestrator: select → run → merge → write stub report
- CLI: `research analyze` (default / single / include / exclude / `--format` / `--refresh` stub)
- Soft-fail notes on batch

## Out of scope

- SQLite / `research/` tree (Plan 2)
- Micro Agents (Plan 3)
- Real Competition / Paper / Repo / Experiment analyzers (Plans 4–7)
- Forum providers (Plan F)

## Design summary

- Analyzers emit `ResearchArtifacts`; orchestrator merges and persists report stub under
  `knowledge/<slug>/research/reports/analyze.json` (create parent dirs; full store in Plan 2).
- Prefer evolving toward `research_engine/`; CLI stays thin (`cli/` only).

## Implementation checklist

| Path | Work |
|------|------|
| `src/labpilot/research_engine/intelligence/` | Package skeleton |
| `…/models.py` | ResearchArtifact (+ types, batch) |
| `…/registry.py`, `orchestrator.py`, `context.py` | Core |
| `…/analyzers/base.py` | Protocol |
| `…/renderers/json.py` | Write/validate stub analyze.json |
| `src/labpilot/cli/…` | `research analyze` wiring |
| Tests | Registry, CLI help, dry-run with fake analyzer |

## Acceptance criteria

- `research analyze --help` documents analyzer selection flags.
- Dry-run with a test double analyzer writes valid stub `analyze.json`.
- Unknown analyzer name fails clearly; soft-fail path records notes without crashing.
- Import hygiene: `intelligence` does not import `cli`.

## Test plan

- Unit: registry register/list/select include/exclude.
- Unit: ResearchArtifact schema round-trip.
- CLI: invoke analyze on temp workspace with fake analyzer (no network).

## Review notes

- No LLM calls; no Kaggle/GitHub network in this plan.
- Confirm `ResearchArtifact` field set matches §3.1 (incl. migration aliases).
