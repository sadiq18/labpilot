# Plan 4 — Memory CLI (seed / inspect)

Back to [README.md](README.md).

**Status:** Done.

## Goal

Human-visible warm-start and debug surface:

```text
research memory seed --from <source-slug> [--competition <target>]
research memory inspect --similar-to <slug> [-q "…"] [--limit N]
research memory list [--competition <slug>] [--outcome success|fail] [--tag …]
research memory show <experience-id>
```

- **seed** — explicitly attach / materialize priors from source competition experiences
  into the target workspace (operator-controlled; auditable).
- **inspect** — show what retrieve would surface (trust/debug), without implying
  Conductor will auto-apply strategy changes.
- **list / show** — browse the ExperienceStore.

Retrieve-always (plan 3) remains the architectural default; CLI does not replace it.

## Acceptance

- [x] `seed` and `inspect` implemented and documented in CLI docs
- [x] `seed` is explicit (no side effect from `conduct` / campaign start alone)
- [x] `inspect` output shows experience ids, outcomes, tags, artifact refs
- [x] `list` / `show` work against ExperienceStore
- [x] Smoke tests or CLI tests for seed + inspect happy paths
- [x] Help text states: memory influences via ContextBundle; seed is operator-driven

## Implementation notes

- CLI: `labpilot.cli.memory_cli` → `research memory …`
- Seed writes `knowledge/memory/seeds/<source>.json` (auditable); provider may boost those ids

## Out of scope

- Automatic seeding at campaign start ([backlog](../../backlog/automatic-transfer-confidence.md))
- Write hooks (plan 5)
- Pattern extraction dashboards
- Multi-tenant org sharing UI
