# Plan 11 — Capstone (terminal + analyze.json)

Back to [Milestone 3](README.md). Design: README mockup · §12.5 · §13.

**Status:** Not started. **Depends on:** Plan 10. **Unlocks:** Milestone 3 Phase 1 complete.

---

## Goal

Deliver mockup-parity terminal rendering and a validated public `analyze.json` contract under
`knowledge/<slug>/research/reports/analyze.json`. End-to-end offline fixture competition
proves the pipeline without HTML.

## Why this matters

JSON is the contract for CLI / future HTML / API. Terminal is the v1 human view. Capstone
makes Milestone 3 demoable and reviewable as a product slice.

## In scope

- Terminal renderer sections aligned with design mockup (profile, techniques, transfers,
  failures, top-10)
- JSON schema validation for analyze.json
- E2E fixture: fake store + analyzers or recorded fixtures → golden-ish terminal/JSON
- `--format text|json` behavior locked

## Out of scope

- HTMLRenderer (Milestone 4+)
- Live network e2e in CI
- Forum sections unless Plan F shipped (show empty/unavailable)

## Implementation checklist

| Path | Work |
|------|------|
| `intelligence/renderers/terminal.py` | Human summary |
| `intelligence/renderers/json.py` | Schema validate |
| Fixtures + e2e test | Offline |
| Docs: CLI.md / SOP touch if commands finalized | |

## Acceptance criteria

- Offline e2e writes analyze.json that validates against schema.
- Terminal output includes top-10 and does not claim Established for external-only techniques.
- No HTML artifacts required.
- **Success-criteria gate (README §1):** the five north-star questions each return the
  expected grounded answer against the seeded fixture store — evidence ids resolve, Q3 is
  exact, provider-gated facts (Q2 winning solutions) report `Unavailable` with a reason, and
  Q5 excludes already-tried techniques. Passes with Micro Agents disabled (`rule_engine`).

## Test plan

- Golden or snapshot terminal (stable fields).
- Schema validation test.
- Full analyze with include flags on fixture slug.
- **Success-criteria tests:** one test per README §1 question over the seeded fixture store,
  asserting the expected evidence-id set (deterministic parts) and honest `Unavailable` for
  ungated providers — run with Micro Agents both on and off.

## Review notes

- Presentation only — analyzers must not import renderers.
