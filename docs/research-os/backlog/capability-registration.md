# Backlog — Capability registration

**Status:** Design ready — [../design/11-capability-registration.md](../design/11-capability-registration.md)  
**Pickup order:** [backlog-grooming.md](backlog-grooming.md) `#2` (after Stage 2 facets).

## Problem

M3 maps research actions onto the **existing** tool catalog only. When policy
repeatedly emits `no_capability` / suggestions, the OS needs a way to **register
new tools** so Conductor can expand without code forks.

## Design answers (summary)

| Question | Answer |
|----------|--------|
| **When to add?** | Recurring gap evidence (count / rate thresholds) **or** explicit milestone need; not one-off LLM asks. Prefer **alias** if a synonym of an existing tool. |
| **How created?** | Path A wrap library → Path B external adapter → Path C new handler + tests; all become `ToolDescriptor` on `ToolRegistry` via **maintainer PR**. |
| **How track?** | Local `os_suggestions` (user DB) → **opt-in redacted export** → **maintainer** product gap feed → promote/defer/reject → implement in repo. |
| **Who promotes?** | **Maintainer only** — not end users. Users see status / can export; they do not promote into LabPilot. |

## Proposed work (phased)

See design §8:

- **P0** — Structured suggestion context + allowlist refresh each loop
- **P1** — Export bridge (`export-gaps` / telemetry) so maintainers can read gaps without user SQLite
- **P2** — Maintainer review (promote / defer / reject) over export/feed + audit
- **P3** — Optional local gap rollup for `conduct status` UX only
- **P4** — Plugins + risk/approval

## Signals to watch (already partially emitted)

- `CampaignMetrics.no_capability`
- `os_suggestions` themes / `context.intent` / missing tool names
- Repeated human interventions around the same missing step
- Export volume / opt-in rate (once P1 exists)

## Out of scope

- Auto-implementing tools without tests
- Merging Engineer capabilities into `ToolRegistry` wholesale
- Silent high-risk tool enablement
- End-user promote into the shared catalog
- Maintainers reading raw user competition DBs by default
