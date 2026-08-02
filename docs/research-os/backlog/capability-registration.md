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
| **How created?** | Path A wrap library → Path B external adapter → Path C new handler + tests; all become `ToolDescriptor` on `ToolRegistry`. |
| **How track?** | Enrich suggestions → **gap ledger** across sessions → CLI review → promote/defer/reject → verify gap stops growing. |

## Proposed work (phased)

See design §8:

- **P0** — Structured suggestion context + allowlist refresh each loop
- **P1** — `os_capability_gaps` ledger + `research tools gaps`
- **P2** — Human promote / defer / reject + audit
- **P3** — Plugins + risk/approval
- **P4** — Telemetry export

## Signals to watch (already partially emitted)

- `CampaignMetrics.no_capability`
- `os_suggestions` themes / `context.intent` / missing tool names
- Repeated human interventions around the same missing step

## Out of scope

- Auto-implementing tools without tests
- Merging Engineer capabilities into `ToolRegistry` wholesale
- Silent high-risk tool enablement
