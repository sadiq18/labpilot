# Backlog — Capability registration

**Status:** Parked — design done, **do not implement yet**  
**Blocked on:** [telemetry-suggestions-export.md](telemetry-suggestions-export.md)
(client-side log / gap collection so maintainers can see `no_capability` without
user SQLite)  
**Design:** [../design/11-capability-registration.md](../design/11-capability-registration.md)

## Why parked

Product capability evolution needs a **maintainer gap feed**. Local
`os_suggestions` alone is not enough. Implement registration **after** telemetry
(or opt-in export) can collect redacted gaps from clients.

## Problem

M3 maps research actions onto the **existing** tool catalog only. When policy
repeatedly emits `no_capability` / suggestions, the OS needs a way to **register
new tools** so Conductor can expand without code forks.

## Design answers (summary)

| Question | Answer |
|----------|--------|
| **When to add?** | Recurring gap evidence (count / rate thresholds) **or** explicit milestone need; not one-off LLM asks. Prefer **alias** if a synonym of an existing tool. |
| **How created?** | Path A wrap library → Path B external adapter → Path C new handler + tests; all become `ToolDescriptor` on `ToolRegistry` via **maintainer PR**. |
| **How track?** | Local `os_suggestions` (user DB) → **telemetry / opt-in export** → **maintainer** product gap feed → promote/defer/reject → implement in repo. |
| **Who promotes?** | **Maintainer only** — not end users. Users see status / can export; they do not promote into LabPilot. |

## Proposed work (when unblocked)

See design §8 — start only after telemetry client collection exists:

- **P0** — Structured suggestion context + allowlist refresh each loop
- **P1** — Consume telemetry/export as product gap feed (not raw user DB)
- **P2** — Maintainer review (promote / defer / reject) over feed + audit
- **P3** — Optional local gap rollup for `conduct status` UX only
- **P4** — Plugins + risk/approval

## Signals to watch (already partially emitted)

- `CampaignMetrics.no_capability`
- `os_suggestions` themes / `context.intent` / missing tool names
- Repeated human interventions around the same missing step
- Telemetry opt-in / export volume (once telemetry ships)

## Out of scope

- Auto-implementing tools without tests
- Merging Engineer capabilities into `ToolRegistry` wholesale
- Silent high-risk tool enablement
- End-user promote into the shared catalog
- Maintainers reading raw user competition DBs by default
- Implementing this item before telemetry client collection
