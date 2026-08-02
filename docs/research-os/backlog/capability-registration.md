# Backlog — Capability registration

**Status:** Implementing (local ledger + export + maintainer decisions)  
**Design:** [../design/11-capability-registration.md](../design/11-capability-registration.md)  
**Pre-launch dependency:** Client telemetry remains required before public launch —
see [telemetry-suggestions-export.md](telemetry-suggestions-export.md) and
[backlog-grooming.md](backlog-grooming.md) § Pre-launch must-haves.

## Problem

M3 maps research actions onto the **existing** tool catalog only. When policy
repeatedly emits `no_capability` / suggestions, the OS needs a way to **register
new tools** so Conductor can expand without code forks.

## Shipped / in progress

| Piece | Status |
|-------|--------|
| Structured suggestion context + allowlist refresh each loop | Done |
| Local `os_capability_gaps` ledger + decisions audit | Done |
| `research tools list` / `gaps` / `export-gaps` | Done |
| Maintainer promote/defer/reject (`LABPILOT_MAINTAINER=1`) | Done |
| Client telemetry → maintainer product feed | **Pre-launch TODO** (telemetry item) |
| Plugin discovery + descriptor risk fields | Later |

## Design answers (summary)

| Question | Answer |
|----------|--------|
| **When to add?** | Recurring gap evidence **or** explicit milestone need; prefer **alias**. |
| **How created?** | Path A/B/C → `ToolDescriptor` via **maintainer PR** into LabPilot. |
| **How track (now)?** | Local ledger + `export-gaps` file. |
| **How track (public)?** | Telemetry must collect redacted gaps from clients before go-live. |
| **Who promotes?** | **Maintainer only** (`LABPILOT_MAINTAINER=1`). |

## Out of scope

- Auto-implementing tools without tests
- End-user promote into the shared catalog
- Going public without telemetry client collection
