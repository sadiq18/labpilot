# Design — Campaigns

Back to [../README.md](../README.md) · Milestone: [../milestones/03-campaigns/](../milestones/03-campaigns/).

**Milestone:** M3 · **Impl branch:** `research-os-m3-campaigns`

---

## Goal

Turn the Conductor kernel into **goal-driven campaigns**: research actions that
map onto existing tools, approval ladder (0/1), stop conditions, and checkpoint CLI.

---

## Dynamic research actions (compose onto tools)

Conductor proposes a **research action** (intent). A mapper expands it to a chain
of **registered** tools only, e.g.:

```text
"Investigate augmentation for minority classes"
  → search_papers → generate_plan → run_plan → reflect
```

Not allowed: inventing tools; orphan tasks with no execution path.

Missing capabilities → suggestion log + `no_capability` metrics
([capability backlog](../backlog/capability-registration.md)).
Later: metrics → OTel/Phoenix/Langfuse and suggestions → S3
([telemetry backlog](../backlog/telemetry-suggestions-export.md));
shared org/team/user tables
([tenancy backlog](../backlog/shared-multi-tenant-store.md)).

---

## Autonomy ladder (M3 ships 0–1 only)

| Level | Gates |
|-------|-------|
| 0 (default) | Pause before `generate_plan` + submit family |
| 1 | Pause before submit family only |
| 2 | Deferred (M4/M5) — budget/policy-change pauses |
| 3 | Deferred (M6+) — trusted full autonomy |

Submit/submit_learn are **always** gated (even at level 1).

---

## Stop conditions

- Goal metric reached (e.g. LB ≥ target)
- Submission / wall-time / $ budget exhausted
- Plateau policy (N experiments, no gain)
- Operator pause / cancel (manual)

---

## Goal CLI

```text
research conduct run "<goal>" [--autonomy 0|1] …
research conduct continue | pause | resume | status [--session S-xxx]
```

`resume` defaults to the latest active session for the competition.

Runtime remains **sync** in M3; asyncio workers are M4/M5 scope.

---

## Non-goals

- New tool registration (backlog)
- Parallel multi-agent trees (M5)
- Cross-competition transfer (M6)
- Full context engine (M4)
