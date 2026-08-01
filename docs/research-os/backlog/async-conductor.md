# Backlog — Async Conductor / distributed scheduler

**Status:** Backlog. **Pull this item** when implementing long-running multi-campaign
orchestration (not during M5 specialist runtime).

## Problem

M5 keeps Conductor **sync** and spawns async workers underneath. That is enough for
limited parallel experiments. Multi-campaign, multi-host, GPU pools need an async
control plane and distributed scheduling.

## M5 already ships

```text
Conductor (sync) → Task Queue → Async workers (Impl / Experiment / …)
```

## Proposed later work

```text
Async Conductor
    |
Distributed scheduler
    |
Remote workers / GPU clusters
```

- Async decision loop only when sync facade becomes a bottleneck
- Remote worker protocol; Ray/K8s as needed (not gates for earlier milestones)
- Preserve: strategy and approval stay Conductor-owned; no peer agent control flow

## Pickup trigger

Long-running **multi-campaign** orchestration across processes/hosts — not “we have
asyncio workers.”
