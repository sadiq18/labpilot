# Design — Workspace facade

Back to [../README.md](../README.md) · Milestone: [../milestones/01-foundation/](../milestones/01-foundation/).

**Milestone:** M1 · **Impl branch:** `research-os-m1-foundation`  
**Related V1:** [competition-workspace](../../research-pipeline/design/competition-workspace.md)

---

## Goal

One **Workspace** object every tool receives (with a Task/Plan handle), instead of
ad-hoc paths and flags.

```text
Workspace
  goal?
  root / competition slug
  files, git, env, logs
  datasets, models, notebooks
  artifact roots (knowledge/, pipeline/, …)
  memory handles (db paths, query ports)
```

---

## Relationship to competition workspace

V1 already has client-owned competition workspaces (`labpilot.yaml`, data/, …).
The OS Workspace is a **facade** over that layout + knowledge tree — not a second
directory convention.

| Concern | Facade provides |
|---------|-----------------|
| Paths | Stable accessors (`data_dir`, `pipeline_dir`, `knowledge_dir`) |
| Goal | Optional string / Objective ref (filled by Conductor M2+) |
| Artifacts | Resolve ids → paths / DB rows |
| Git / logs | Optional handles; may be stubs in M1 |

---

## Unit of work

Everything operates on **Workspace**, not a bare `competition_slug` as the primary
API. Slug remains an identifier *inside* the workspace.

From M2 onward (M4 adds context):

```text
tool.run(workspace: Workspace, task: Task, context: ContextBundle | None) -> ArtifactRefs
```

M1 may pass `workspace` + existing plan/execution ids until Task exists.

---

## Rules

1. Tools must not hard-code repo-relative `competitions/<slug>` assumptions when a
   Workspace is provided.
2. Side effects stay inside workspace roots (or explicitly declared global caches).
3. Workspace construction is shared by CLI and (later) Conductor / IDE clients.
4. Checkpointing (M2+) snapshots workspace-relevant state; `research continue`
   restores it ([05-tasks](05-tasks.md), [06-campaigns](06-campaigns.md)).

---

## Non-goals

- Full IDE/terminal multiplexing in M1
- Replacing `labpilot.yaml` schema
- Remote workspace sync

---

## Acceptance (when implementing)

- `Workspace` type constructed from competition root / slug
- At least analyze + plan + run obtain paths via Workspace
- Unit tests for path resolution for client-owned and legacy layouts
