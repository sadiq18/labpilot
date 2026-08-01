# Backlog — CodingTool adapters

**Status:** Backlog (not M5). M5 ships `CodingTool` → **V1 Code Engineering** only.

## Problem

M5 proves the **interface boundary**, not coding-engine swap. Claude Code / Aider /
OpenHands adapters add integration cost before the specialist runtime is proven.

## M5 already ships

```text
Implementation Specialist → CodingTool → V1 Code Engineering
```

## Proposed later work

```text
CodingTool
    ├── V1 Code Engineering (default)
    ├── Claude Code
    ├── Aider
    ├── OpenHands
    └── Custom Research Coder
```

- One backend selectable via config / capability
- Same `implement(task, context) -> Artifact` contract
- No Conductor or specialist changes when swapping engines

## Out of scope here

Rebuilding a LabPilot coding agent as the differentiator (architecture: reuse data plane).
