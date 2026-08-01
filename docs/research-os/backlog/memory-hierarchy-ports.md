# Backlog — Public memory hierarchy ports

**Status:** Backlog (not M4 public API). M4 may use tiers **internally** inside
the Context Engine; do not expose them as the main product surface.

## Problem

Design ([10-memory-os](../design/10-memory-os.md)) describes Short / Working /
Long / Episodic / Semantic tiers. Exposing them early couples Conductor, agents,
and CLI to storage layout.

## Proposed later work

- Explicit ports per tier (query/store) with TTLs
- Migration of internal Context Engine sources onto those ports
- Docs + examples for operators inspecting tier contents

## Out of scope here

Shipping hierarchy as M4’s primary API — Context Engine hides implementation.
