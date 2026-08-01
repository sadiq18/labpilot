# Backlog — Parallel research branches

**Status:** Backlog (post-M5). M5 ships **thin** asyncio workers only.

## Problem

Full parallel research campaigns need branch state, merge policy, evidence
comparison, resource allocation, and cancellation/retry — closer to Campaign
Engine v2 than M5 specialist runtime.

## M5 already ships (do not redo)

- Max workers + shared budget
- `asyncio.gather` / AnyIO for independent **experiment tasks**
- Collect results; no autonomous branch tree

## Proposed later work

```text
Branch A: Try CNN
Branch B: Try Transformer
Branch C: Search papers
        ↓
Evidence merge
        ↓
Conductor chooses winner
```

Requirements when picked up:

- Independent research branches with durable branch state
- Merge / compare policy and conflict handling
- Budget and cancellation across the tree
- Conductor-owned winner selection (agents still do not call each other)

## Out of scope here

Distributed multi-machine orchestration (see also [async-conductor](async-conductor.md)).
