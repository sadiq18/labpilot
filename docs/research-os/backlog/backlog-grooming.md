# Backlog grooming — recommended pickup order

Back to [README.md](README.md).

**Audience:** post-M6 (Experience Records + `research memory seed|inspect` shipped).  
**Method:** value + dependency order — not the catalog index order in the README.

Catalog of deferred items: [README.md](README.md).

---

## Pre-launch must-haves

Do **not** open LabPilot to the public until these are done:

| # | Capability | Why |
|---|------------|-----|
| ★ | [telemetry-suggestions-export](telemetry-suggestions-export.md) | Collect campaign metrics + redacted `no_capability` gaps from **client** installs into a maintainer-readable feed. Local SQLite / `export-gaps` is not enough for multi-user product evolution. **Blocks go-live.** |

Capability registration may ship locally first; **telemetry is still required before public launch.**

---

## Near-term (compound on M6)

| # | Capability | So what | Impact |
|---|------------|---------|--------|
| 1 | [experience-facet-extraction](experience-facet-extraction.md) (stages 3–5) | Optional embeddings/LLM facets after Stage 2 | Better cross-comp retrieve quality without auto-transfer risk |
| 2 | [capability-registration](capability-registration.md) ([design](../design/11-capability-registration.md)) | Local gap ledger + export + maintainer decisions (telemetry later for public) | Campaigns surface missing tools; maintainers can promote from evidence |
| 3 | [coding-tool-adapters](coding-tool-adapters.md) | Swap V1 Code Engineering for Claude Code / Aider / OpenHands behind `CodingTool` | Stronger implementation quality without rewriting the agent OS |
| 4 | [future-specialists](future-specialists.md) (Critic / Eval / Submit first) | Promote specialists when skill loops are measurable | Less “one mega path”; clearer ownership of review, eval, submit |
| 5 | [git-worktrees-patches](git-worktrees-patches.md) | Parallel experiment dirs + patch review instead of one dirty tree | Safer concurrent experiments; cleaner review/revert |
| 6 | [parallel-research-branches](parallel-research-branches.md) | Real branch-merge research campaigns (beyond thin asyncio) | Explore multiple hypotheses in parallel with merge semantics |

---

## When scale / ops pain shows up

| # | Capability | So what | Impact |
|---|------------|---------|--------|
| 7 | [telemetry-suggestions-export](telemetry-suggestions-export.md) | **Pre-launch ★** — metrics + gaps leave user SQLite → maintainer feed | Ops visibility; required before public |
| 8 | [hybrid-semantic-retrieval](hybrid-semantic-retrieval.md) | Embeddings + ANN when BM25 misses paraphrases | Transfer memory and context recall improve on “same idea, different words” |
| 9 | [automatic-transfer-confidence](automatic-transfer-confidence.md) (M7+) | Auto warm-start with scored confidence (not silent) | Second competition starts warmer **with** auditability |
| 10 | [experience-pattern-extraction](experience-pattern-extraction.md) | Emergent prompt / model / feature / paper patterns from usage | Memory becomes reusable playbooks |
| 11 | [async-conductor](async-conductor.md) | Long-running multi-campaign orchestration | Run several competitions/goals without a single blocking loop |
| 12 | [shared-multi-tenant-store](shared-multi-tenant-store.md) | Shared tables across comps / users / teams | Org-scale memory and campaigns |
| 13 | [git-remote-adapters](git-remote-adapters.md) | GitHub/GitLab behind GitTool | Remote PRs/branches as experiment artifacts |
| 14 | [kuzu-graph-backend](kuzu-graph-backend.md) | Graph-native store behind `GraphPort` | Faster/complex graph queries when SQL graph hurts |
| 15 | [memory-hierarchy-ports](memory-hierarchy-ports.md) | Public Short/Working/Long/Episodic/Semantic APIs | Cleaner agent memory contracts |

---

## How to choose next

| Pain | Pick |
|------|------|
| Retrieval feels weak | `#1` facets → `#8` hybrid **only** after BM25 metrics show gaps |
| Campaigns can’t do enough | `#2` capability registration (local) → `#3` / `#4` coding + specialists |
| Going public soon | **★ telemetry** before launch (non-negotiable) |
| Experiments collide | `#5` worktrees → `#6` parallel branches |
| Warm start still too manual | `#9` auto-transfer (after facets + enough experience corpus) |

Defer `#12–15` until multi-user or real graph/scale pressure — high cost, little solo-researcher upside today.

---

## Non-goals of this note

- Does not reorder or retire catalog entries in [README.md](README.md).
- Does not schedule a milestone; pull items explicitly when pain justifies.
- Does not authorize silent auto-transfer or Conductor bypass from memory.
