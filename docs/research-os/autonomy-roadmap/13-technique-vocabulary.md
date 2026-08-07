# M18 — The technique vocabulary earns its entries

**Status:** shipped 2026-08-07 (PR #100 step 1, PR #101 step 2) · **Design:**
[design/13-technique-vocabulary.md](design/13-technique-vocabulary.md) ·
**Blocks:** [M19](14-experiments-as-deltas.md), candidate adjudication

---

## Purpose

Three things currently answer "what is a technique?", and none of them is right.

| Source | What it actually knows |
|---|---|
| `technique/registry.py` | which techniques a **template gate** can run |
| `feature_recipes.py` miner | regexes over prose |
| `KnowledgeHub` | how often a paper **mentions** a phrase |

Measured on rogii 2026-08-07: the `techniques` table holds **116 rows** with no
status column, so every row is equally real. Downstream they are all promotable,
plannable and implementable — a live campaign asked the code engineer to
implement a technique called **`the`**, mined from "we added the features to the
model". Alongside it sit `3D garment modeling` and `Breath Focus practice`,
harvested from papers about other fields entirely.

## Why not a curated list

Rejected in [§8.7](01-technique-to-model.md) and the reasoning still holds: a
closed list answers an open-world question, and the techniques worth finding are
the ones not on it. The problem is not that the vocabulary is open — it is that
**entry is free**. A phrase appearing in a retrieved PDF costs nothing to become
a first-class technique with a belief, a confidence and a claim.

## What changes

Status is **derived from evidence, never authored**: `candidate` on entry,
`confirmed` only on a measured effect, `rejected` when measured and adverse,
`dormant` when proposed but never selected. Recomputed from the current card set
in the existing repair chain, so it stays correct after a card is repaired — the
same recompute-don't-step rule that `belief_repair` needed.

## Exit criteria

1. No technique reaches the planner without a status. ✅ (schema v11)
2. Status is recomputed, not stepped, and is idempotent. ✅
3. Promotion to a claim requires `confirmed`. ✅
4. The junk (`the`, `Breath Focus practice`) is unreachable by the planner,
   **and `SWA` — the one measured improvement — still is.** ✅

## Status

**Step 1** (PR #100): `techniques.status`, `technique_status_history`, recompute
in the conductor repair chain, `research techniques report`.

**Step 2** (PR #101): consumers filter on status — `generate_candidates`,
`SymbolicFetcher`, claim promotion.

### What the review of the first attempt caught

Step 2 was originally bundled into step 1, and the consumer filters made
`dormant` a **closed loop**: a technique leaves `dormant` only by appearing on a
hypothesis, and the filters removed the only two paths that could put it there.
Measured on rogii, planner-visible dropped **116 → 1**. Every technique the miner
ever learned would have been permanently excluded — the failure §9 of the design
names, where a research system stops proposing novel work and looks healthy doing
it.

The design's step-1 review gate exists to catch exactly that, and skipping it is
what let it through. The rule now ages: a freshly mined technique stays
`candidate` and visible, verified on a sandbox copy —

```
merge_technique("gradient_boosting_dart") -> candidate
recompute_technique_status()              -> candidate   planner-visible: True
```
