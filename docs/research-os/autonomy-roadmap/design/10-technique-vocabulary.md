# Design — M-25: the technique vocabulary earns its entries

**Status:** design · **Owner:** unassigned · **Depends on:** [§8.7](01-technique-to-model.md) ·
**Blocks:** #26 (candidate adjudication)

---

## 1. Problem

Three things currently answer "what is a technique?", and none of them is right.

| Source | What it actually knows | Failure observed on rogii |
|---|---|---|
| `technique/registry.py` | which techniques a **template gate** can run | bounded by how many gates someone wrote — correct for its job, useless as vocabulary |
| `feature_recipes.py` miner | regexes over prose | minted `the`, `add`, `built`, `computed`, `average`, `context`, `model`, `neighbour`, `tangent`, `booster` |
| `KnowledgeHub` | how often a paper **mentions** a phrase | `3D garment modeling`, `Breath Focus practice`, `Radiomics` — from retrieved papers about other fields |

The `techniques` table holds **116 rows** on rogii. It has no status column, so
every row is equally real. `beliefs` holds **124**. Downstream, all of them are
promotable, plannable, and implementable: a live campaign asked the code engineer
to implement `the`.

The registry's own docstring already states where this should live:

> That vocabulary lives in the `techniques` store with a confirmed/candidate/
> rejected status, because it grows by learning.

That store exists. The status does not.

### Why not just curate a list

Rejected in §8.7 and the reasoning still holds: a closed list answers an
open-world question, and the techniques worth finding are the ones not on it. The
problem is not that the vocabulary is open — it is that **entry is free**. A
phrase appearing in a retrieved PDF costs nothing to become a first-class
technique with a belief, a confidence, and a claim.

---

## 2. Requirements

**Functional**

1. Every technique has a **status**, and status is derived, not authored.
2. A technique enters as `candidate` and can only become `confirmed` by **measured
   evidence** — the same bar `ClaimPromoter` already enforces for claims.
3. Consumers filter by status. Planning and implementation see `confirmed` plus
   `candidate`; belief promotion and claims see `confirmed` only.
4. Status is **recomputed** from current evidence, not stepped — the same lesson
   as `belief_repair`: a step cannot be un-done when the evidence behind it is
   corrected.
5. Demotion is possible and never destructive. A `rejected` technique keeps its
   row and its reason.

**Non-functional**

- Recompute for a full workspace in **< 2s** at 10× current scale (1,160
  techniques, 500 evidence cards). It runs at campaign start, next to
  `repair_card_directions`.
- No network, no LLM. Adjudication by LLM is #26 and sits on top of this.

---

## 3. Success metrics

Measured on rogii, before → after:

| Metric | Now | Target |
|---|---|---|
| Techniques visible to planning | 116 | ≤ 25 |
| Techniques with any measured evidence | 5 | unchanged (5) — this is not about creating evidence |
| Junk identities (`the`, `Breath Focus practice`) reachable by the planner | yes | **0** |
| Techniques the Conductor can propose that have never been measured | 116 | only those with a `candidate` justification |

The point is not fewer techniques. It is that **position in the vocabulary
reflects what we know**, so the Conductor's attention goes somewhere defensible.

---

## 4. Design

### Lifecycle

```
                    mined / cited / LLM-proposed
                              │
                              ▼
                        ┌───────────┐
      no evidence,      │ candidate │◄──── demoted when its evidence
      never run   ┌─────┤           │      is retired or contested
                  │     └─────┬─────┘
                  ▼           │ measured effect on ≥1 conclusive card
            ┌──────────┐      ▼
            │ dormant  │  ┌───────────┐
            └──────────┘  │ confirmed │
                          └─────┬─────┘
                                │ measured, and the effect is adverse
                                ▼
                          ┌──────────┐
                          │ rejected │   (kept, with a reason)
                          └──────────┘
```

Four statuses, each with exactly one derivation rule:

| Status | Rule | Who may see it |
|---|---|---|
| `candidate` | proposed by any source, no conclusive card yet | planner, code engineer |
| `confirmed` | ≥1 conclusive evidence card attributes a non-zero effect | everyone |
| `rejected` | measured, and the net effect is adverse beyond noise | nobody proposes it; retained for the ledger |
| `dormant` | proposed ≥ N campaigns ago, never selected, never measured | nobody, unless explicitly listed |

`dormant` is the one that does the cleanup work. `3D garment modeling` is not
*wrong* — it is a real technique, in another field — so `rejected` would be a lie.
It has simply never been chosen, and never will be, and it should stop competing
for the planner's attention without the system claiming to have disproved it.

### Where it plugs in

Recompute runs at campaign start, in the existing repair chain — order matters
and is already established:

```
repair_card_directions      # cards first: verdicts are the input
rederive_beliefs_from_cards # beliefs from repaired cards
recompute_technique_status  # ← new, from the same cards
revalidate_outcome_claims   # claims last, reads all of the above
```

Same principle as the existing repairs: **correct what is recorded before acting
on it**, and derive from the current card set so the result is idempotent.

---

## 5. Schema

One column and one table. The column carries the answer; the table carries why.

```sql
ALTER TABLE techniques ADD COLUMN status TEXT NOT NULL DEFAULT 'candidate';
CREATE INDEX idx_techniques_status ON techniques(status);

-- Why a technique holds its status, and what would change it. Append-only:
-- the history of what the system believed about its own vocabulary is itself
-- research evidence, and the same "contested, never deleted" rule that governs
-- claims applies here.
CREATE TABLE technique_status_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    technique_id    TEXT NOT NULL,
    competition_slug TEXT NOT NULL DEFAULT '',
    from_status     TEXT,
    to_status       TEXT NOT NULL,
    reason          TEXT NOT NULL,
    evidence_card_id TEXT,
    observations    INTEGER NOT NULL DEFAULT 0,
    net_effect      REAL,
    created_at      TEXT NOT NULL
);
```

`observations` / `net_effect` mirror what `ClaimPromoter.measured_effect` already
returns, so the promotion bar is literally the same computation — not a second
one that can drift from it. That drift is a defect this repo has hit repeatedly
(three definitions of "runnable plan", two of "belief identity").

---

## 6. Components

| Component | Responsibility | Reuses |
|---|---|---|
| `technique/vocabulary.py` (new) | derive status from cards; write history | `ClaimPromoter.measured_effect` |
| `TechniqueStore` (extend) | `list(status=…)`, `set_status(…, reason=…)` | existing `techniques` table |
| `candidates.py` (change) | filter proposals by status instead of the hardcoded modality token lists | — |
| `loop.py` (change) | one call in the repair chain | — |

The modality filter is folded in here deliberately. It is the same defect in a
different costume: `cnn` is blocked on every tabular problem by a hardcoded token
list, including geosteering, where convolution over a depth sequence is standard
practice. Applicability should come from the dataset profile the system already
computes, not from a tuple. Once status exists, "is this applicable?" becomes a
`candidate` justification rather than a hard block.

---

## 7. Tradeoffs

| Decision | Alternative | Chosen | Why |
|---|---|---|---|
| Status derived from cards | Authored by an LLM at ingest | Derived | An LLM judgement is a claim, and claims need measurement. #26 adds LLM *adjudication of candidates*, which is a different question from *what has this done here* |
| `dormant` as a distinct status | Delete unused rows | `dormant` | Deleting loses the fact that we saw it. A technique dormant in one competition may be central in the next, and the memory is cross-competition |
| Recompute, not step | Increment on each card | Recompute | Learned twice already: `apply_card_to_beliefs` stepped, and repairing a card afterwards changed nothing |
| Status on the existing table | New `vocabulary` table | Existing | 116 rows already carry name/category/domain/summary. A parallel table would immediately raise "which one is authoritative", which is exactly the `belief_tech_*` vs `belief:comp:*` bug |
| Confirmed requires *any* non-zero effect | Requires a *positive* effect | Any non-zero | Direction depends on the metric, and this repo has now been bitten four times by assuming it. A technique measured to hurt is confirmed *and* rejected — both are knowledge |

---

## 8. Testing

The failure modes here are all "the filter looked protective and was not", which
this repo has hit five times. So the tests are specific rather than exemplary:

1. **The junk cannot reach the planner.** Seed `the`, `Breath Focus practice`,
   `3D garment modeling`; assert none appears in what `candidates.py` offers.
2. **Measured techniques survive.** `SWA` (obs=1, net −3.83 on MSE) must be
   `confirmed`, not filtered out with the junk. This is the test that fails if
   the rule is "drop anything unusual".
3. **Recompute is idempotent.** Twice in a row changes nothing the second time.
4. **Retiring a card demotes.** The `vit` path: card becomes `inconclusive`,
   technique returns to `candidate`, not `confirmed`.
5. **A zero-observation technique is never `confirmed`,** whatever its citation
   count — the `belief_tech_vit`-at-0.95 failure in vocabulary form.
6. **History is append-only** and survives a demotion.

Validation is against the rogii store, on a **sandbox copy**, before anything
runs against the workspace — the standing rule for this system.

---

## 9. Rollout

Additive: the column defaults to `candidate`, so an un-migrated workspace behaves
as today. Recompute is a no-op without evidence cards.

Ship in two steps, because step 1 is observable and reversible on its own:

1. **Schema + recompute + a report.** Nothing filters yet. Run it on rogii and
   read what it *would* do. If it proposes demoting `SWA`, the rule is wrong and
   nothing has been lost finding out.
2. **Consumers filter by status.** Only after step 1's output has been read.

Rollback is dropping the filter in step 2; the column and history are inert data.

**The risk worth naming:** an over-strict rule silently narrows the search space,
and a research system that stops proposing novel techniques fails in a way that
looks like working. That is why step 1 reports before step 2 filters, and why
metric 2 in §3 is *unchanged evidence count* — this design must not be able to
make the system look better by making it smaller.
