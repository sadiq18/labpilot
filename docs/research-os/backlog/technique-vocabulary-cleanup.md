# Backlog — the vocabulary rows already written

**Status:** Backlog · **Kind:** data repair in existing workspaces ·
**Found:** PR #112, 2026-08-08 · **Related:**
[M18 technique vocabulary](../autonomy-roadmap/13-technique-vocabulary.md) ·
[`shared/labels.py`](../../../src/labpilot/research_engine/shared/labels.py)

PR #112 guarded the **writers**. It did not clean what earlier runs already
wrote, because that edits the user's workspace, which is their call.

## Problem

`techniques.name` is the technique vocabulary. Before #112 it had four read-side
guards and **zero write-side guards**, so record references and regex-miner
fragments accumulated in it. Filtering readers cannot shrink a table that keeps
being written to.

#112 closed both write sites — `merge_technique` now raises rather than skipping
silently, and `_techniques_from_plan` filters what it appends. New junk stops
arriving. The existing rows stay.

## Measured 2026-08-08

Two real workspaces:

| workspace | `techniques` rows | record references (`hyp:` / `fork:`) |
|---|---|---|
| `rogii-wellbore-geology-prediction` | 116 | **12** |
| `rogii-runA-baseline-20260807-003705` | 70 | **5** |

`labels.py` measured 5 in the main workspace before it existed. It is now 12 —
the growth this backlog item exists to reverse, and the direct evidence that
read-side filtering alone does not hold a table steady.

The record references are the *mechanically identifiable* class. The rest of the
junk is not one thing:

| class | examples | identifiable by rule? |
|---|---|---|
| record references | `hyp:H-010`, `hyp:H-BASELINE` | **yes** — `is_record_reference` |
| prose fragments from the recipe miner | `the`, `test`, `add`, `built`, `computed`, `context`, `tangent`, `neighbour` | no |
| categories, not techniques | `feature_engineering`, `preprocessing`, `clustering`, `model evaluation` | no |
| off-domain rows from paper mining | `3D garment modeling`, `Breath Focus practice`, `Focused Attention Meditation (FAM)`, `Radiomics` | no |

The miner's prose fallback was removed in `500eb75` (2026-08-07), so the first
class stopped growing. The replacement fallback returns `feature_engineering` — a
category — which means the second class is still being written today. That is a
separate defect from this one and belongs with the miner, not the migration.

## The design constraint that decides the shape

**A migration may only delete what a rule can identify.** That is the 12 record
references, and nothing else.

Deleting `the` and `test` needs a hand-written list of bad names, and deciding
`Radiomics` is off-domain for a wellbore competition needs judgement about the
domain. Both are the curated-set-answering-an-open-world-question pattern this
codebase has now rejected four times — `KNOWN_TECHNIQUES`, the package
allowlist, the template pack, and the technique→symbol map that killed the
original 1b plan. Doing it here would be the fifth.

So: **delete the identifiable class, report the rest.** A count of suspicious
rows the user can act on beats a cleanup that quietly removes a real technique
because it happened to be one word long.

## Where it should live

Not a one-off script. [`evidence/repair.py`](../../../src/labpilot/research_engine/evidence/repair.py)
already sets the pattern, and states the reason:

> Repair runs from the campaign, not as a one-off migration script the user must
> remember. […] a workspace whose profile is fixed later heals on the next run
> without anyone editing stored artifacts by hand.

Same principle as `ClaimPromoter.revalidate_claims`: correct what is recorded
before adding to it.

The schema migrator at
[`accessor/sqlite/migrate.py`](../../../src/labpilot/accessor/sqlite/migrate.py)
is the wrong home — it is `CREATE TABLE IF NOT EXISTS` DDL, deliberately, and
`SCHEMA_VERSION` tracks structure. This is row content, not structure.

## Impact if left alone

**Lower than it first appears, and the reason is M18.** Stated plainly so this
is not picked up ahead of work that matters more.

- **Reads are already filtered** at all four `is_record_reference` sites, so a
  record reference does not reach planning or attribution today.
- **The prose fragments are covered too — by status, not by name.** M18 exit
  criterion 4 pins exactly this (`the`, `Breath Focus practice` unreachable;
  `SWA` still reachable), and `derive_technique_status` is what delivers it:
  never measured **and** never selected **and** at least
  `DORMANT_AFTER_CAMPAIGNS = 2` campaigns old ⇒ `dormant`, which is outside
  `PLANNER_VISIBLE_STATUSES`.

So the junk is mostly inert. What is left is narrower:

| still real | why |
|---|---|
| **A record reference ages to `dormant`, which mislabels it** | `dormant` means "a technique nobody tried". `hyp:H-010` is not a technique at all. The status machinery gives a wrong reason for the right outcome |
| **Counts are inflated** | vocabulary-size reporting and frequency ranking see 116 rows, 12 of which are not techniques at all — and more that aging hides rather than removes |
| **New junk is visible for up to 2 campaigns** | aging needs `age >= 2`, so a freshly mined `feature_engineering` is `candidate` — and planner-visible — until then |

The third is **not fixed by a migration** — it is the miner's replacement
fallback still writing categories, upstream of this. Worth splitting out rather
than folding in here.

Both remaining rogii workspaces predate schema v11 (`techniques` has no `status`
column), so they get the column on next open and their statuses on the next
recompute. That heals the visibility question without deleting anything, which
is a further argument for keeping this item small: **delete the 12 record
references, report the rest, leave aging to do its job.**

## Open question

Whether repair should run per-competition on campaign start (matching
`repair_card_directions`) or as an explicit `research doctor` action. The repair
precedent argues for the former; the fact that this deletes rows rather than
recomputing derived fields argues for asking first.
