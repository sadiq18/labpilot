# M26 — A feature is a claim its code must honour

**Status:** not started · **Blocked by:** [M25](20-eda-findings.md) (a feature with no finding behind it is a guess)

---

## Purpose

`future-specialists.md` deferred a Feature Engineering specialist with an explicit
promotion condition:

> *"Feature decisions frequently wrong or thrash without a dedicated loop"*

rogii met it. Six experiments produced 31 engineered features — `MD_x_GR`,
`GR__roll_mean_5`, `rel_pos_sin_0`, `kriged_TVT_input` — and the pipeline scored
worse than predicting a constant. Two of those six experiments returned
**byte-identical metrics** to their parent, for different hypotheses.

Nothing in the system can say which of the 31 features helped, because nothing
knows the 31 features exist.

## What the loop needs, and what it already has

A feature proposal should carry five things. Three already have a home:

| | Status |
|---|---|
| the idea | exists, but at *technique* granularity, not *feature* granularity |
| why it might help | **exists** — the hypothesis triad, threaded into codegen |
| expected cost | **nothing.** Fields exist with SQL columns behind them, no writer, no reader |
| cheap validation | **nothing** — but 80% shaped and 0% wired |
| keep / reject | **exists**, and is the most mature machinery in the repo |

## The propose-and-verify loop is already built

`DeltaBriefAgent` → codegen → `check_delta_consistency` is exactly the shape this
milestone needs, and it is shipped.

Critically, `DeltaBrief`'s claim vocabulary is **code identifiers** — `kept`,
`added`, `combined`. That maps precisely onto derived feature names:
`GR__roll_mean_5` is a symbol `check_addition` can verify. `feature_engineering`
is not. So a spec that names the identifiers its edit will introduce gets
verification for free.

The ordering rule already encoded there is the one that matters: **the brief runs
before the edit**, so the claim is independent of the diff. Deriving the claim
from the diff, or asking the writer what it did, is circular.

Also free:

- **`check_redundancy`** already answers *"does the pipeline already compute
  this?"* deterministically, over the *live* parent — so dead code from a failed
  earlier experiment does not count as implemented. Run it before proposing.
- **`check_confinement`** (limit: 5 functions) is precisely the "a second change
  rode along" detector, and it is what makes per-feature attribution honest.
- **Keep/reject** — `_decide` → `EvidenceDecision`, `attribute_techniques` →
  `technique_attribution`, `derive_technique_status` →
  `confirmed`/`rejected`/`dormant`, all *recomputed* from evidence cards rather
  than stepped.

So the milestone is not "build a feature engineering agent". It is: **give the
existing propose-and-verify loop a proposal worth verifying, and give it a unit
smaller than a technique.**

## The name the agent must never mint

`feature_engineering` is the system's own marker for *"I failed to name this."*

It is the miner's terminal fallback when no known pattern matches. It is listed
under **Useless** claims in `delta_brief_system.md`. `test_technique_registry.py`
asserts it stays out of the executable registry, commented *"the miner's
catch-all, a category not a method"*. An earlier regex minted techniques called
`the`, `add`, `built`, `average` and `context`, and the Conductor asked codegen to
implement `the`.

A feature spec names identifiers. Never categories.

## The two things that do not exist

### Expected cost

`ResearchTask.estimated_cost` and `estimated_time` exist, with SQL columns behind
them in the plan store, and **nothing writes them and nothing reads them**.
`ExpectedOutcomes.runtime` is hardcoded `None`. The evidence card has a slot for
expected runtime and it is always empty.

The nearest live thing is `EffortEstimate` feeding `hypothesis/ranking.py`, but
its values are hardcoded per candidate *kind* — they are not estimated from
anything about the change being proposed.

### Cheap validation — 80% shaped, 0% wired

This is smaller than it looks, because both halves already exist:

- **`LABPILOT_SMOKE=1` is set by the smoke gate and read by nothing.** A search
  finds exactly one hit: the `env=` that sets it. Generated code is never told it
  exists, and `code_engineer_system.md` never mentions it. Teaching codegen to
  subsample when it is set turns the existing 120-second "does it run" gate into
  a real "is it worth it" probe.
- **`_augmentation_template` already has the `train_smoke → compare → train_full`
  shape**, and `train_full`'s description reads *"Gated: continue to full training
  only if the comparison shows improvement (see plan success_criteria)."* But
  `success_criteria` is read by **nothing** in `execution/` or `conductor/`, so
  `train_full` runs unconditionally.

And the template that would use it is not the one FE hypotheses reach:
`select_template` matches feature-engineering keywords **first**, so every FE
hypothesis routes to `_feature_engineering_template` — one training run, no gate.

Wiring those two is the cheap-experiment path, and it is a prerequisite for
"expected cost" to mean anything.

## Ablation is proposed but never run

`maybe_mint_ablation_from_combo_win` mints one leave-one-out hypothesis per combo
member — *"Combination H-003 gained 0.4649; ablate by dropping `dataset`"*. There
is no ablation executor, no ablation template, no ablation runtime. Those
hypotheses go back into the normal queue and become **normal full-cost training
runs**.

So the system proposes ablations it cannot afford to run. The cheap-validation
path above is what would make them affordable, which is why the two belong in the
same milestone.

## Exit criteria

1. A `FeatureSpec` names the identifiers its edit will introduce, and
   `check_addition` confirms the applied code introduced them.
2. Every spec cites the finding that motivated it. A feature with no finding
   behind it is not proposed.
3. Gain is attributed **per feature group**, never per blob — "added 40 features,
   score moved" is not evidence about any one of them, and `check_confinement`
   enforces it.
4. A feature the pipeline already computes is rejected before an experiment is
   spent on it, via `check_redundancy`.
5. A cheap probe runs on a subsample and its result decides whether the full run
   happens — `success_criteria` stops being prose.
6. A rejected feature is not re-proposed next campaign. **Note this needs a
   deliberate decision**: `persist.py:98` records that `rejected` and
   `inconclusive` hypotheses stay eligible *by design*, so this is a change to
   existing behaviour rather than a gap to fill.
7. `feature_engineering` never appears as a proposed technique name.

## Traps

- **Do not let the agent write feature code.** It reintroduces a second code
  writer alongside codegen, with exactly the attribution problem that killed the
  Jinja pack in M19 §2. Specs go through the existing delta pipeline.
- **Do not build a feature registry yet.** "Which features exist" genuinely has
  no home — `_selected_columns` reads *input* columns and only for leakage;
  `FeatureRecipe` has the right schema and is only ever populated from mined
  literature; `TechniqueSpec.feature_recipes` points at a `CodeRenderer` deleted
  in M19 §2. It is a tempting fifth thing to build, it is not required for the
  loop to close, and `check_addition` over identifiers covers the verification
  need.
- **Do not propose 500 features.** The unit of value is a feature *group* with a
  finding behind it and a measured incremental gain. A blob of 40 cannot be
  attributed, and `check_confinement` will flag it as a wide delta.
- **`applied` is a verdict about preconditions, not about code.** `resolver.py`
  says so: it is stamped before codegen runs, from the dataset profile alone, and
  *"nothing downstream compares `feature_recipes` against what the delta actually
  computed."* A spec that is `applied` has not been verified.
