# M23 — A model that loses to a constant is not a baseline

**Status:** not started · **Blocked by:** [M22](17-dataset-understanding.md) (the schema must be trustworthy first)

---

## Purpose

Measured on rogii, 2026-08-13, on a split that mimics the real test — first 27.3%
of each well known, predict the tail, 120 wells, 576k rows:

| predictor | RMSE |
|---|---|
| carry forward the last known TVT — one line | **15.10** |
| predict the global mean | 648.6 |
| linear extrapolation from the prefix | 1,199.6 |
| **the pipeline, after six experiments** | **1,380.4** |
| target | 2.236 |

The pipeline is **2× worse than predicting a constant** and 91× worse than one
line of code. And the campaign reported `1789 → 1409 → 1380` as progress for
fourteen steps, because nothing in the system computes what the dumbest possible
answer scores. There was no yardstick, so "improving" and "below chance" were
indistinguishable.

The cost is not the wasted runs. It is what got written down:

- **19 child hypotheses** (H-109 … H-127) whose entire premise is extending a
  result that loses to a constant — *"Parent H-020 gained 1595 on E-244"* — and
  which are now the pool the next campaign selects from.
- **Eight techniques driven to confidence 0.0** — `vit`, `rolling_features`,
  `warmup`, `SWA`, `add`, `dataset`, and `tabular_regression_partitioned` at
  0.196. The system has concluded those do not work here. It cannot know that:
  the experiments could not discriminate, because everything in that regime
  scores ~1400 regardless. Beliefs persist across campaigns.

A wasted run costs a run. A false belief costs every campaign after it.

## The system already half-noticed

`H-BASELINE`'s own reflection reads:

> *"The model failed to learn from the training data, possibly due to an issue
> with the template configuration."*

Correct diagnosis, parked in an `inconclusive` hypothesis where it gates nothing.
`H-BASELINE` has been `proposed` since it was created, because the baseline plan
is **the only template with no `COMPARE` task**, so no evidence card is ever
built for it. Three sites read its status and all three *exclude* it. Nothing
anywhere reads it as a precondition.

Meanwhile the trigger that flips a campaign from baseline to research is
`_baseline_plan_exists` — *"any plan whose `metadata["plan_kind"] == 'baseline'`"*.
**Plan existence.** Not plan success, not run success, not score. The moment
`P-001` is compiled the campaign starts minting improvement hypotheses, whether
or not it ever ran.

## Why this is not the deleted template pack

M19 §2 deleted the Jinja baseline pack for a good reason, recorded at
`code_engineering/capability.py:609`:

> *"A rendered template is a baseline, not the experiment the hypothesis asked
> for, and it was recorded as a successful step: twelve distinct hypotheses once
> scored MSE 194.80 identically because each got the same rendered file. The run
> looked healthy and tested nothing."*

The objection is not "deterministic code is bad". It is that **a rendered
artifact stood in for an experiment and was recorded as that experiment's
success**. The floor is the structural opposite:

| | deleted Jinja pack | Baseline 0 (the floor) |
|---|---|---|
| writes `pipeline/train.py` | yes | **never** |
| attached to a hypothesis | yes | **never** — carries no `hypothesis_id` |
| recorded as an execution | yes, `succeeded` | **never** — no `E-` id |
| one per | hypothesis | **competition** |
| identical values across reads | the symptom | the **contract** |
| answers | "what did this experiment do?" | "what does this dataset score with no model?" |

The floor is a **reading of the dataset**, in the same family as `profile.json`.
Neither can be mistaken for an experiment, because **neither produces code**.

That is also why **Baseline 1 stays LLM codegen**. A deterministic `train.py`
writer is the pack returning, and "fallback only" is the pack at the same call
site with better code. The failure was never code quality.

## Baseline 0 — the floor

- Folds come from the **same `ValidationPlan`** the real model is told to use,
  read from `baseline_choice.json` so the two cannot drift. A floor computed on a
  different split is not a floor.
- The constant is fitted **per fold, on the train side only**. Fitting on the
  whole target is the leakage version and looks unbeatable on skewed data.
- Scored by the existing `execution/metrics.py:compute_metric` — never a second
  implementation — and written under `cv_{metric_name}`, which is the detail that
  makes the comparison work for free.
- Every strategy is tried and recorded; the floor is the **best** trivial
  predictor. One that picked the worse constant is a gate too easy to pass.

The optimal constant depends on the metric, and getting that right is what makes
the floor honest: mse/rmse → mean · mae → **median** · rmsle →
`expm1(mean(log1p(y)))` · accuracy/f1 → majority · logloss → the **prior vector**,
not the argmax · auc → **exactly 0.5, asserted analytically** rather than
computed.

**The floor is determined by the shape of the prediction target, not the modality
of the input.** An image competition's label is still a class column; its floor is
the class prior, identical to tabular. Input modality determines only whether
Baseline 1 is affordable.

Floors are **undefined** for detection, segmentation, keypoints, generation and
RL. Their honest floors — predict-empty, random policy, the provided sample agent
— all require executing something in the competition's own harness. That is an
environment runner — a capability that does not exist and is deliberately not
scoped here — and pretending the profiler can
produce one would be a gate testing something easier than it promises. Undefined
means `floor_undefined`, which **blocks**, with a recorded waiver as the escape.

## The gate

Nine states, each with a distinct operator action — against `_baseline_plan_exists`,
which collapsed all of it into one boolean about a plan's existence:

`unknown` · `floor_missing` · `floor_undefined` · `blocked_uncertain` ·
`awaiting_ml` · `stale` · `failed` · `passed` · `waived`

`stale` matters as much as `failed`: a fingerprint over validation scheme, target,
metric and `profile.schema_version` means a re-derived profile invalidates
readings that described a different setup.

The gate **deliberately does not read `H-BASELINE.status`** — five layers of
derivation, one of which raises. A bookkeeping fault would read as "baseline not
passed". It reads its own record, written by one writer. H-BASELINE finally
getting a status is a valuable *consequence*, cross-checked in tests, never a
dependency.

**Only `submit` and `submit_learn` gain the gate condition.** Untouched:
`analyze_competition`, `generate_plan`, `run_plan`, `run_experiment`, `implement`,
`reflect`, `query_memory` — those are how a campaign *builds* the baseline.
Gating `run_plan` would be a bug: it is how the gate gets opened.

Hypothesis minting is not a tool, so `available_tools` cannot reach it. It is
blocked at `_hypothesize_run` and at `persist_recommendations` — the only durable
write, which also covers the CLI path.

## The failure report names only what fired

Every detector reads an artifact. No LLM on this path.

```
🚨 BASELINE FAILURE — rogii
Dummy baseline (carry_forward)   RMSE   15.10
Generic ML     (lightgbm)        RMSE 1380.42
Improvement                      −9042.5%   ✗ worse than a constant

Observed (facts read from artifacts):
  leakage/ID handling   profile.anchor_column='TVT_input' equals the target
                        wherever present; pipeline/train.py never mentions it.
  validation mismatch   plan declares partition_suffix_holdout; no function in
                        train.py performs it.

Not ruled out: target identification · preprocessing · metric mismatch
Do not proceed to research.
```

A list that prints identically on every failure is a list nobody reads — this
repository has paid for that twice, in `check_confinement` (*"a flag on everything
is a flag nobody reads"*) and in `validation_region`. When nothing fires, the
report says so, and that sentence is more useful than six bullets.

The anchor-column detector is the one that catches rogii directly: the profiler
already identifies `TVT_input` and records that carrying it forward scores 15.1
against the pipeline's 1380. **The signal exists; nothing reads it.**

## Exit criteria

1. A floor exists for every competition whose target has a defined shape, is
   computed under the same `ValidationPlan` as the model, and is recorded as a
   dataset reading — **no `hypothesis_id`, no execution id, no file under
   `ALLOWED_ROOTS`**.
2. The baseline plan gains a `COMPARE`, and `H-BASELINE` finishes a campaign with
   a status other than `proposed`.
3. A generic model that loses to the floor **fails the run** — the first place in
   this system where a run fails for being worse than a reference rather than for
   crashing.
4. With the gate closed, `available_tools` still returns at least one tool that
   can open it. The allowlist never empties.
5. `floor_undefined` never reads as `passed`, for any modality.
6. Every failure report cites an artifact for each cause it names.
7. Run against rogii, the gate reports `failed` and the report names the
   anchor-column cause.

## Traps

- **Ship observe-only first.** `_observe_delta` records why: *"calibrated against
  hand-written samples, and that is precisely how the two bugs got in. The first
  real campaign supplies a false-positive rate; blocking is a one-line change
  after that."* Same shape, same discipline — record verdicts against every
  existing workspace, then enforce.
- **Do not extend `ObservedOutcomes` with a third reading.** `_decide` is the
  single funnel for every verdict in the system. For the baseline plan, make the
  floor *the control*; everything downstream then works unchanged, and a metric
  mismatch is detected for free by machinery that already exists.
- **No env-var kill switch.** It gets set once during a frustrating afternoon and
  never unset, and nothing records that it happened. Use a durable
  per-competition waiver carrying a reason and who decided.
- **Do not ask codegen to emit the dummy baseline.** That puts the floor under
  the control of the thing the floor measures.
- **A gate demanding something unaffordable gets disabled.** The floor is cheap
  and universal, so it is the hard requirement. Baseline 1 is conditional, with
  its requirement level derived and recorded rather than assumed.
