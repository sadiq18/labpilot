# Design — M23: a floor to measure against

**Plan:** [../18-baseline-correctness.md](../18-baseline-correctness.md) ·
**Status:** design · **Blocked by:** ~~[M22](17-dataset-understanding.md)~~ **cleared** ·
**Consumes:** `feature_columns`, `train_test_relationship`, `MetricRef` ·
**Feeds:** [M25](20-eda-findings.md)

---

## 1. Background

M22 made the description of a dataset trustworthy: which column is the label,
which are usable features, how the scored units relate to the training ones,
what the metric is — each with the evidence behind it and a question where there
is none. What nothing in the system does is compute **what the dumbest possible
answer scores**, so there is no yardstick against which a number means anything.

## 1.1 The pipeline, and where the pieces already are

This milestone is the last two stages of one line, not a feature on its own:

```
Dataset Understanding → Task Understanding → Objective Resolver → ObjectiveSpec
        → Validation Strategy → Baseline 0 → Baseline 1 → Compare
```

and everything after it hangs off the same spine — `ObjectiveSpec → Research
Goal → Hypothesis → Experiment → Evaluation`. The metric registry is **one
knowledge source inside the Objective Resolver**, not the architecture: that is
already how `objective.py` is written, where the registry is level 4 of six
ranked sources.

Read on `main` at `e6712c6`, the stages exist and the *pipeline* does not:

| Stage | Exists as | State |
|---|---|---|
| Dataset understanding | M22 `DatasetSchema` | ✅ with evidence per field |
| Task understanding | `target_type` + `target_distribution` (step 1) | ✅ measured over the resolved target; the keyword matcher is now the fallback |
| Objective Resolver | `resolve_objective` (#145) | ⚠️ exists, six ranked sources, probe, contradictions — called **once, from `cli/conduct.py:299`** |
| `ObjectiveSpec` | `objective.json` (step 0) | ✅ persisted beside `profile.json`, resolved from the schema, reused while its inputs hold |
| Validation Strategy | `ValidationPlan` in `baseline_choice.json` | ⚠️ derived from the *profile*, not from the objective |
| Baseline 0 / 1 / compare | — | ❌ this milestone |

The defect is structural, not conceptual: each stage re-derives from
`profile.json` instead of reading the stage before it, so the resolver's
contradiction detection and its `unresolved` list — which already name *what to
ask* — reach a console line and nothing else. **`selector.py` and the
code-engineering capability contain no reference to an objective at all.**

Step 0 closes the first half of that: `objective_stage.py` reads the schema,
resolves once, and writes `objective.json` from `prepare_workspace` — so every
entry point has an objective, not just `research conduct`. The stored artifact
carries the *inputs* it was resolved from rather than a fingerprint of them, so
staleness is a comparison and a re-resolution can say which input moved. What
remains is the second half: making the stages after it **read** that file, which
is step 2.

## 1.2 The gate, as nine ordered checks

| # | Check | Where it comes from |
|---|---|---|
| 1 | Understand dataset | M22 ✅ |
| 2 | Identify target | M22 ✅ — with a question when it cannot |
| 3 | Identify train/test | M22 ✅ `train_test_relationship` |
| 4 | Identify ID | M22 ✅ — and rogii's is the open question |
| 5 | Infer feature types | M22 ✅ `feature_columns` + typed exclusions |
| 6 | Identify evaluation metric | `resolve_objective` — exists, needs wiring |
| 7 | Build trivial baseline | **this milestone** |
| 8 | Build ML baseline | **this milestone** |
| 9 | Compare against trivial | **this milestone** |

Checks 1–5 do not need building. They need *reading* — which is the difference
M22 made, since each now carries a confidence and can refuse to answer.

---

## 2. Problem

Measured on rogii, 2026-08-13, on a split that mimics the real test — first
27.3% of each well known, predict the tail, 120 wells, 576k rows:

| predictor | RMSE |
|---|---|
| carry forward the last known `TVT` — **one line** | **15.10** |
| predict the global mean | 648.6 |
| linear extrapolation from the prefix | 1,199.6 |
| **the pipeline, after six experiments** | **1,380.4** |

Two× worse than a constant, 91× worse than one line of code — and the campaign
reported `1789 → 1409 → 1380` as progress for fourteen steps. Without a floor,
*improving* and *below chance* are the same reading.

### What it cost, which is not the runs

- **19 child hypotheses** (H-109 … H-127) extending a result that loses to a
  constant. They are the pool the next campaign selects from.
- **Eight techniques driven to confidence 0.0** — `vit`, `rolling_features`,
  `warmup`, `SWA`, `add`, `dataset`, `tabular_regression_partitioned` at 0.196.
  The system concluded they do not work here; it cannot know that, because
  everything in that regime scores ~1400 regardless. Beliefs persist across
  campaigns.

> A wasted run costs a run. A false belief costs every campaign after it.

### The mechanism, read on `main` at `e6712c6`

| Site | Fact |
|---|---|
| `conductor/loop.py:928` | `_baseline_plan_exists` returns true for *any* plan whose `metadata["plan_kind"] == "baseline"`. Plan existence. Not that it ran, not that it scored |
| `conductor/actions.py:206` | That boolean is the flip: once a baseline plan is compiled, a `baseline=True` request is rewritten to plan against a hypothesis. The campaign starts minting improvement hypotheses the moment `P-001` compiles |
| `H-BASELINE` | Has been `proposed` since it was created: the baseline plan is the only template with no `COMPARE` task, so no evidence card is ever built for it. Three sites read its status and all three *exclude* it |
| its own reflection | *"The model failed to learn from the training data, possibly due to an issue with the template configuration."* — the correct diagnosis, parked in an `inconclusive` hypothesis where it gates nothing |
| `profile.anchor_column` | Names `TVT_input`, records that carrying it forward is the baseline to beat, and **nothing reads it**. The signal exists |
| `accessor/profiler/tabular.py` | `target_type` and `target_distribution` are still absent. M22 deferred them here: they are measurements over a resolved target, and this is their first consumer |

---

## 3. Requirements

### Functional

1. Compute a **floor** — the best trivial predictor — for every dataset whose
   target has a defined shape, under the **same `ValidationPlan`** the model is
   told to use.
2. Record it as a **dataset reading**: no `hypothesis_id`, no execution id, no
   file under `ALLOWED_ROOTS`. It describes the data, not a run.
3. Report a verdict with **nine distinct states**, each with a distinct operator
   action, and never collapse them into a boolean.
4. A model that loses to the floor **fails the run**, and the failure report
   names only causes it can cite an artifact for.
5. `floor_undefined` blocks rather than passing, for every modality.
6. The gate is **observe-only** until a real campaign supplies a false-positive
   rate; enforcement is a config flip, not a rewrite.
7. Nothing on this path calls an LLM.

### Non-functional

| Property | Target |
|---|---|
| Cost | ≤ 1 pass over the target column per fold; no model fitting |
| Determinism | Same bytes + same `ValidationPlan` → same floor, bit for bit |
| Metric implementations | **1** — `execution/metrics.py:compute_metric`, never a second |
| Staleness | A re-derived profile or a changed answer invalidates the reading |
| Gate churn | `available_tools` never empties: with the gate closed, at least one tool that can open it remains |

---

## 4. Goals & success metrics

The plan's seven exit criteria, plus one this design adds:

| # | Criterion | Measured by |
|---|---|---|
| 1 | A floor exists per defined-shape target, under the model's own `ValidationPlan`, recorded as a dataset reading | No `hypothesis_id`/execution id/`ALLOWED_ROOTS` file in the artifact |
| 2 | The baseline plan gains a `COMPARE`; `H-BASELINE` finishes with a status other than `proposed` | A campaign transcript |
| 3 | A model that loses to the floor fails the run | The first failure in this system for being *worse* rather than for crashing |
| 4 | With the gate closed, `available_tools` still returns a tool that can open it | The allowlist never empties |
| 5 | `floor_undefined` never reads as `passed`, for any modality | Parametrized over every `target_type` |
| 6 | Every failure report cites an artifact per cause | No cause without a citation |
| 7 | On rogii the gate reports `failed` and names the anchor-column cause | Sandbox copy of the real workspace |
| **8** | **An uncertain schema is not a failed baseline** | A dataset with an open M22 question reports `blocked_uncertain`, never `failed` |

Goal 8 is new because M22 shipped after this plan was written: the schema can
now say *"I do not know which column is the label"*, and a floor computed
against a guessed target is worse than no floor. `blocked_uncertain` was in the
plan's nine states with no defined trigger; this is it.

## 4.1 Acceptance: a reliable experiment foundation

The goals above are per-mechanism. The bar for the *milestone* is a hit rate
across unseen competitions — given one it has never seen, the system produces a
correct schema, validation strategy, dummy baseline and strong generic baseline:

| Stage | Target |
|---|---|
| Dataset understanding | > 95% |
| Correct target | > 95% |
| Correct train/test | > 95% |
| Correct metric | > 95% |
| Dummy baseline | **100%** |
| Generic baseline | consistently beats dummy |

**That is [M24](19-competition-benchmark.md)**, and it is the reason M24 cannot
wait until this milestone is finished. Its plan lists itself as blocked by M22
and M23 — *"there must be something to score"* — which is true of the last two
rows and false of the first four: M22 shipped, and a corpus of 10–20 captured
competitions could score its four stages today. Held to the end, this milestone's
own hit rate is an assertion until the moment it ships.

So the corpus lands beside the work rather than after it: **step 1 of this
rollout is the first that can be scored**, and every step after it is measured
rather than argued. `tests/integration/` has held only stale `.pyc` since
`109745c`, which is the room it goes in.

**Nothing after this milestone starts until those numbers hold.** Hypothesis,
experiment, reflection, research memory and autonomous research all consume a
schema, a split and a reference — and each of them writes beliefs that outlive
the run that made them. rogii is the measurement of what that costs: 19 child
hypotheses and eight techniques at 0.0 confidence, every one of them derived
from a pipeline 91× worse than a single line of code.

---

## 5. Scope

**In:** `target_type` and `target_distribution` on the schema; the floor and its
strategies; the reading and its fingerprint; the nine-state verdict; the failure
report and its detectors; the `COMPARE` on the baseline plan; observe-only
enforcement with a durable waiver.

**Out, and why:**

| Excluded | Why |
|---|---|
| Floors for detection, segmentation, keypoints, generation, RL | Their honest floors — predict-empty, random policy, the provided sample agent — all require running something in the competition's own harness. That is an environment runner, which does not exist. `floor_undefined`, which blocks |
| ~~Baseline 1~~ | **Now in scope** — see §7.7. The gate's output is the *comparison*, and a floor with nothing to compare against is half a gate |
| Asking codegen to emit the dummy baseline | It puts the floor under the control of the thing the floor measures |
| An env-var kill switch | It gets set once during a frustrating afternoon and never unset, and nothing records that it happened |
| A third reading on `ObservedOutcomes` | §7.5 — the floor becomes the *control* instead, and every downstream verdict works unchanged |

---

## 6. Architecture

```
profile.json  ── target_type, target_distribution, feature_columns ─┐
baseline_choice.json ── ValidationPlan, metric ─────────────────────┤
                                                                    ▼
                                         floor.py  (no model, no LLM, no codegen)
                                                    │  best trivial predictor per fold
                                                    ▼
                                        baseline_floor.json   ← the dataset reading
                                                    │
   metrics.json (the model's cv_<metric>) ──────────┤
                                                    ▼
                                          gate.py  → one of nine states
                                                    │
                              ┌─────────────────────┴──────────────────┐
                              ▼                                        ▼
                    report.py (detectors, each citing an artifact)   submit / submit_learn
```

### Components

| Component | Responsibility |
|---|---|
| `accessor/profiler/` | `target_type`, `target_distribution` — measurements over the resolved target |
| `execution/baseline/floor.py` | Strategies, per-fold fitting, scoring through `compute_metric` |
| `execution/baseline/gate.py` | The nine states, the fingerprint, the waiver |
| `execution/baseline/report.py` | Detectors that read artifacts and cite them |
| `conductor/policy.py` | `submit`/`submit_learn` preconditions; `_baseline_plan_exists` retired |

---

## 7. Implementation

### 7.1 The floor

```python
class FloorReading(BaseModel):
    """What the dumbest defensible answer scores. A fact about the dataset."""

    metric_name: str
    strategies: dict[str, float]      # every strategy tried, and its score
    best_strategy: str
    score: float
    validation: ValidationPlan        # the one the model is told to use
    fingerprint: str
    computed_at: str
```

- **Folds come from the same `ValidationPlan`**, read from `baseline_choice.json`
  so the two cannot drift. A floor on a different split is not a floor.
- **Fitted per fold, on the train side only.** Fitting on the whole target is
  the leakage version and looks unbeatable on skewed data.
- **Every strategy is recorded; the floor is the best of them.** One that picked
  the worse constant is a gate too easy to pass.
- **Scored by `compute_metric`**, written under `cv_<metric_name>` — which is
  what makes the comparison free.

The optimal constant follows the metric's `target_kind`, which `MetricRef`
already carries from M22 step 6:

| metric | floor |
|---|---|
| mse, rmse | mean |
| mae | **median** |
| rmsle | `expm1(mean(log1p(y)))` |
| accuracy, f1 | majority class |
| logloss | the **prior vector**, not the argmax |
| auc | **exactly 0.5, asserted analytically** rather than computed |

And one that is not a constant at all: where `profile.anchor_column` exists,
**carry it forward**. That is rogii's 15.1 against the pipeline's 1380, and the
profiler has named it since 2026-08-13 with nothing reading it.

> **The floor is determined by the shape of the prediction target, not the
> modality of the input.** An image competition's label is still a class column;
> its floor is the class prior, identical to tabular. Modality decides only
> whether Baseline 1 is affordable.

### 7.2 `target_type` and `target_distribution`

M22 deferred these here. Both are **measurements** over the resolved target —
cardinality, dtype, class balance, skew, zero-inflation — so they carry no
confidence of their own and inherit the target's. `TargetType` gains its
detector at last: `binary`, `multiclass`, `multilabel`, `continuous`, `count`,
`ordinal`, and `none` for a target that is not a column.

### 7.3 The nine states

`unknown` · `floor_missing` · `floor_undefined` · `blocked_uncertain` ·
`awaiting_ml` · `stale` · `failed` · `passed` · `waived`

Each has a distinct operator action, which is the entire reason for nine rather
than a boolean — M20's finding is that collapsing states is how eight gates
reported `pass` on things that could not run.

- **`blocked_uncertain`** is goal 8: `pending_schema_questions` is non-empty, so
  the target may be wrong and a floor computed against it means nothing. The
  operator answers; they do not debug a baseline.
- **`stale`** matters as much as `failed`. The fingerprint covers validation
  scheme, target, metric, `profile.schema_version` **and the M22 answers
  fingerprint** — an operator answering "the label is `Depth`" invalidates every
  reading that described `Zone_Depth`.
- The verdict is **derived on read** from the reading and the model's score, not
  stored. A stored verdict is derived state that outlives its cause, which is
  AGENTS.md rule 2 and the mistake `apply_card_to_beliefs` cost this repo.

**The gate deliberately does not read `H-BASELINE.status`** — five layers of
derivation, one of which raises, so a bookkeeping fault would read as "baseline
not passed". It reads the floor reading, written by one writer. H-BASELINE
finally getting a status is a valuable *consequence*, cross-checked in tests,
never a dependency.

### 7.4 What the gate gates

**Only `submit` and `submit_learn`.** Untouched: `analyze_competition`,
`generate_plan`, `run_plan`, `run_experiment`, `implement`, `reflect`,
`query_memory` — those are how a campaign *builds* the baseline. Gating
`run_plan` would be a bug: it is how the gate gets opened, and it is what keeps
goal 4 true.

Hypothesis minting is not a tool, so `available_tools` cannot reach it. It is
blocked at `_hypothesize_run` and at `persist_recommendations` — the only
durable write, which also covers the CLI path.

`_baseline_plan_exists` is retired. Its one consumer (`actions.py:206`) asks
"has the baseline been done?" and gets "was a plan object compiled?"; it now
gets the gate's verdict, so a campaign stops flipping to research mode on the
strength of a compiled file.

### 7.5 The floor is the control

The trap the plan records, and the reason this stays small: **do not extend
`ObservedOutcomes` with a third reading.** `_decide` is the single funnel for
every verdict in the system. For the baseline plan, the floor becomes
`parent_cv` — the control — and everything downstream works unchanged. A metric
mismatch is then detected for free by machinery that already exists.

That also delivers goal 2: the baseline plan gains a `COMPARE` whose control is
the floor rather than a missing parent, so an evidence card is built and
`H-BASELINE` finishes with a real status.

### 7.6 The report names only what fired

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

Every detector reads an artifact; **no LLM on this path**. A list that prints
identically on every failure is a list nobody reads — this repository has paid
for that twice, in `check_confinement` and in `validation_region`. When nothing
fires, the report says so, and that sentence is more useful than six bullets.

### 7.7 Baseline 1 — the strong generic model

The floor says a number is not worse than nothing. **Baseline 1 says whether the
pipeline is worth having at all**, and the gate's output is the comparison
between them:

```
Dummy baseline (mean)      RMSE  1.42
Generic ML     (lightgbm)  RMSE  0.91
Improvement                35.9%   ✓
```

- **LightGBM**, which is already a dependency (`pyproject.toml:21`) — no new
  install, no GPU, no tuning. CatBoost is a later option behind the same
  interface, not a second code path.
- **Minimal preprocessing**, and the same `feature_columns` M22 resolved: the
  point is a competent default, not a good model. Anything clever here makes the
  reference move.
- Fitted on the **same `ValidationPlan`** as the floor and as the pipeline. Three
  numbers on three splits compare nothing.
- **Written by us, not by codegen** — same rule as the floor, same reason.

**Affordability is derived, not assumed.** The plan's trap is real: a gate
demanding something unaffordable gets disabled. So the requirement level is
computed and recorded — a tabular dataset under a size threshold *must* have
Baseline 1, and where it is genuinely unaffordable (an image or environment
dataset, or one that exceeds the budget) the state is `awaiting_ml`, which is
one of the nine and is not `passed`. The floor remains the hard requirement
because it is cheap and universal; Baseline 1 is a hard requirement wherever it
can run, which on tabular Kaggle is everywhere.

### 7.8 What "do not proceed to research" means

Enforcement is at hypothesis generation, not at submission. A campaign whose
pipeline loses to a constant may still run plans, implement, and reflect — those
are how the gate gets opened — but it may not **mint hypotheses**, because that
is where a false belief enters the store and outlives the run:

- `_hypothesize_run` and `persist_recommendations`, which is the only durable
  write and covers the CLI path;
- `generate_plan` against a *proposed hypothesis* stays open, since the baseline
  plan itself is one.

That is a stricter rule than gating `submit`, and the right one: rogii's cost
was 19 child hypotheses and eight techniques driven to 0.0 confidence, all
written down while the pipeline was 91× worse than one line of code. None of it
was a submission.

---

## 8. Design choices & tradeoffs

| Choice | Rejected | Chosen | Tradeoff |
|---|---|---|---|
| Who computes the floor | Codegen emits it | The system computes it in Python we own | The floor cannot be gamed by the thing it measures; one more capability step |
| Where it lives | An experiment with an execution id | A **dataset reading** beside `profile.json` | It describes data, not a run, so it survives every experiment; a new artifact kind |
| Verdict storage | Store the nine-state verdict | Derive it from the reading on read | Cannot outlive its cause; recomputed per read (a dict lookup and a comparison) |
| Enforcement | Block from day one | **Observe-only**, then a config flip | A real campaign supplies the false-positive rate before anything is refused — `_observe_delta`'s discipline, and how its two bugs were found |
| Escape hatch | `LABPILOT_SKIP_BASELINE=1` | A durable per-competition waiver with a reason and a decider | An env var gets set once and never unset, and nothing records that it happened |
| Undefined floors | Pass them through | `floor_undefined`, which **blocks** | An honest refusal for detection/RL rather than a gate that quietly always passes |
| Schema uncertainty | Compute a floor anyway | `blocked_uncertain` | A floor against a guessed target is a confident wrong number — the failure M22 exists to prevent |

---

## 9. Observability

| Surface | Shows |
|---|---|
| `research baseline show` | Every strategy tried and its score, the winner, the model's number, the verdict and what to do about it |
| `baseline_floor.json` | The reading itself — one writer, and the fingerprint that says what it describes |
| Campaign transcript | `stop:baseline_failed` distinct from `stop:failing`; the report inline |
| `H-BASELINE` | A status other than `proposed`, for the first time |

Alert-worthy: a floor that beats the model by any margin; `floor_undefined` on a
competition the operator believes is tabular; a waiver older than the profile it
waives.

---

## 10. Testing

| # | Check | Proves |
|---|---|---|
| 1 | Floor strategies against hand-computed values per metric, including logloss's prior vector and AUC's analytic 0.5 | The constants are the optimal ones, not plausible ones |
| 2 | The floor is computed on the `ValidationPlan` from `baseline_choice.json`, and a different plan yields a different floor | Goal 1 — a floor on another split is not a floor |
| 3 | Fitting is per fold on the train side: a fixture where whole-target fitting scores better exposes the leakage version | The version that looks unbeatable is refused |
| 4 | rogii's shape: the anchor-carry-forward floor beats the recorded pipeline score, gate reports `failed`, report names the anchor cause and cites `profile.anchor_column` | Goals 3, 6, 7 |
| 5 | Every `target_type` maps to a state; the undefined ones to `floor_undefined`, never `passed` | Goal 5, parametrized |
| 6 | With the gate closed, `available_tools` is non-empty and contains `run_plan` | Goal 4 |
| 7 | A workspace with an open schema question reports `blocked_uncertain`, not `failed` | Goal 8 |
| 8 | Observe-only: with enforcement off, a failing verdict is recorded and no tool is withheld | The rollout's first stage does what it claims |

Real-data validation is a **sandbox copy**, never the live workspace
(AGENTS.md rule 1).

---

## 11. Rollout

| Step | Content | Ships when |
|---|---|---|
| 0 ✅ | **`ObjectiveSpec` becomes a stage**: resolved from the schema rather than loose CLI args, persisted as `objective.json` beside `profile.json` | The resolver's contradictions and `unresolved` list reach something other than a console line |
| 1 ✅ | **Task understanding**: `target_type`, `target_distribution` on the schema, feeding the resolver's `task` instead of metadata keywords | M22's deferred measurements land |
| 2 | **Validation Strategy reads the objective**, not the profile alone | One spine, no re-derivation |
| 3 | `floor.py`: strategies, per-fold fitting, `compute_metric`; `baseline_floor.json` | Checks 1–3 |
| 4 | `baseline_one.py`: LightGBM, minimal preprocessing, same plan; the comparison | The gate's own output exists |
| 5 | `gate.py`: nine states, fingerprint, waiver — **observe-only** | Checks 5–8; nothing is refused yet |
| 6 | The report and its detectors | Checks 4, 6 |
| 7 | `COMPARE` on the baseline plan; floor as control through `_decide` | Goal 2 |
| 8 | Enforcement: no hypothesis minting until the gate passes; `_baseline_plan_exists` retired | Goal 3 |

**Migration.** `baseline_floor.json` is new; its absence reads as
`floor_missing`, which is a state rather than an error. Existing workspaces
acquire one on their next baseline run.

**Rollback.** Steps 0–7 change no campaign behaviour: the verdict is recorded
and reported, nothing is withheld. Step 8 is the only one that can stop a run,
and it is a config flip with a durable waiver.

**On enforcing sooner.** The instruction is *no hypothesis generation until the
baseline passes*, and step 8 is exactly that. It is last because of what the
plan's own trap records about `_observe_delta`: it was *"calibrated against
hand-written samples, and that is precisely how the two bugs got in"*. One
campaign's worth of recorded verdicts is what turns "the gate is right" from an
argument into a false-positive rate — and a gate that wrongly refuses to mint
hypotheses stops the system dead, which is a worse failure than the one it
prevents. Steps 0–7 are not a slower route to enforcement; they are what makes
the flip in step 8 a one-line change instead of a rewrite.

**What could go wrong.** A floor that is wrong in the model's favour hides a
real failure — which is why every strategy is recorded, not just the winner. A
false `failed` stops a good campaign — which is why enforcement waits for a
measured false-positive rate rather than an argument.
