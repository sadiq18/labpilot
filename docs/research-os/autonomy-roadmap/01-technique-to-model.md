# M7 — A technique must change the model

**Status:** designed · **Blocks:** everything · **Blocked by:** M10 (needs a
model that can write code)

**Design:** [design/01-technique-to-model.md](design/01-technique-to-model.md)

---

## Purpose

Hypotheses cannot currently be *tested*. `technique` is carried on the plan and
ignored by codegen, so every experiment reproduces the baseline. Observed on
rogii: twelve hypotheses, one reachable model state, **MSE 194.80 every time**.

Until this lands, the Conductor is a very sophisticated way of running the same
experiment repeatedly, and no other milestone can pay off.

## Goal

Two different hypotheses produce two different CV scores.

## Approach

**Recipe layer first, LLM codegen second.**

A technique becomes a declarative transform over the rendered template, not
free-form generated code:

```yaml
target_encoding:
  applies_to: [tabular_regression, tabular_classification]
  adds_features: [target_encoded]
  requires: [categorical_columns]
feature_interactions:
  applies_to: [tabular_*]
  adds_features: [pairwise_products]
model_swap_catboost:
  applies_to: [tabular_*]
  model: catboost
  params: {depth: 8}
ensemble_blend:
  applies_to: [tabular_*]
  combines: [previous_best, current]
```

Most useful techniques (`target_encoding`, `feature_interactions`, `SWA`,
`ensemble`, hyperparameter moves, model swaps) are expressible this way.
Reserve LLM codegen for what recipes cannot express.

**Why recipes before codegen:**

1. **Deterministic and testable.** A recipe either changed the feature set or it
   did not; assertable in a unit test. Generated code is neither.
2. **No model dependency.** Recipes work with a weak local model, so M7 is not
   hostage to M10 completing.
3. **Attributable failures.** When a recipe technique does not help, that is
   evidence about the *technique*. When generated code does not help, it is
   evidence about nothing (see Traps).

**Wiring:** `generate_plan(hypothesis_id=X)` must carry `technique` through to
the renderer, and the renderer must apply the matching recipe. The seam already
exists — `BaselineChoice` reaches `CodeRenderer.render()`, and hypothesis
metadata already flows into `code_engineering/capability.py` as `technique` /
`technique_stack` / `combo_techniques`. It is read into the prompt and then
dropped on the template path.

## Exit criteria

Not "recipes exist". Run a campaign and assert:

1. Three hypotheses with different techniques produce **three different**
   `cv_*` values in `metrics.json`.
2. Each experiment record names the technique that produced it.
3. A recipe that cannot apply to the problem type is rejected at plan time, not
   discovered at train time.

Criterion 1 is the whole milestone. It cannot be satisfied by accident.

## Traps

- **Do not start with LLM codegen.** On `qwen2.5-coder:14b` it produced no
  usable training code across an entire day. The emergency stub that replaced it
  wrote `cv_accuracy: 0.0` on a regression task and a submission with header
  `id,prediction` instead of `id,tvt` — while reporting success.
- **A weak model implementing a technique is worse than not running it.** The
  result is recorded as "technique X did not help": a false negative
  indistinguishable from a real one, which then poisons hypothesis generation
  permanently. This is why M10 routes codegen with *wait*, never *degrade*.
- **Do not let recipes bypass the validation plan.** Anything that adds features
  must respect `validation.exclude_features`, or a technique will quietly
  reintroduce the leakage columns the profiler worked out (`ANCC`, `ASTNU`,
  `TVT` …). A derived feature inherits its parents' availability.

## Related code

- `src/labpilot/research_engine/execution/baseline/selector.py` — `BaselineChoice`, `ValidationPlan`
- `src/labpilot/research_engine/execution/capabilities/code_engineering/capability.py` — technique already in context, dropped on the template path
- `.../code_engineering/offline_codegen/renderer.py` — where a recipe would apply
- `.../templates/tabular_regression_partitioned/` — the template to transform
