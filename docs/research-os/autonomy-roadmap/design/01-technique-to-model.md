# Design — M7: Technique → Model

**Plan:** [../01-technique-to-model.md](../01-technique-to-model.md) ·
**Status:** design · **Owner:** unassigned

---

## 1. Background

The Research OS proposes hypotheses, plans against them, trains models, and
reflects on results. A hypothesis carries a `technique` — the thing under test
(`target_encoding`, `feature_interactions`, `SWA`, a model swap).

Validation against `rogii-wellbore-geology-prediction` showed the loop turning
correctly end to end — twelve hypotheses, distinct plans (`P-002→H-010`,
`P-003→H-013`), real training on 773 partitions — while **every experiment
scored MSE 194.80, identically**.

The technique reaches the LLM codegen path in full
(`capability.py:214-220` → `code_engineer_user.j2:7-9`). It reaches the
deterministic template path **not at all**. When the LLM produces no files
(routine on a local 14B model), the run renders the same default LightGBM
regardless of hypothesis.

## 2. Problem statement

> A hypothesis cannot be tested, because nothing translates its technique into a
> change in the trained model.

Three concrete breaks, verified:

| Break | Evidence |
|---|---|
| Fallback drops technique | `_render_template_fallback(self, root, choice)` — `capability.py:380`. No `context`, no `plan_meta`. |
| Renderer never receives recipes | `CodeRenderer(config.training).render(template, choice, root)` — `capability.py:402`. All four keyword args omitted. |
| Partitioned template has no hooks | `tabular_regression_partitioned/train.py.j2` — zero `{% if %}` blocks. The template rogii actually uses. |

Consequence beyond the wasted cycle: reflection records "technique X did not
help" for a run in which technique X was never applied. That is a **false
negative written into durable research memory**, and it is indistinguishable
from a real one.

### What already exists (and changes the design)

The roadmap plan proposed *inventing* a recipe layer. Exploration found one,
half-wired:

- `CodeRenderer.render(..., feature_recipes: list[str] | None = None, ...)` —
  `renderer.py:24-34`. **No production caller supplies it.**
- Template gates: `{% if "log_numeric" in feature_recipes %}` and
  `{% if "target_encoding" in feature_recipes %}` — `tabular_regression/train.py.j2:51,61,145,160`
  and the classification twin.
- A canonical mined vocabulary producing exactly these labels —
  `intelligence/feature_recipes.py:149-175`: `target_encoding`,
  `one_hot_encoding`, `frequency_encoding`, `polynomial_features`,
  `feature_interactions`, `rolling_features`, `lag_features`,
  `aggregation_features`, `log1p_transform`, `binning`, `tfidf`.
- The historical supplier, now orphaned: `TrainingOverrides`
  (`legacy_run_overrides.py:43-47`) is a **field-for-field match** with the
  renderer's four kwargs. This was wired once and became disconnected.

**Two `feature_recipes` namespaces exist and never meet**: structured
`FeatureRecipe` objects mined from papers/repos/forums feed hypothesis
generation; the renderer's `list[str]` is matched against two hardcoded
literals. The bridge between them is most of this milestone.

## 3. Goal

Two different hypotheses produce two different models, and the difference is
attributable to the technique.

## 4. Requirements

### Functional

| # | Requirement |
|---|---|
| F1 | A hypothesis's `technique` reaches the rendered training code on **both** the LLM and template paths |
| F2 | A technique resolves to a declarative spec: feature recipes, model family, model params |
| F3 | A technique that cannot apply to the problem type / data is rejected **at plan time**, not discovered at train time |
| F4 | A technique that resolves to *no change* fails loudly rather than silently rendering the baseline |
| F5 | The applied technique is recorded on the experiment record and in `baseline_choice.json` |
| F6 | Technique stacks (`technique_stack`, `combo_techniques`) compose without conflicting |
| F7 | Recipes respect `validation.exclude_features` — a derived feature inherits its parents' availability |

### Non-functional

| # | Requirement |
|---|---|
| N1 | **No LLM dependency.** Recipes are deterministic, so M7 is not blocked on M10/M14 |
| N2 | Adding a technique is a registry entry + a template gate — no changes to Conductor, planner or capability |
| N3 | Rendering stays deterministic: same (choice, technique) → byte-identical `train.py` |
| N4 | Unknown techniques degrade to an explicit, recorded rejection — never a silent baseline |
| N5 | Backward compatible: a plan with no technique renders exactly what it renders today |

## 5. Scope

### In scope

- `TechniqueSpec` + registry, with problem-type and data applicability
- Wiring `technique` from `plan.metadata` through the fallback into the renderer
- Feature-recipe gates in `tabular_regression_partitioned` (currently zero)
- Model-family / model-param techniques (e.g. a CatBoost swap, depth changes)
- Provenance: applied technique on evidence metadata + `baseline_choice.json`
- Contract test: different technique → different artifact digest

### Out of scope

- Free-form LLM code generation quality → [M10](../04-llm-tiering.md), [M14](../09-llm-required.md)
- Reflection generating hypotheses from results → [M8](../02-objective-loop.md)
- Score-driven technique *selection* → [M13](../08-policy-reasoning.md)
- Bridging mined `FeatureRecipe.transform` (executable code from papers) — this
  design bridges the **name**, not the body. Executing mined transforms is a
  separate, riskier milestone
- Image/text/deep templates — tabular first, since that is where the vocabulary
  and the validation case are

## 6. High-level design

```
Hypothesis.technique ──┐
technique_stack        ├──► plan.metadata          (exists: planner.py:91-93)
combo_techniques   ────┘          │
                                  ▼
                      ┌───────────────────────┐
                      │  TechniqueResolver    │  NEW
                      │  name → TechniqueSpec │
                      │  + applicability gate │
                      └───────────┬───────────┘
                                  │ TechniqueSpec
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
    LLM codegen path                         Template path
    (spec in prompt — richer                 render(choice, *, feature_recipes,
     than today's bare name)                  model_params, model_family)
              │                                       │
              └───────────────┬───────────────────────┘
                              ▼
                     rendered train.py
                              │
                              ▼
              provenance: applied_technique on
              evidence metadata + baseline_choice.json
```

The insertion point is `CodeEngineeringCapability._write()`, which already holds
both `plan_meta` (`capability.py:188`) and `choice` (`:173`). Nothing upstream
changes.

## 7. Components and responsibility boundaries

| Component | Owns | Does **not** own |
|---|---|---|
| `TechniqueSpec` (new, `execution/technique/models.py`) | Declarative description of what a technique changes | How to apply it; whether it is a good idea |
| `TechniqueRegistry` (new, `execution/technique/registry.py`) | Name → spec lookup; applicability predicate | Choosing which technique to run |
| `TechniqueResolver` (new, `execution/technique/resolver.py`) | plan.metadata + choice + profile → resolved spec or rejection | Rendering; persistence |
| `CodeRenderer` (`offline_codegen/renderer.py`) | Turning a spec into Jinja context | Deciding the spec |
| Templates (`templates/*/train.py.j2`) | Executing a recipe correctly | Knowing what a technique means |
| `CodeEngineeringCapability` | Orchestrating resolve → render → apply; recording provenance | Technique semantics |
| `BaselineSelector` | Template + validation plan from **data** | Anything technique-related — deliberately unchanged |

**The boundary that matters:** the registry knows *what a technique changes*; the
template knows *how to execute a recipe*; the capability knows *nothing about
either* beyond passing one to the other. Adding `feature_interactions` touches a
registry entry and a template gate — nothing else.

## 8. Design choices

### 8.1 Extend the existing `feature_recipes` mechanism, not a new one

**Chosen.** The kwarg, the template gates, the mined vocabulary and the
legacy supplier all already exist and agree on the same label set.

*Rejected:* a new recipe YAML (what the roadmap originally proposed). It would
create a third `feature_recipes` namespace alongside the two that already fail
to meet.

### 8.2 Declarative specs before LLM codegen

**Chosen.** A recipe either changed the feature set or it did not — assertable in
a unit test, and independent of model quality.

*Rejected:* relying on LLM codegen. On `qwen2.5-coder:14b` it produced no usable
training code across an entire day; the emergency stub that replaced it wrote
`cv_accuracy: 0.0` on a regression task. A weak model implementing a technique
badly is **worse than not running it**, because the failure is recorded as
evidence against the technique.

### 8.3 Reject at plan time, not train time

**Chosen.** Applicability is checked when the technique is resolved, so an
inapplicable technique never consumes a training run.

Reuses the concept already shipped in
`intelligence/hypothesis/candidates.py::filter_incompatible_techniques`
(cross-modality rejection). That filter stops `vit` being *proposed* for tabular;
this one stops an unsupported-but-plausible technique being *executed*.

### 8.4 A no-op resolution is a failure

**Chosen.** If a technique resolves to an empty spec, the capability raises
rather than rendering the baseline.

This is the direct lesson of the bug: the system's disease is *success reported
without checking the effect*. Silently rendering the baseline for an unknown
technique is exactly how MSE 194.80 repeated twelve times while looking healthy.

*Rejected:* warn-and-continue. It reproduces the failure mode with better logging.

### 8.5 Resolve in the capability, not on `BaselineChoice`

**Chosen.** `BaselineSelector.select(competition, profile)` derives from **data
only** and never sees a plan — a clean property worth keeping. The capability
already holds both halves.

The resolved spec is *stamped onto* `baseline_choice.json` for provenance
without the selector gaining a plan dependency.

### 8.6 Bridge the recipe *name*, not the mined `transform` body

**Chosen.** Mined `FeatureRecipe` objects carry a `transform` string from papers
and repos. Executing that is arbitrary-code-from-the-internet with a
correctness problem on top.

Bridging names gives most of the value (the vocabulary already matches) at none
of that risk. Executing mined transforms is a later, separately-justified
milestone.

## 9. Low-level design

### 9.1 `TechniqueSpec`

```python
class TechniqueSpec(BaseModel):
    name: str                                  # canonical, matches mined vocabulary
    feature_recipes: list[str] = []            # -> renderer kwarg -> template gates
    model_family: str | None = None            # lightgbm | catboost | ...
    model_params: dict[str, Any] = {}          # merged over DEFAULT_TABULAR_MODEL_PARAMS
    applies_to: list[str] = []                 # problem types; empty = any
    requires: list[str] = []                   # categorical_columns | datetime_column | partitioned
    description: str = ""

    def is_noop(self) -> bool:
        return not (self.feature_recipes or self.model_family or self.model_params)
```

### 9.2 Registry — initial entries

Chosen to cover the mined vocabulary that the templates can plausibly execute:

| Technique | feature_recipes | model | requires |
|---|---|---|---|
| `target_encoding` | `["target_encoding"]` | — | categorical_columns |
| `log1p_transform` | `["log_numeric"]` | — | numeric_columns |
| `feature_interactions` | `["feature_interactions"]` | — | ≥2 numeric |
| `polynomial_features` | `["polynomial_features"]` | — | ≥2 numeric |
| `lag_features` | `["lag_features"]` | — | partitioned |
| `rolling_features` | `["rolling_features"]` | — | partitioned |
| `aggregation_features` | `["aggregation_features"]` | — | partitioned |
| `catboost` | — | catboost | — |
| `deeper_trees` | — | — (`max_depth`, `num_leaves`) | — |
| `more_estimators` | — | — (`n_estimators`) | — |

The first two already have template gates. The rest need gates added.

### 9.3 Resolution

```python
def resolve_technique(plan_meta, choice, profile) -> TechniqueResolution:
    """Returns applied spec, or a rejection with a reason. Never silently empty."""
```

- Reads `technique`, then `technique_stack`, then `combo_techniques` — the same
  precedence `capability.py:214-220` already uses for the LLM prompt.
- Composes a stack by union of `feature_recipes` and last-wins on `model_params`;
  a conflicting `model_family` is a rejection, not a silent pick.
- Applicability: `choice.problem_type in spec.applies_to` (or empty), and every
  `requires` satisfied by the `DatasetProfile`.
- **Leakage guard (F7):** any recipe whose inputs intersect
  `choice.validation.exclude_features` is rejected. A derived feature inherits
  its parents' availability — this is how a technique would otherwise quietly
  reintroduce `TVT`/`ANCC` on rogii.

### 9.4 Wiring — the three verified breaks

```python
# capability.py:380 — signature gains the plan metadata it already has at :188
def _render_template_fallback(self, root, choice, plan_meta, profile) -> CodeProposal | None:

# capability.py:402 — stop discarding the kwargs the renderer already accepts
CodeRenderer(config.training).render(
    template, choice, root,
    feature_recipes=spec.feature_recipes,
    model_params=spec.model_params,
)
```

`renderer.py` gains `model_family` in the Jinja context (`:47-75`); the four
existing kwargs need no signature change.

### 9.5 Template gates

`tabular_regression_partitioned/train.py.j2` gains gates in
`_add_partition_features` — the natural seam, since it already builds per-entity
features and is where `lag`/`rolling`/`aggregation` belong. Recipes must be
applied **after** the `EXCLUDE_FEATURES` filter, mirroring the existing
delta-feature guard that already skips excluded columns.

### 9.6 Provenance (F5)

- `baseline_choice.json` gains `applied_technique: TechniqueResolution`
- Evidence metadata (`capability.py:309-320`) gains `technique` and
  `technique_origin` (`llm` | `registry` | `none`)
- The experiment record carries the technique, so reflection can attribute a
  result to it

## 10. Observability

| Signal | Where | Answers |
|---|---|---|
| `technique resolved: <name> → recipes=[...] model=<family>` | progress + log | Did the technique reach the renderer? |
| `technique rejected: <name> — <reason>` | progress + `record_suggestion` | Why did nothing change? |
| `applied_technique` block | `baseline_choice.json` | What ran, after the fact |
| `technique` on evidence metadata | experiment record | Attribution during reflection |
| `train.py` digest | already captured (`capability.py:265`) | Did the artifact actually differ? |

The digest is the load-bearing one. It is the check that would have caught this
bug class on day one, and it is already computed — nothing reads it across runs.

Rejections route through `record_suggestion`, the existing mechanism for "the
system naming a capability it lacks", so an unsupported technique becomes a
roadmap signal rather than a silent skip.

## 11. Production readiness

**Correctness gates**

- Contract test (generalises to [M15](../10-capability-audit.md)): two techniques
  → two different `train.py` digests
- Every registry entry has a test asserting its recipe changes the generated
  feature set
- Leakage test: a recipe deriving from an excluded column is rejected
- Regression: a plan with **no** technique renders byte-identically to today (N5)

**Failure modes**

| Mode | Handling |
|---|---|
| Unknown technique | Reject, record suggestion, fail the step (F4) |
| Inapplicable to problem type | Reject at resolve time (F3) |
| Recipe raises at train time | Existing `ExperimentProducedNoMetricsError` catches the no-metrics case |
| Conflicting stack | Reject rather than silently pick |

**Rollout.** Ship the registry + wiring behind the existing behaviour first: with
no technique on the plan, nothing changes (N5). The first real signal is a rogii
campaign where `P-00N` and `P-00M` produce different scores — which is also
[M8](../02-objective-loop.md)'s prerequisite, since `metric_history` only becomes
meaningful once scores can differ.

**Explicit non-goal.** This does not make the system competitive on rogii. The
remaining gap there is geosteering domain modelling (gamma-ray/typewell log
correlation), which no registry entry infers. M7 makes hypotheses *testable*;
whether the proposed hypotheses are *good* is [M8](../02-objective-loop.md) and
[M14](../09-llm-required.md).
