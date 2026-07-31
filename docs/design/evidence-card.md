# Evidence Card — atomic learning unit

Canonical design for causal learning after each hypothesis execution.
See also [research-graph.md](research-graph.md).

Operator docs: [SOP.md](../SOP.md), [CLI.md](../CLI.md). Architecture:
[ARCHITECTURE.md](../ARCHITECTURE.md).

---

## Verdict

Every hypothesis execution produces one **Evidence Card** — not merely a higher
score. The card is the system of record for learning: expected vs observed,
technique attribution, claim/belief updates, accept/reject, and reusable domains.

Success means: **did these techniques cause an improvement vs the parent
control?** — not “score went up in isolation.”

---

## Feedback loop

```mermaid
flowchart TD
  artifacts[Artifacts techniques] --> hyps[Hypotheses]
  hyps --> run[Plan run treatment E]
  run --> metrics[Rich metrics]
  metrics --> compare[COMPARE vs control E]
  compare --> card[Evidence Card]
  card --> graph[Research Graph edges]
  card --> beliefs[Beliefs confidence]
  card --> claims[Claims confidence]
  graph --> query[Graph query for planner]
  beliefs --> artifacts
  query --> hyps
```

---

## Canonical path (every edge carries evidence)

```text
Technique  --supports-->  Claim
    --used_in-->  Hypothesis
    --executed_as-->  Experiment/Execution
    --produced-->  Evidence Card (+0.007 CV, …)
    --updates-->  Belief
```

Example:

```text
Mixup
  │ supports
  ▼
"Regularization improves generalization"
  │ used_in
  ▼
H-014
  │ executed_as
  ▼
E-042
  │ produced
  ▼
EV-xxx  (observed.cv_gain=+0.007)
```

---

## Schema

Persisted at `knowledge/<slug>/research/evidence/EV-xxx.json`, linked from
`execution_outcome.json` and shown on `research hypothesize show`.

| Field | Meaning |
|-------|---------|
| `hypothesis_id` | Treatment hypothesis |
| `control_experiment` | Parent / champion execution id |
| `treatment_experiment` | Current execution id |
| `expected.cv_gain` | From hyp `expected_impact` |
| `expected.runtime` | Optional fractional runtime prior |
| `observed.cv_gain` | Primary CV Δ vs control |
| `observed.lb_gain` | Public LB Δ vs prior (patched on `research submit`) |
| `observed.runtime` | Fractional train-time change |
| `observed.stability` | `improved` \| `similar` \| `worse` \| `unknown` (from `cv_std`) |
| `technique_attribution` | Signed credit of `cv_gain` across techniques (priors) |
| `claim_updates[]` | `{claim, evidence, confidence_delta}` |
| `decision` | `accepted` \| `rejected` \| `inconclusive` |
| `reusable_for` | Modality / domain tags for later shared knowledge |
| `impact_error` | `observed.cv_gain - expected.cv_gain` |

### Decision rules (deterministic)

- **accepted** — clear positive CV (maximize) or non-negative LB without overfit; not ruined by severe stability regression.
- **rejected** — clear negative CV/LB vs control.
- **inconclusive** — missing control, within noise epsilon, or overfit (local up / LB down).

Hypothesis status maps: accepted→`confirmed`, rejected→`rejected`, else `inconclusive`.

---

## Default control

Parent hypothesis execution metrics (plan `parent_execution_id` /
`parent_metrics`). Else best confirmed champion. Never silent “vs P-001 with
null delta.”

Default cost: **one** treatment experiment. Full ablation is sparse (material
gains / low-confidence techniques / large expected–observed miss).

---

## Metrics the card needs

Train pipelines should emit when measured (omit/null if not):

`cv_*`, `cv_folds`, `cv_fold_scores`, `cv_mean`, `cv_std`, `train_time_s`,
`inference_time_s`, `peak_memory_mb`.

---

## Submit patch

`research submit` sets `observed.lb_gain`, recomputes `decision`, refreshes
Research Graph edges, and **steps** belief confidence — it must not overwrite
belief confidence with absolute constants.

---

## Implementation pointers

- Models/store: `research_engine/evidence/`
- COMPARE: `evidence/compare_service.py` + EvaluationCapability
- Graph write/query: `intelligence/graph/`
- Apply beliefs/hyp: `evidence/apply.py`
