# Plan 6 — Experiment Ranking

Back to [Milestone 2](README.md).

**Status:** Design. **Depends on:** Plan 1 (`ExperimentGraph`, for novelty/runtime history),
Plan 2 (`Hypothesis`, the candidates), Plan 5 (`KnowledgeBase`, for expected-gain estimate).
**Unlocks:** Plan 8.

---

## Goal

Given N proposed hypotheses, rank them:

```
Expected Gain | Implementation Cost | GPU Cost | Risk | Novelty
```

Task 8 in the brief. **No LLM planning here** — this is scoring/sorting a backlog that already
exists (from manual authoring via Plan 2, or from Plan 4's reflection-generated drafts), not
generating new ideas. This is explicitly a **recommendation engine, not a planner**: it ranks,
a human still runs `research improve --hypothesis <id>` to act on the top pick.

## Design

### 1. Candidates

The candidate pool is `HypothesisStore.list(status=PROPOSED)` for a competition (Plan 2) — no
separate "candidate" concept is introduced. A hypothesis is a candidate exactly when nobody has
started testing it yet.

### 2. Scoring dimensions

```python
class RankedCandidate(BaseModel):
    hypothesis: Hypothesis
    expected_gain: float          # estimated absolute delta on the competition's primary metric
    implementation_cost: float    # 0.0 (trivial) - 1.0 (major template work)
    gpu_cost_seconds: float       # estimated runtime, from historical comparables
    risk: float                   # 0.0 (safe) - 1.0 (risky) = 1 - hypothesis.confidence, adjusted
    novelty: float                # 0.0 (already tried this exact combo) - 1.0 (never tried)
    score: float                  # weighted combination, see below
```

**Expected gain**: match `hypothesis.tags` against `KnowledgeBase` entries (Plan 5) for the
competition's primary metric. If a matching `KnowledgeEntry` exists, `expected_gain =
entry.delta_estimate * entry.confidence`. If no match exists (a genuinely novel idea, no prior
evidence), fall back to a configurable prior (`experiments.ranking.default_expected_gain`,
e.g. `0.0` — unknown ideas don't get a free pass, they get ranked by novelty/risk instead).

**Implementation cost**: a small static heuristic keyed on `hypothesis.tags`, e.g. tags present
in a `cheap_tags` allowlist (`{"hyperparameter", "loss", "scheduler"}` — things `improvement/
tuner.py`/`recipes.py` already support) score low; anything else (implying new template work)
scores high. This is intentionally coarse and documented as a heuristic, not a real cost model
— it's the "implementation cost" axis from Task 8, and the brief itself doesn't ask for
anything more precise than a relative ranking signal.

**GPU cost**: mean `runtime_seconds` of past `Experiment`s in the same `ExperimentGraph` (Plan
1) whose `feature_recipes`/`model_params` overlap with tags similar to this hypothesis's tags;
fall back to the parent/root experiment's own runtime if there's no closer match. This reuses
data Plan 1 already computes — no new instrumentation needed.

**Risk**: `1 - hypothesis.confidence`, with a bonus reduction if `KnowledgeBase` already has a
high-confidence *positive* entry for a matching tag (i.e. "we already know this kind of change
tends to work" lowers perceived risk beyond the hypothesis author's own stated confidence).

**Novelty**: `1.0` minus the highest tag-set Jaccard similarity between this hypothesis and any
`model_params`/`feature_recipes` combination already present in an `Experiment` in the graph.
An identical combination already tried scores novelty `0.0`.

### 3. Combining into a score

```python
score = (
    weights.expected_gain * normalize(expected_gain)
    - weights.implementation_cost * implementation_cost
    - weights.gpu_cost * normalize(gpu_cost_seconds)
    - weights.risk * risk
    + weights.novelty * novelty
)
```

Weights are named, documented config, not hidden constants:
`configs/default.yaml: experiments.ranking.weights.{expected_gain,implementation_cost,gpu_cost,risk,novelty}`,
with sane defaults (e.g. `expected_gain=2.0, implementation_cost=0.5, gpu_cost=0.5, risk=1.0,
novelty=0.5` — expected gain dominates, matching "we mostly want the thing most likely to help,
adjusted for cost/risk"). `normalize()` is a simple min-max normalization across the current
candidate pool so weights aren't sensitive to a metric's raw scale.

### 4. New/changed files

| File | Change |
|---|---|
| `src/labpilot/experiments/models.py` | + `RankedCandidate` |
| `src/labpilot/experiments/ranking.py` | new — `rank_candidates(competition, runs_dir, knowledge_dir) -> list[RankedCandidate]` |
| `src/labpilot/cli/main.py` | + `experiments rank` |
| `configs/default.yaml` | + `experiments.ranking.{weights,default_expected_gain,cheap_tags}` |

### 5. CLI

```
research experiments rank --competition <slug> [--top 5]
```

Output table: hypothesis id, one-line prediction, expected gain, cost, GPU cost estimate,
risk, novelty, final score — sorted descending by score. Top row is the "Recommended Next"
line used verbatim in Plan 8's dashboard.

## Non-goals

- **No automatic execution of the top-ranked candidate.** Ranking stops at "here's the
  recommendation and why" — the brief's own Task 7 for the (Milestone-3) planner says "user
  still approves," and this plan treats that as true even earlier, for ranking alone.
- **No learned/trained scoring model.** Weighted heuristic sum only, matching Task 8's
  explicit "No LLM planning yet. Just ranking." There's no ML-on-ML here.
- **No candidate *generation*.** Ranking only ever operates on hypotheses that already exist
  (manually authored via Plan 2, or reflection-drafted via Plan 4). If the backlog is empty,
  `rank` returns an empty list with a clear message, it does not invent ideas.

## Open questions

1. Should `implementation_cost` eventually be informed by which templates actually support a
   tag (introspecting `improvement/recipes.py`'s registered recipes), rather than a static
   allowlist? → Worth doing once the allowlist proves too coarse in practice; not required for
   v1's acceptance criteria.
2. What happens when two hypotheses have overlapping/contradictory tags (e.g. two different
   ways to address the same observation)? → No dedup/merge logic in v1; both are ranked
   independently, a human picks one.

## Acceptance criteria

- Given a fixture `KnowledgeBase` with a high-confidence positive entry for `"loss"`-tagged
  techniques and two candidate hypotheses (one tagged `"loss"`, one tagged with an unmatched
  novel tag), `rank_candidates()` scores the known-good one higher when confidence/risk
  dominate, and demonstrably factors in `expected_gain`, `risk`, and `novelty` independently
  (each can be toggled in the fixture to change the ranking, proving they're not dead weights).
- `research experiments rank --competition <slug>` with zero proposed hypotheses prints a
  clear "no candidates" message, not an error.
- Changing `experiments.ranking.weights` in `configs/default.yaml` changes the resulting order
  on a fixture with at least two candidates whose relative score depends on that weight.
