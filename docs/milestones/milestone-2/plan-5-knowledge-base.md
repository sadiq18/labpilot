# Plan 5 — Knowledge Base

Back to [Milestone 2](README.md).

**Status:** Design. **Depends on:** Plan 3 (`ExperimentComparison`, required signal), Plan 4
(`StructuredReflection`, optional enrichment). **Unlocks:** Plan 6, Plan 8.

---

## Goal

Every experiment should contribute knowledge, not just a score. Instead of:

```
Experiment 61
Score 0.842
```

Store:

```
Knowledge: SpecAugment
  Effect:      improves
  Metric:      Macro F1 (rare classes)
  Confidence:  0.91
```

Task 5 in the brief. This is the accumulation layer — the thing that turns "142 runs" into
"we know SpecAugment helps" without a human reading 142 `reflection.md` files.

## Design

### 1. Storage

`knowledge/<competition-slug>/knowledge_base.json` — a single file per competition, containing
a list of `KnowledgeEntry`. One file (not one-per-entry like hypotheses in Plan 2) because
entries are frequently *updated in place* (a technique's confidence/estimate shifts every time
a new experiment tags it) rather than created once and rarely touched — a single file with
atomic rewrite-on-update is simpler than N small files with the same churn.

### 2. Data model

```python
class KnowledgeEffect(StrEnum):
    IMPROVES = "improves"
    HURTS = "hurts"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

class KnowledgeEntry(BaseModel):
    technique: str                 # e.g. "SpecAugment", "EMA", "Heavy CutMix" — free-form, matched by normalized string
    metric_key: str                # which metric this observation is about, e.g. "cv_macro_f1"
    effect: KnowledgeEffect
    delta_estimate: float          # rolling average of observed deltas, signed per metric direction
    confidence: float              # see formula below
    sample_size: int                # number of experiments contributing to this entry
    evidence_run_ids: list[str]
    updated_at: datetime
```

### 3. Update trigger — deterministic, driven by the comparator

`KnowledgeBase.update_from_comparison(comparison: ExperimentComparison, technique_tags: list[str])`
is called right after `comparison.json` is written (Plan 3's new step in `Pipeline.improve()`).
`technique_tags` come from `comparison.changes` — each `ConfigChange` already carries a
human-readable `label` (`"+ Mixup"`, `"+ EMA"`); the technique name is that label with the
leading `+`/`-` stripped and normalized (lowercased, whitespace-collapsed) for matching against
existing entries.

For each technique tag touched by this comparison:

```python
def _update_entry(existing: KnowledgeEntry | None, delta: float, run_id: str) -> KnowledgeEntry:
    n = (existing.sample_size if existing else 0) + 1
    prior_avg = existing.delta_estimate if existing else 0.0
    new_avg = prior_avg + (delta - prior_avg) / n          # incremental mean, no need to store history
    consistent = existing is None or (prior_avg >= 0) == (delta >= 0)
    confidence = min(0.95, 0.5 + 0.1 * n) if consistent else max(0.3, (existing.confidence if existing else 0.5) - 0.15)
    effect = KnowledgeEffect.IMPROVES if new_avg > epsilon else KnowledgeEffect.HURTS if new_avg < -epsilon else KnowledgeEffect.NEUTRAL
    ...
```

This is deliberately a simple, explainable heuristic (incremental mean + a consistency-based
confidence bump/penalty), not a statistical model — matching Task 5's "no LLM required" framing
for the base signal. The exact constants (`0.1` per sample, cap `0.95`, penalty `0.15`) are
named constants in `knowledge.py`, not magic numbers buried in logic, and are revisitable
without a design change.

### 4. Optional LLM enrichment (from Plan 4)

If a `StructuredReflection.new_hypotheses` or the reflection's `tags`-equivalent free text
mentions a technique not already covered by the comparator's `ConfigChange` labels (e.g. a
qualitative observation like "overfitting on rare classes" that isn't a config field), Plan 5
*may* also let `KnowledgeBase.update_from_reflection(structured_reflection)` add a
`KnowledgeEntry` with `effect=UNKNOWN`, `confidence` capped low (e.g. ≤ 0.4) unless corroborated
by a comparator-derived entry for the same technique. This keeps the base signal
deterministic-only (works with Plan 3 alone) while allowing Plan 4 to add value once it exists.
This is explicitly optional and can ship in a later PR without changing Plan 5's core contract.

### 5. New/changed files

| File | Change |
|---|---|
| `src/labpilot/experiments/models.py` | + `KnowledgeEffect`, `KnowledgeEntry` |
| `src/labpilot/experiments/knowledge.py` | new — `KnowledgeBase` (load/save, `update_from_comparison`, `update_from_reflection`, `list_entries`, `top_discoveries`, `known_failures`) |
| `src/labpilot/orchestrator/pipeline.py` | after writing `comparison.json`, call `KnowledgeBase.update_from_comparison(...)` |
| `src/labpilot/cli/main.py` | + `experiments knowledge list` |

### 6. CLI

```
research experiments knowledge list --competition <slug> [--technique specaugment] [--effect improves]
```

`KnowledgeBase.top_discoveries(n=5)` → entries with `effect == IMPROVES` sorted by
`delta_estimate * confidence` descending. `KnowledgeBase.known_failures(n=5)` → entries with
`effect == HURTS` sorted by the same score ascending. Both feed directly into Plan 8's
dashboard.

## Non-goals

- **No cross-competition knowledge transfer.** "SpecAugment helps on BirdCLEF" does not
  automatically inform a different competition's knowledge base in v1 — each competition's
  `knowledge_base.json` is independent. A shared/global knowledge base is a plausible future
  extension, not required here.
- **No technique taxonomy/ontology.** Matching is by normalized string equality on the
  `ConfigChange` label. If two experiments spell a technique differently (`"specaugment"` vs
  `"SpecAugment (time+freq)"`), they won't merge automatically in v1 — acceptable given the
  categorization dict in Plan 3 already controls the exact labels emitted.
- No manual `knowledge add` CLI command in v1 — entries are always derived from comparisons
  (and optionally reflections), never hand-authored, to keep the base's confidence numbers
  meaningfully tied to evidence.

## Open questions

1. Should `KnowledgeEntry` be keyed on `(technique, metric_key)` so the same technique can have
   different, independently-tracked effects on different metrics (e.g. helps `cv_macro_f1` but
   hurts `cv_accuracy`)? → Yes — this is important enough to bake into the model now rather
   than retrofit later; `KnowledgeBase` internally indexes by that composite key.
2. Do we need file locking for concurrent writes (two runs finishing at once updating the same
   `knowledge_base.json`)? → Not for v1 — LabPilot runs are sequential CLI invocations today,
   no concurrent pipeline execution exists yet. Note this assumption explicitly so it's
   revisited if/when remote/parallel execution (deferred P2 execution work) lands.

## Acceptance criteria

- Two sequential fixture comparisons that both tag `"EMA"` with a consistent positive delta
  produce one `KnowledgeEntry` for `("ema", "<metric>")` with `sample_size == 2`, `effect ==
  IMPROVES`, and increased confidence vs. after just the first comparison.
- A comparison tagging a technique with a delta of the opposite sign from its existing entry
  reduces that entry's confidence (per the penalty branch) rather than raising it.
- `research experiments knowledge list --effect hurts` returns only entries with
  `effect == HURTS`, matching the brief's "Known Failures" concept.
- `KnowledgeBase.top_discoveries(3)` and `.known_failures(3)` on a fixture base with mixed
  entries return correctly ordered, correctly filtered lists.
