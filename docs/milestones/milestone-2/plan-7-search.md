# Plan 7 — Experiment Search

Back to [Milestone 2](README.md).

**Status:** Design. **Depends on:** Plan 1 (`Experiment`/`ExperimentGraph`). **Optionally
enriched by:** Plan 3 (`ExperimentComparison.verdict`/technique tags as filters). **Unlocks:**
nothing downstream — this is a leaf capability, useful standalone.

---

## Goal

Answer questions like the brief's Task 7 examples:

> Show every experiment where EMA was enabled and Macro F1 improved and training time < 4h.
>
> Find all failed attempts using Focal Loss.

without hand-grepping `runs/*/manifest.json`.

## Design

### 1. v1: composable flag filters, not a query language

Building a real expression parser (`--where "ema == true AND macro_f1_delta > 0"`) is real
engineering effort — a small grammar, a safe evaluator, error messages for malformed queries —
for a capability that a handful of composable flags can satisfy just as well for the brief's
own examples. v1 ships flags; a mini parser is called out as an explicit, optional v2 (§4).

```
research experiments search --competition <slug> \
    [--config <key>=<value> ...]        # e.g. --config model_params.ema=true
    [--recipe <name> ...]                # e.g. --recipe target_encoding
    [--metric-gt <key>:<value>]          # e.g. --metric-gt cv_macro_f1:0.80
    [--metric-lt <key>:<value>]
    [--metric-delta-gt <key>:<value>]    # requires comparison.json (Plan 3)
    [--metric-delta-lt <key>:<value>]
    [--runtime-max <duration>]           # e.g. 4h, 90m
    [--runtime-min <duration>]
    [--verdict <worth_keeping|not_worth_keeping|regression|inconclusive>]  # requires Plan 3
    [--status <completed|failed|...>]
    [--template <name>]
```

All provided flags combine with **AND** semantics (matches every brief example, which are all
conjunctions). Repeatable flags (`--config`, `--recipe`) combine as AND-of-equality-checks
across repetitions, not OR — e.g. two `--config` flags means both must hold.

### 2. Implementation

```python
@dataclass
class SearchFilters:
    config_equals: list[tuple[str, Any]]
    recipes: list[str]
    metric_gt: list[tuple[str, float]]
    metric_lt: list[tuple[str, float]]
    metric_delta_gt: list[tuple[str, float]]
    metric_delta_lt: list[tuple[str, float]]
    runtime_max_seconds: float | None
    runtime_min_seconds: float | None
    verdict: Verdict | None
    status: str | None
    template: str | None

def search(graph: ExperimentGraph, comparisons: dict[str, ExperimentComparison], filters: SearchFilters) -> list[Experiment]:
    return [
        exp for exp in graph.nodes.values()
        if _matches(exp, comparisons.get(exp.id), filters)
    ]
```

`_matches` is a straightforward chain of `if filter_set and not condition: return False` checks
— no clever query planning needed at "142 experiments" scale (a full linear scan of in-memory
`Experiment` objects, already loaded once per invocation, is milliseconds). `--config
model_params.ema=true` resolves dotted-path lookups against the merged config snapshot on
`Experiment` (same dotted-path convention as Plan 3's `ConfigChange.field`, for consistency).

`comparisons` is built by loading every `runs/<id>/comparison.json` that exists for runs in the
graph (Plan 3); filters relying on it (`--metric-delta-gt`, `--verdict`) simply exclude
experiments with no comparison on file (e.g. root runs) rather than erroring.

Duration parsing (`4h`, `90m`) reuses or mirrors whatever minimal duration-string parsing
exists elsewhere in the codebase; if none exists, a tiny local helper
(`_parse_duration("4h") -> 14400.0`) supporting `h`/`m`/`s` suffixes is enough — not worth a
dependency.

### 3. New/changed files

| File | Change |
|---|---|
| `src/labpilot/experiments/search.py` | new — `SearchFilters`, `search()`, duration parsing |
| `src/labpilot/cli/main.py` | + `experiments search`, flag parsing → `SearchFilters` |

### 4. v2 (explicit stretch, not required for this plan's acceptance)

A tiny `--where "<expr>"` boolean expression option, parsed with Python's `ast.parse(mode="eval")`
restricted to a safe subset (comparisons, `and`/`or`, dotted names resolved against the same
fields the flags expose, no calls/attribute-access beyond field lookup). This is real but
bounded effort (a single visitor over a restricted `ast.Expression` grammar) — flagged here so
a future PR can pick it up without re-deriving the design, but Plan 7's acceptance criteria
below don't depend on it.

## Non-goals

- No persistence of saved searches / named views in v1.
- No fuzzy/semantic search ("find experiments *like* this one") — exact predicate matching
  only. Semantic similarity is arguably better served by the `novelty` axis already being
  built in Plan 6, not duplicated here.
- No pagination — result sets at "hundreds of experiments" scale print fine in a single table;
  revisit if a competition's run count grows to a size where that's no longer true.

## Open questions

1. Should search be scoped to one competition only (matching every other new command in this
   milestone), or allow a global cross-competition search? → Scoped to one competition in v1,
   consistent with Plans 1/5/6; cross-competition search is a natural `--all-competitions`
   follow-up once there's a real use case for it.
2. Do `--metric-gt`/`--metric-lt` operate on the raw metric key from `metrics.json`
   (`cv_macro_f1`) or does the CLI need to know the competition's canonical metric key to
   default a bare `--metric-gt` without specifying which one? → Require an explicit `key:value`
   pair always; no implicit "the" metric, since a competition can have several logged (e.g.
   `cv_accuracy` and `cv_auc` both present).

## Acceptance criteria

- On a fixture graph with a mix of runs (some with `model_params.ema=true`, some without; some
  with a `comparison.json` showing a metric improvement, some without), the exact brief example
  — `--config model_params.ema=true --metric-delta-gt cv_macro_f1:0 --runtime-max 4h` — returns
  exactly the expected subset.
- `--recipe focal_loss --verdict regression` (the brief's second example, adapted to this
  codebase's recipe-naming convention) returns exactly the fixture's failed Focal Loss
  attempts.
- Providing no filters returns every experiment in the competition (a no-op filter, not an
  error or an empty result).
- An unparseable `--runtime-max` value produces a clear CLI error, not a silent no-match.
