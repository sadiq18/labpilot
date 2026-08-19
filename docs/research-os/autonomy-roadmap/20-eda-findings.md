# M25 — A finding is a statistic, not a plot

**Status:** not started · **Blocked by:** [M23](18-baseline-correctness.md) (the floor is what makes a correlation meaningful)

---

## Purpose

`docs/research-os/backlog/future-specialists.md` deferred a Data Understanding
specialist and wrote down what would promote it:

> *"~30%+ of campaign time on EDA, or repeated leakage/distribution misses"*

Both conditions are now met, from one workspace. rogii spent six experiments on
feature variations that lost to a constant, and the distribution miss is
`TVT_input` — a column that equals the target wherever present, which the
pipeline used for KMeans clusters and a kriging feature and never anchored to.

The ordering matters and the evidence supports it rather than merely illustrating
it. rogii's two highest-value findings — `TVT_input` is the target's known
prefix, and carry-forward scores RMSE 15.1 — came from the **profiler**
([M22](17-dataset-understanding.md)) and the **floor**
([M23](18-baseline-correctness.md)), not from EDA. EDA that runs before those two
produces confident analysis of the wrong problem.

After them, its value is real. Measured on 60 wells, none of these is computed
anywhere in the system today:

| finding | value | why it changes the next experiment |
|---|---|---|
| residual after carry-forward | **std 7.55** | *this* is the modelling target, not 1380; 2.236 is a 3.4× reduction from it |
| `Z` and `MD` against that residual | **\|r\| = 0.453** each | real exploitable signal, unused |
| `GR` missingness **on scored rows** | **9.7%–51.2%**, a 5× spread across wells | decides whether any GR feature is usable, per well |

The last one is the shape this milestone exists to produce: not a plot, a fact
that changes what to try next. A single overall missingness figure hides it
entirely.

## The design point that matters more than the statistic list

**Correlate against the residual from the floor, not against the raw target.**

On rogii, `Z` against raw `TVT` is near-trivially correlated — both are depths,
and the number tells you nothing. `Z` against the carry-forward residual is
0.453, and that is a feature.

This is why M25 genuinely *depends* on M23 rather than merely following it. Without
a floor there is no residual, and every correlation is computed against a quantity
the trivial predictor already explains.

## What is there today

Essentially nothing. The complete inventory of numeric computation over a dataset
in `src/`:

- `profile_columns` — dtype, `null_count`, `null_pct`, `unique_count`, `is_numeric`
- `_numeric_stats` — `{min, max, mean, std}`. **That is the entire distribution
  summary.** No median, no quantiles, no skew.
- `enrich_column_stats` — `avg_length`, the only non-numeric statistic anywhere

A search for `.corr(`, `corrwith`, `mutual_info`, `feature_importances_`,
`permutation_importance`, `.skew(`, `kurtosis`, `.quantile(`, `value_counts`,
`ks_2samp`, `wasserstein`, `psi` returns **zero real hits**. Every match on
"drift" is prose about *code* drifting apart. There is no train/test comparison of
any kind.

`DatasetAnalyzer` derives exactly one thing beyond a field-for-field copy of
`profile.json`: `null_heavy` at ≥20%. It never opens a CSV. But its contract is
already precisely what this milestone needs:

> *"Never calls an LLM and never touches the network (§2.4 Hard No: statistics
> and distributions are deterministic)."*

M25 gives it something to report.

## Deterministic statistics, LLM interpretation

The split is not novel here — `ExperimentReviewerAgent` already ships it:

> *"The comparator (CV/LB deltas) stays deterministic; this agent only interprets
> those numbers."*

M25 is that pattern applied to dataset statistics. `CompetitionAnalyzer` is the
precedent that an analyzer *may* call an LLM, so the layering is established.

Every statistic is emitted as an `EdaFinding` with a **stable id**. The LLM is
given the findings and **not the data**, and every hypothesis it proposes must
cite a finding id. **A hypothesis citing a finding that does not exist is
discarded** — [M22](17-dataset-understanding.md)'s data-veto test applied to
interpretation, and the thing that stops an "EDA agent" becoming a fluent guess
generator.

## The finding → hypothesis path already exists and is open

This is what makes the milestone small, and it means **no new evidence type is
needed**.

`intelligence/hypothesis/ledger.py:93-118` scans **all** artifacts with *no type
filter* and harvests three fields:

```python
for art in store.list_artifacts():
    for tech in art.techniques:                     → TECHNIQUE / STACKED candidates
    for recipe in art.metadata["feature_recipes"]:  → same, category=feature_engineering
    for claim in art.claims:                        → UNUSED_CLAIM candidates
```

`DatasetAnalyzer` already emits a `DATASET` artifact on every analyze run — and
**populates none of those three fields**. The pipe from "a dataset fact" to "a
ranked hypothesis candidate" is built, connected, and carrying nothing. The job
is to put something in it, not to lay new pipe.

Two models exist for the payload and are **declared and never used** — zero
producers, zero consumers, anywhere in `src/` or `tests/`:

- `ResearchFinding{source, finding, applicability}`
- `ResearchArtifactType.NOTE`, commented *"manual / imported note"*

Use them rather than inventing a third shape.

**One filter genuinely needs opening**, and it is a decision rather than an
oversight to route around: `intelligence/retrieval/fetchers.py:172-180` admits
only `PAPER`, `EXPERIMENT` and `REPOSITORY` into the retrieval bundle. So a
`DATASET`/`NOTE` artifact reaches the *ledger* but never the *context* the
hypothesis drafter reads. Ledger-only is enough to generate candidates; retrieval
is what lets the drafter explain them.

## Exit criteria

1. Running twice on the same bytes produces **identical** findings — statistics
   are reproducible or they are not evidence.
2. Every finding carries the statistic, the threshold and the columns that
   produced it. A finding without its number is a plot.
3. Every EDA-derived hypothesis cites a finding id, and one citing a finding that
   does not exist is rejected.
4. Correlations are computed against the floor's residual, and a fixture where
   raw-target correlation would mislead (rogii's `Z`) demonstrates the difference.
5. **EDA-derived hypotheses win more often than the existing generators'.** This
   is the criterion that cannot be satisfied by shipping structure — it needs a
   campaign, which is the same debt M8 and M11 still carry.
6. Missingness is reported on the scored rows separately, not only overall.

## Traps

- **Do not produce plots.** Nothing in the loop can read an image, and a plot
  nobody reads is the "structure without function" failure this roadmap opens
  with.
- **Do not let an analyzer call another analyzer** — `analyzers/base.py:32-34`
  forbids it. The statistics pass is inside `DatasetAnalyzer` or a sibling
  reading the same profile, never a chain.
- **Reuse M23's leakage detector.** A second implementation of "feature correlates
  ~1.0 with the target" will drift from the first.
- **`scipy` and `sklearn` appear nowhere in `src/` today** — they exist only inside
  generated `train.py`, and only nine files in `src/` import pandas or numpy at
  all. Using sklearn here is fine but should be a deliberate dependency decision.
  scipy is currently available only transitively via sklearn.
- **Do not compute everything.** Every statistic must have a named consumer — a
  hypothesis shape it can motivate. A number nobody acts on is the same waste as
  a plot, with better formatting.
