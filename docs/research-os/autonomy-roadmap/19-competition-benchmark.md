# M24 — Understanding is measured, not asserted

**Status:** in progress — capture, expander, scorer and tier 1 shipped
2026-08-20 with two real fixtures · **Blocked by:** ~~[M22](17-dataset-understanding.md)~~
**cleared**; [M23](18-baseline-correctness.md) blocks only the *baseline*
criteria, so the four schema stages are scoreable now — and are, at
`research bench score`

---

## Purpose

This roadmap's standing complaint about itself:

> **What is now unmeasured is the loop itself.** M8 and M11 both ship
> implementations whose exit criteria can only be met by a campaign log, and no
> campaign has been run since the template pack was retired.

M22 and M23 would repeat that exactly. Both make claims — *"the schema is right
>95% of the time"*, *"a model that loses to a constant is caught"* — that a unit
test on a hand-written fixture cannot support. A guard calibrated only on samples
someone wrote is how the last two rounds of bugs got in.

So this milestone is the instrument. It answers one question, per competition,
reproducibly: **given a competition it has not seen, does the system understand
the dataset?**

Target rates: dataset understanding, correct target, correct train/test and
correct metric each **>95%**; dummy baseline **100%**; generic baseline
consistently beats dummy.

## What exists to build on, and what does not

`tests/integration/` contains **only stale `__pycache__`** — 17 `.pyc` files
naming a suite deleted in `109745c` (2026-07-27). `git ls-files tests/integration`
returns nothing. There is no multi-competition harness and no scorecard anywhere.

Existing fixtures are synthetic: `titanic_data_dir` is 12 hand-typed rows, not
the real file. `tests/fixtures/capstone/analyze.golden.json` is a 47KB
**fabricated** report for `birdclef-2026`.

The pattern to mirror is `tests/fixtures/real_failures/` — four real artifacts
captured from live rogii runs, each dated and sourced in a `MANIFEST.md`, with a
self-integrity test. That test exists because a 79-byte paraphrase once sat in the
corpus claiming to be 624 bytes.

## Capture: headers-only by default

Header-set arithmetic alone buys target, id, submission columns, train-only
columns and file roles — five criteria and three of the six success measures, at
zero licence risk and ~400 bytes per competition. Rows buy only four more things:
dtype, cardinality, anchor equality, suffix contiguity.

So rows are a **per-fixture justification**, not a policy. That also lets
sponsored datasets ship as `headers_only` with `redistribution: forbidden` and
still score most criteria.

Modes: `verbatim` · **`headers_only`** · `head:<n>` · `stride:<k>` ·
`synthesized` (tiny media only, to prove a *reader* works — never a *dataset
fact*). Every file records `source_sha256`, `source_bytes`, `source_rows`,
`fixture_rows`, so a fixture can be re-derived from a re-download and proven
identical.

### `stride:` exists because of a measurement

rogii has three test wells, and the scored suffix starts at rows **1442 / 1545 /
2083**. `_detect_suffix_scoring` requires `scored.max() == n_rows - 1` and
`len(scored) == n_rows - scored.min()` — absolute row indices. `_is_known_prefix_of`
requires `known.argmin() == known.sum()`, also absolute.

`head:50` destroys both invariants. A verbatim prefix needs ~290KB for test alone.

Keeping every 25th row and renumbering the submission ids preserves
contiguous-prefix, contiguous-suffix, equality-where-known, per-kind column sets
and dtypes — changing only counts and numeric statistics, which are declared
unverifiable anyway. Such fixtures are marked `provenance: derived`, never
verbatim.

## What truncation cannot validate — declared, not hidden

Each fixture carries an `unverifiable_from_fixture:` block with a reason per
entry, and the scorer **refuses to score** those rather than scoring a truncation
artifact:

`row_count` (independently broken — see below) · cardinality · `null_pct` ·
distributions · partition counts · media ratios · baseline quality.

`row_count` cannot be a criterion in any tier until the profiler is fixed:
`playground-series-s6e7/profile.json` records `row_count: 100000` with
`row_count_estimated: false` for a 690,088-row file, because `max_rows_sample`
caps at 100,000. That is a confident lie, not a truncation artifact.

## Media fixtures — where "headers + rows" stops working

For tabular, the facts live in the bytes of the header, so truncation is
**lossless**. For media the facts live in directory structure, **file counts and
ratios**, per-file properties, and the manifest↔media join.

Shrinking a media tree destroys the ratios — and ratios are exactly what
`_detect_image` decides on. A fixture with 4 images and 3 CSVs classifies as
*tabular* while the real dataset with 100k images classifies as *image*. **A
proportionally shrunk media fixture does not fail to validate modality detection;
it validates the wrong answer.**

Three parts instead:

1. **`listing.tsv`** — a *complete* recorded listing of the real dataset, one row
   per file with probed media properties. ~1.5MB of text, ~300KB gzipped,
   diffable, and it carries counts and ratios exactly.
2. **`data/`** — a micro-tree with every CSV under tabular capture rules, plus
   3–5 **byte-verbatim** real media files (the smallest in the real set).
3. **A test-time expander** that materializes the listing as zero-byte
   placeholders, then overlays the real bytes.

The expander is the key move: `_detect_image` counts by extension and **never
opens a file**; `_role_of` reads directory parts; `_match_filename_column` opens
only the CSV. All of that is satisfied by empty files at the right paths — 30,000
`touch()` calls run in under a second. Real bytes are supplied exactly where
decoding happens.

**Honest caveat:** a probed property recorded in a listing is an assertion by the
capture tool, not a fact the test re-derives. That is why tier 3 re-probes the
real dataset and fails on drift. Without that job, `listing.tsv` is a paraphrase.

## The corpus

Chosen for the failure modes that actually matter, not for coverage of a taxonomy.

**Tabular** — `titanic` (submission is `gender_submission.csv`; detection must not
be name-locked) · `house-prices` (RMSLE/minimize; a 663-unique target must not
read as classification) · `spaceship-titanic` (bool target, which `is_numeric`
excludes) · `playground-series-s6e7` (**the metric-synonym fixture** — ships as a
known-failure on day one) · **`rogii`** (the highest-value fixture: partitioned,
two kinds, per-kind withholding, suffix scoring, anchor column — the one the
system got wrong) · `store-sales` (temporal, multi-table) · `home-credit`
(7-table relational) · `santander` (200 anonymous columns) · `m5` (**metric maps
to nothing** — expected `unknown`, proving the system says so instead of
defaulting to maximize).

**Text** — `nlp-getting-started` · `feedback-prize` (six target columns; today
`tabular.py:302` *raises* on any submission that is not `[id, target]`).

**Image** — `dogs-vs-cats` (no `train.csv` at all; the label is in the filename) ·
`aerial-cactus` (the manifest-join path) · `biohub` (zarr, node/edge submission,
**no target column**).

**Audio / mixed** — `birdclef` (`train_audio/<species>/*.ogg` + metadata +
per-species-wide submission: audio detection, mixed modality, wide multi-label —
and it replaces the *fabricated* birdclef golden fixture with the real thing) ·
`rainforest-connection`, so "audio" is not proven by one example.

**Boundary** — `connectx` (empty data directory, kernel-only submission; scored
**only** on whether the system refuses) · a parquet-only competition (proves
format coverage is a known boundary rather than a crash).

## Scoring

Fifteen criteria, each `pass | fail | not_applicable | unverifiable |
known_failure`.

**Partial credit is per-criterion, never fractional.** "Target right, id wrong" is
two criteria with two answers. The criterion table *is* the partial credit, and it
is legible; a fractional per-fixture score just invites tuning the denominator.
"Dataset understanding >95%" is the strict aggregate — a fixture counts as
understood only if **every** applicable criterion passes.

The harness **runs the shipped path**, never a reimplementation: the real
`CompetitionParser`, the real `prepare_workspace` capability (its
tabular→inventory fallback *is* part of the behaviour under test), the real
`BaselineSelector`. Observed values are read from the **serialized artifacts** —
a fact that is correct in memory and lost on serialization is not correct.

Two criteria the original brief did not have:

- **correct modality**, including `mixed` — the case today's binary detector
  cannot express;
- **correct abstention** — a fixture whose expected outcome is *"should have
  asked"* is as valuable as one with a known answer. A `must_ask` fixture is
  scored on that alone and is `not_applicable` on target/metric/validation.
  Otherwise adding a genuinely hard competition tanks the accuracy number, which
  creates pressure to guess — the exact failure the criterion exists to prevent.

## Three tiers

- **Tier 1 — hermetic, every PR.** All schema/metric/validation/modality scoring,
  plus dummy-baseline **validity**: it runs, emits a submission with exactly the
  sample's columns and row count, no NaN, only labels seen in training. That is
  the honest reading of *"dummy 100%"*. Under 60 seconds.
- **Tier 2 — full data, nightly.** Row counts, cardinality, distributions, real
  media probing, undecimated rogii, and **generic-beats-dummy**, defined as
  strictly better in the metric's declared direction *by more than the
  fold-to-fold std*. Not "better by any epsilon" — that is noise.
- **Tier 3 — the agreement check.** ✅ Score both the truncated fixture and the
  full dataset, and assert they agree on every tier-1 criterion.
  `tests/integration/test_corpus_agrees_with_reality.py`, marked `slow` and
  skipped loudly with every path it looked in. All five fixtures agree today,
  over 4–5 claimed criteria each, rogii's 1,546 tables included.

**Tier 3 is the single most important test here.** It is what licenses a hermetic
corpus to stand in for real data. If the two disagree on a criterion, that
fixture's capture mode is wrong and the criterion moves to `unverifiable`. Tier 1
is the gate, tier 2 is the truth, tier 3 keeps tier 1 honest about tier 2.

Tier 1 explicitly does **not** claim generic-beats-dummy: on 50 rows LightGBM
routinely loses to the mean, and asserting it there is asserting noise.

## One vocabulary for metrics

Three overlapping metric vocabularies exist and none is authoritative:
`CANONICAL_METRIC_KEYS`, `SUPPORTED_METRICS_BY_PROBLEM_TYPE`, and
`_primary_cv_keyed`'s hand-written priority list — which privileges
`cv_balanced_accuracy` and `cv_roc_auc`, neither of them canonical anywhere.

Matching becomes **exact against a declared alias set, never substring**.
Substring matching is why `mean_squared_error` matches nothing while
`"mse" in "rmse"` is True — the current hint tuple survives that only by listing
`rmse` first, which is a coincidence rather than a design.

And **direction becomes a property of the metric, not of the spec that names it**,
so `key="rmse", direction="maximize"` is unrepresentable. `MetricSpec.direction`
defaults to `"maximize"` today, and `direction.py` exists because that default
caused every one of rogii's fifteen evidence cards to be built as though MSE were
maximized — recording its single genuine improvement as `rejected`.

## Exit criteria

1. ✅ A scorecard exists, is reproducible across two runs except its timestamp, and
   is pinned to a corpus hash. `RATCHET.json` carries the digest; `bench score`
   prints it.
2. ✅ Thresholds are a **ratchet starting at today's measured value**, with 95%
   recorded as the goal. Asserting 95% on day one makes the suite red and teaches
   everyone to ignore it. Recorded 2026-08-22; it fails in both directions, so an
   improvement is raised rather than absorbed.
3. The ledger is **bidirectional** — CI fails on `pass → fail` *and* on
   `known_fail → pass` (*"this now passes; update the ledger"*). Silently
   absorbing improvements is how a ratchet rots.
4. Adding a competition requires **zero test-file edits**.
5. Every `known_failure` carries a reason and a date; a stale xfail is a lie about
   intent.
6. Tier 1 and tier 2 agree on every tier-1 criterion for every fixture, or the
   disagreeing criterion is demoted to `unverifiable`.
7. `pytest.skip` when the full-data cache is absent is **loud** — never a silent
   pass. `capability.py:333` documents exactly that bug.

## Traps

- **A media fixture that is proportionally shrunk validates the wrong answer.**
  Use the listing plus placeholders.
- **Do not score `row_count` anywhere** until the 100k sampling cap is fixed.
- **RL does not belong in this corpus** beyond the single boundary case. Three of
  the six criteria have no referent for an environment competition, and forcing
  it in corrupts the denominators.
- Under the placeholder design, image and audio modality detection need no
  decoder, so they belong in the **default** CI job — making them blocking for
  the first time. Today's `image` and `deep` jobs are `continue-on-error: true`
  running import smoke tests.
