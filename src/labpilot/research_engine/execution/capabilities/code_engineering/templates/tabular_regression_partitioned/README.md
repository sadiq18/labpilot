# tabular_regression_partitioned

Baseline for datasets stored as **one file per entity** (well, store, patient,
device) where the scored rows are a **contiguous suffix** of each test
partition — i.e. the model sees the head of a sequence and must predict
forward.

Chosen automatically when the dataset profile reports `partitioned=True`.

What it does differently from `tabular_regression`:

- loads and concatenates every train partition instead of one `train.csv`
- drops columns unavailable at inference (`validation.exclude_features`)
- validates with `partition_suffix_holdout`: hold out the tail of each
  training partition so validation reproduces the real predict-forward gap.
  A shuffled row split is meaningless here — adjacent rows within a
  partition are near-duplicates.
- adds generic within-partition features: position, distance from the last
  observed row, and the last observed target value (anchor), so the model
  predicts a *correction* rather than an absolute level.
