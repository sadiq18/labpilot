# Competition contracts (local, optional overrides)

LabPilot works for *any* Kaggle competition without a hand-written file. When
you run `research run --competition <slug>`, LabPilot automatically resolves
title, description, and evaluation metric from the Kaggle API, and infers the
problem type (classification vs. regression) and file roles (train/test/
sample submission) directly from the downloaded data.

A local, untracked file still exists for the cases automatic resolution can't
cover:

```
configs/competitions/<slug>.yaml
```

`<slug>` is the Kaggle competition slug you pass to `--competition`
(e.g. the slug in `https://www.kaggle.com/competitions/<slug>`).

Use one when you need to:

- Override file-naming patterns for a competition whose files don't follow
  the `train*` / `test*` / `*submission*` convention the profiler assumes.
- Force a specific `problem_type` if the automatic inference from the target
  column's dtype/cardinality guesses wrong for an unusual dataset.
- Pin a `title`/`description` for the brief and reflection when the
  Kaggle-search-based lookup can't find or disambiguate the competition
  (very generic slugs sometimes match dozens of unrelated public
  competitions and are skipped rather than risk resolving to the wrong one).

These files are git-ignored on purpose — every user's local overrides are
their own.

## Schema

```yaml
title: <competition display name>
description: <one paragraph summary>
problem_type: tabular_classification | tabular_regression | text_classification | image_classification | unknown
evaluation_metric:
  name: <metric name, e.g. accuracy, rmse, auc>
  direction: maximize | minimize
  description: <short description of the metric>
submission_format: <comma-separated column names as a string, informational>
submission_columns:
  - <id column>
  - <target column>
tags:
  - <free-form tag>

# Optional overrides. Only needed if a competition's file names don't follow
# the "train*/test*/*submission*" convention that the profiler assumes by
# default (see TabularProfiler.profile_directory).
train_file_pattern: train
test_file_pattern: test
submission_file_pattern: submission
```

Only `problem_type` and `evaluation_metric.name` actually change pipeline
behavior (problem type selects the baseline template; the evaluation metric
is informational only — P0's templates always report `cv_accuracy` or
`cv_rmse` regardless of the competition's real metric). Everything else is
context passed to the brief and reflection generators.
