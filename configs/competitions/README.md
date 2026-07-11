# Competition contracts (local only)

LabPilot is meant to work for *any* Kaggle competition, so this repository does
not ship any competition-specific data. Instead, each competition's contract
lives in a local, untracked file:

```
configs/competitions/<slug>.yaml
```

`<slug>` is the Kaggle competition slug you pass to `--competition`
(e.g. the slug in `https://www.kaggle.com/competitions/<slug>`).

These files are git-ignored on purpose — see the pending "generic competition
metadata resolution" task in `docs/MILESTONES.md`. The long-term direction is
for the CLI/agent to resolve this contract automatically from the Kaggle URL
or slug (via the Kaggle API/portal). Until that lands, create the file
yourself for whatever competition you are running.

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

Only `problem_type`, `evaluation_metric`, and `submission_columns` are used by
the pipeline today; the rest is informational context passed to the brief and
reflection generators.
