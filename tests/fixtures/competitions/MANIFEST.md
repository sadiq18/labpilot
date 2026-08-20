# Competition corpus

M24. Understanding is measured, not asserted — so the system is scored against
**real** competitions rather than hand-typed ones. `titanic_data_dir` is twelve
rows somebody wrote; a fixture here is the header set of the actual file, with
the sha256 of the source beside it.

Each fixture carries what was kept, what was lost, and what it cannot prove. A
capture that does not say what it lost is a paraphrase, which
`tests/fixtures/real_failures/MANIFEST.md` exists because of: a 79-byte fragment
typed from memory sat in that corpus for a day claiming to be 624 bytes.

| slug | mode | source | scores | ships red |
|---|---|---|---|---|
| `playground-series-s6e7` | `headers_only` | 91 MB, 690,088 train rows | target · id · train/test · modality | **`metric_name`** — see below |
| `rogii-wellbore-geology-prediction` | `headers_only`, ≤6 files per directory | 1.2 GB, 1,546 tables | target · modality · **abstention** | — |
| `titanic` | `headers_only` | 891 train rows, 12 columns | target · id · train/test · modality | — |
| `spaceship-titanic` | `headers_only` | 8,693 train rows, 14 columns | target · id · train/test · modality | — |
| `house-prices-advanced-regression-techniques` | `headers_only` | 1,460 train rows, 81 columns | target · id · train/test · modality | — |

## What each one is for

**`playground-series-s6e7`** is the metric-synonym fixture. Its captured
`competition.json` reads `{"name": "balanced_accuracy_score", "key": "accuracy"}`
— written 2026-07-30 by the substring mapper #145 replaced. Today's parser
resolves the name correctly, and **nothing re-derives a stored spec**, so every
workspace captured before #145 still optimises the wrong number. The fixture
ships `known_failure` on that criterion so the day it goes green is visible.

**`rogii-wellbore-geology-prediction`** is the one the system got wrong for
eleven days. Two table kinds, per-kind withholding, and a submission whose `id`
column exists in no table — so its expectation is not an answer but a *question*:
`must_ask: ["id_columns"]`. It scores `abstention`, and passes.

**`titanic`** names its submission template `gender_submission.csv`. A profiler
that finds the template by filename convention rather than by what the file
holds resolves nothing here, and target, id and submission columns fall with it.
It passes today; the fixture is what keeps that true.

**`spaceship-titanic`** is the one with a boolean target (`True`/`False`, not
0/1) and an id that is not a number (`0001_01`). Every other fixture in the
corpus has an integer id, so a rule that only holds for integers passes all of
them.

**`house-prices-advanced-regression-techniques`** is the widest schema here — 81
columns — and the only continuous target, against three classification fixtures.

These three carry **no `competition.json`**, so `metric_name` is `unverifiable`
rather than scored: the cached download holds the data and not the rules, and a
metric typed in from memory would be the paraphrase this corpus exists to
prevent. Capturing a spec beside them upgrades that criterion from
`unverifiable` to scored, for all three at once.

## Adding to it

    research bench capture <path-to-data> --slug <slug> --spec <competition.json>

Then fill `expected` by hand from the competition's own rules page — never from
what the profiler produced, which would score the system against itself.

State the mode. `headers_only` buys target, id, submission columns, train-only
columns and file roles at ~400 bytes and no licence risk. Rows buy four more
things — dtype, cardinality, anchor equality, suffix contiguity — so they are a
per-fixture justification, not a policy. A dataset whose scored rows are a
partition tail needs `stride:<k>`, because `head:<n>` does not truncate that
fact, it inverts it: `_detect_suffix_scoring` reads absolute row indices.

Whatever the capture destroys goes in `unverifiable` with a reason, and the
scorer refuses to score it. A criterion scored against a truncation artifact
measures the capture, not the profiler.

## Licence

`redistribution` is a **constraint on the fixture**, not a note beside it. Both
entries here are `forbidden` — one Kaggle-licensed, one a private sponsor
dataset — and a forbidden fixture may carry column names and no data rows. That
is checked (`test_a_fixture_honours_the_licence_it_declares`), because a field
saying "do not redistribute" inside the commit that redistributes it is worse
than no field at all.

Set it honestly when you capture. `allowed` earns a fixture the right to carry
rows, which buys dtype, cardinality, anchor equality and suffix contiguity —
four criteria headers cannot reach.
