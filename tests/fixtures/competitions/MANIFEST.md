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
