"""Dataset shapes the profiler must describe, as builders.

M22 step 0. Each builder writes one *shape* — not one competition. The shapes
are chosen so that the evidence available to a profiler differs sharply between
them, because that difference is what M22's confidence has to express:

* :func:`build_strong_signals` — every strong signal present (house-prices).
* :func:`build_partitioned_with_template` — per-entity tables plus a prediction
  template, with a second table kind carrying a column of the target's name.
  This is the shape that made rogii infer a horizon depth as its label.
* :func:`build_partitioned_without_template` — the same shape one file poorer.
  Two withheld columns are then indistinguishable on evidence, and today's
  answer is decided by a sort.
* :func:`build_no_kaggle_inputs` — one table. No test split, no template, no
  declared metric: the M12 case, where every structural signal is unavailable
  by construction.
* :func:`build_bool_target` — spaceship-titanic's shape; a boolean label and a
  string id, so "target" cannot mean "the numeric one".
* :func:`build_sampled_beyond_cap` — more rows than the sampling cap, which the
  profile currently reports as an exact count.
* :func:`build_environment` — no tabular data at all.

Row values are fixed rather than random: a fixture whose bytes change between
runs cannot be used to check that the profiler's output does not.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "build_bool_target",
    "build_environment",
    "build_image_and_text",
    "build_partition_suffix",
    "build_no_kaggle_inputs",
    "build_partitioned_with_template",
    "build_partitioned_without_template",
    "build_sampled_beyond_cap",
    "build_strong_signals",
    "build_tables_with_previews",
    "build_template_only",
]

#: Entities per kind in the partitioned shapes. The primary kind needs at least
#: `_MIN_PARTITIONS` (3) files to be read as partitioned at all, and the two
#: kinds are given *different* counts so which one is primary is a fact about
#: the fixture rather than about dict ordering.
_PRIMARY_ENTITIES = ("w001", "w002", "w003", "w004", "w005")
_SECONDARY_ENTITIES = ("t001", "t002", "t003")


def build_strong_signals(root: Path) -> Path:
    """Case A — a single train/test pair plus a prediction template.

    Every strong signal fires: the template names the label, the label is
    absent from the scoring input, it is numeric and complete. The profiler
    gets this right today, and the point of the fixture is that it should be
    able to *say why*.
    """
    data_dir = root / "strong-signals"
    data_dir.mkdir(parents=True)

    train = pd.DataFrame(
        {
            "Id": range(1, 13),
            "LotArea": [
                8450,
                9600,
                11250,
                9550,
                14260,
                14115,
                10084,
                10382,
                6120,
                7420,
                11200,
                9000,
            ],
            "Neighborhood": ["CollgCr", "Veenker", "CollgCr", "Crawfor"] * 3,
            "YearBuilt": [2003, 1976, 2001, 1915, 2000, 1993, 2004, 1973, 1931, 1939, 1965, 2005],
            "SalePrice": [
                208500.0,
                181500.0,
                223500.0,
                140000.0,
                250000.0,
                143000.0,
                307000.0,
                200000.0,
                129900.0,
                118000.0,
                157000.0,
                232000.0,
            ],
        }
    )
    test = pd.DataFrame(
        {
            "Id": range(13, 17),
            "LotArea": [11622, 14267, 13830, 9978],
            "Neighborhood": ["NAmes", "Gilbert", "NWAmes", "Gilbert"],
            "YearBuilt": [1961, 1958, 1997, 1998],
        }
    )
    sample_submission = pd.DataFrame({"Id": test["Id"], "SalePrice": [0.0] * len(test)})

    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir


def _partition_frames(entity_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One primary-kind entity's train and test frames.

    `Depth_input` is the target's known prefix — equal to `Depth` wherever it is
    present, absent exactly on the scored tail. rogii's `TVT_input` has this
    shape, is the strongest predictor in the dataset, and reached the profile as
    an ordinary sparse numeric column.
    """
    base = entity_index * 100
    rows = 10
    depth = [float(base + i) for i in range(rows)]
    # Nulls are a contiguous suffix: the scored region.
    depth_input = [*depth[:6], *[None] * 4]
    train = pd.DataFrame(
        {
            "id": [f"e{entity_index}_{i}" for i in range(rows)],
            "md": [float(i) for i in range(rows)],
            "azimuth": [float(base % 360 + i) for i in range(rows)],
            "Depth_input": depth_input,
            # Two withheld columns, indistinguishable on evidence: both float,
            # both complete, both in every partition of this kind. Named so that
            # a sort over the candidates does *not* land on the label.
            "Depth": depth,
            "Zone_Depth": [value + 0.5 for value in depth],
        }
    )
    test = pd.DataFrame(
        {
            "id": [f"e{entity_index}_t{i}" for i in range(4)],
            "md": [float(i) for i in range(4)],
            "azimuth": [float(base % 360 + i) for i in range(4)],
            "Depth_input": [float(base + i) for i in range(4)],
        }
    )
    return train, test


def _write_partitioned(data_dir: Path) -> Path:
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "test").mkdir()
    for index, entity in enumerate(_PRIMARY_ENTITIES):
        train, test = _partition_frames(index)
        train.to_csv(data_dir / "train" / f"{entity}__horizontal_well.csv", index=False)
        test.to_csv(data_dir / "test" / f"{entity}__horizontal_well.csv", index=False)

    # The second kind carries columns of the *withheld columns' own names*, and
    # ships on both sides. In rogii this is what stopped the label looking
    # withheld: column roles were unioned across kinds, so the horizontal well's
    # label was masked by the type well's same-named column.
    #
    # Both withheld names, not just the label's: a decoy that is absent here
    # would be 26% null in the concatenated training frame while the label is
    # complete, and completeness is evidence. The two have to be identical on
    # every recorded fact for the tie in case B′ to be a real one.
    for index, entity in enumerate(_SECONDARY_ENTITIES):
        depth = [float(1000 + index * 10 + i) for i in range(6)]
        marker = pd.DataFrame(
            {
                "id": [f"{entity}_{i}" for i in range(6)],
                "md": [float(i) for i in range(6)],
                "formation": ["shale", "sand", "shale", "sand", "shale", "sand"],
                "Depth": depth,
                "Zone_Depth": [value + 0.5 for value in depth],
            }
        )
        marker.to_csv(data_dir / "train" / f"{entity}__typewell.csv", index=False)
        marker.to_csv(data_dir / "test" / f"{entity}__typewell.csv", index=False)
    return data_dir


def build_partitioned_with_template(root: Path) -> Path:
    """Case B — per-entity tables, and a template that names the label.

    The template is the only thing separating this from case B′, and it is worth
    0.80 on its own. rogii is this shape with 1,546 tables.
    """
    data_dir = root / "partitioned-with-template"
    _write_partitioned(data_dir)
    # Lower-cased, as rogii's is: the header is `id,tvt` for a label spelled
    # `TVT`, so the match has to be case-insensitive to fire at all.
    pd.DataFrame({"id": ["e0_t0", "e0_t1"], "depth": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    return data_dir


def build_partitioned_without_template(root: Path) -> Path:
    """Case B′ — the same shape with no template.

    `Depth` and `Zone_Depth` are then equally supported: both withheld at test,
    both float, both complete, both in every partition. There is no evidence
    that separates them, which is the honest answer; today one is chosen by
    sorting the candidates and taking the last.
    """
    data_dir = root / "partitioned-without-template"
    _write_partitioned(data_dir)
    return data_dir


def build_no_kaggle_inputs(root: Path) -> Path:
    """Case C — one table. No test split, no template, no declared metric.

    The M12 shape: a warehouse extract, a study export, a log dump. Every
    structural signal the profiler relies on is unavailable by construction, so
    the only honest answers come from a declaration or a human.
    """
    data_dir = root / "no-kaggle-inputs"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": [f"ev{i}" for i in range(1, 13)],
            "account": ["a", "b", "c", "d"] * 3,
            "amount": [12.5, 99.0, 3.25, 47.75, 18.0, 62.5, 5.0, 130.25, 21.0, 8.75, 44.0, 71.5],
            "occurred_at": [f"2026-03-{day:02d}" for day in range(1, 13)],
            "churned": [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        }
    ).to_csv(data_dir / "events.csv", index=False)
    return data_dir


def build_bool_target(root: Path) -> Path:
    """A boolean label and a string id — spaceship-titanic's shape.

    Included because "the target is the numeric one" is a rule that works on
    house-prices and fails here, and because `is_numeric` treats bool as
    categorical deliberately (`ColumnProfile.is_numeric`).
    """
    data_dir = root / "bool-target"
    data_dir.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "PassengerId": [f"{i:04d}_01" for i in range(1, 13)],
            "HomePlanet": ["Europa", "Earth", "Mars"] * 4,
            "Age": [39.0, 24.0, 58.0, 33.0, 16.0, 44.0, 26.0, 28.0, 35.0, 14.0, 34.0, 45.0],
            "RoomService": [0.0, 109.0, 43.0, 0.0, 303.0, 0.0, 42.0, 0.0, 0.0, 0.0, 0.0, 39.0],
            "Transported": [
                False,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ],
        }
    )
    test = pd.DataFrame(
        {
            "PassengerId": [f"{i:04d}_01" for i in range(13, 17)],
            "HomePlanet": ["Earth", "Europa", "Mars", "Earth"],
            "Age": [27.0, 19.0, 31.0, 38.0],
            "RoomService": [0.0, 0.0, 12.0, 0.0],
        }
    )
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": [False] * 4}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    return data_dir


#: Rows written by `build_sampled_beyond_cap`. The defect is that the sampling
#: cap binds and the profile reports the sample size as exact; reproducing it at
#: the real default (100,000) would cost seconds per test to prove a property
#: that holds at any cap. Tests pass a `ProfilerConfig` whose cap is below this.
SAMPLED_BEYOND_CAP_ROWS = 25


def build_sampled_beyond_cap(root: Path) -> Path:
    """A table with more rows than the profiler will read.

    `playground-series-s6e7/profile.json` records `row_count: 100000` with no
    estimate flag for a file of 690,088 rows.
    """
    data_dir = root / "sampled-beyond-cap"
    data_dir.mkdir(parents=True)
    rows = SAMPLED_BEYOND_CAP_ROWS
    pd.DataFrame(
        {
            "id": range(1, rows + 1),
            "feature": [float(i) for i in range(rows)],
            "label": [float(i % 3) for i in range(rows)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": range(rows + 1, rows + 5), "feature": [1.0, 2.0, 3.0, 4.0]}).to_csv(
        data_dir / "test.csv", index=False
    )
    pd.DataFrame({"id": range(rows + 1, rows + 5), "label": [0.0] * 4}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    return data_dir


def build_partition_suffix(root: Path) -> Path:
    """A forecast: the scored rows are a contiguous tail of each partition.

    The split that makes a random CV meaningless — at inference the model holds
    the head of each partition and must predict forward, so validation has to
    reproduce the gap rather than sampling rows uniformly.
    """
    data_dir = root / "partition-suffix"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "test").mkdir()
    for entity in ("w0", "w1", "w2"):
        pd.DataFrame(
            {"MD": [1.0, 2.0, 3.0, 4.0], "GR": [1.0, 2.0, 3.0, 4.0], "TVT": [1.0, 2.0, 3.0, 4.0]}
        ).to_csv(data_dir / "train" / f"{entity}__main.csv", index=False)
        pd.DataFrame({"MD": [1.0, 2.0, 3.0, 4.0], "GR": [1.0, 2.0, 3.0, 4.0]}).to_csv(
            data_dir / "test" / f"{entity}__main.csv", index=False
        )
    pd.DataFrame(
        {"id": [f"{e}_{i}" for e in ("w0", "w1", "w2") for i in (2, 3)], "TVT": [0.0] * 6}
    ).to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir


def build_template_only(root: Path) -> Path:
    """Train and a template, and no column withheld between them.

    The one shape where position is the only thing left to go on: nothing is
    missing from the scoring input, so the label can only be guessed from where
    it sits in the template. Capped at 0.50 by the catalogue, which is what
    makes it ask instead of answer.
    """
    data_dir = root / "template-only"
    data_dir.mkdir(parents=True)
    frame = pd.DataFrame({"Id": [1, 2, 3, 4], "y": [0.5, 1.5, 2.5, 3.5]})
    frame.to_csv(data_dir / "train.csv", index=False)
    frame.to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir


def build_tables_with_previews(root: Path) -> Path:
    """Per-entity tables *and* a directory of image previews.

    rogii's shape: 1,546 well logs beside PNG previews of the same wells. The
    tables carry the signal and the previews do not, which is why preferring
    tabular is right — and why discarding them was not, since nothing
    downstream could then know they existed.

    Written as zero-byte `.png` files: the detector counts extensions and never
    opens them, so a real encoder would cost a dependency for nothing.
    """
    data_dir = _write_partitioned(root / "tables-with-previews")
    previews = data_dir / "previews"
    previews.mkdir()
    for entity in _PRIMARY_ENTITIES[:3]:
        (previews / f"{entity}.png").write_bytes(b"")
    pd.DataFrame({"id": ["e0_t0", "e0_t1"], "depth": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    return data_dir


def build_image_and_text(root: Path) -> Path:
    """Images *and* a long free-text column, with neither outnumbering the other.

    The one shape the rule-based detector cannot settle: it finds both and has
    no rule that prefers either, so it asks a model. Whatever the model says is
    capped, because a tie broken by a sentence is not a tie resolved by the data.
    """
    data_dir = root / "image-and-text"
    images = data_dir / "images"
    images.mkdir(parents=True)
    for index in range(1, 21):
        (images / f"{index}.png").write_bytes(b"")

    def note(index: int) -> str:
        # Long enough to pass the average-length threshold, and distinct enough
        # to pass the cardinality one: a column of four repeated sentences reads
        # as a category, which is the rule doing its job.
        return (
            "a long passage of prose about specimen number "
            f"{index}, comfortably past the length at which the detector calls "
            "a column text rather than a label"
        )

    train = pd.DataFrame(
        {
            "id": range(1, 13),
            "file": [f"{i}.png" for i in range(1, 13)],
            "notes": [note(i) for i in range(1, 13)],
            "label": [0, 1] * 6,
        }
    )
    test = pd.DataFrame(
        {
            "id": range(13, 17),
            "file": [f"{i}.png" for i in range(13, 17)],
            "notes": [note(i) for i in range(13, 17)],
        }
    )
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "label": [0] * 4}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    return data_dir


def build_environment(root: Path) -> Path:
    """No tabular data at all — an interactive environment competition.

    ConnectX's shape. `profile_directory` raises on it; the workspace layer
    catches that and writes a filesystem inventory instead, which reports a
    modality and a null target with nothing behind either.
    """
    data_dir = root / "environment"
    (data_dir / "env").mkdir(parents=True)
    (data_dir / "main.py").write_text("def agent(observation, configuration):\n    return 0\n")
    (data_dir / "env" / "spec.json").write_text('{"name": "connectx", "columns": 7, "rows": 6}\n')
    return data_dir
