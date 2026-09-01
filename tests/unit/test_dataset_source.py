"""The seam between where a dataset lives and what is inferred about it.

M22 step 1. Two things are worth proving here and nowhere else: that
`LocalFileSource` answers the questions the protocol promises, and that the
profiler actually goes through it — a seam nothing uses is a comment.

The step's other claim, that routing changed no values, is proved by the golden
snapshots in `test_dataset_shapes.py`, which were taken before the refactor.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest
from helpers.dataset_sources import DictSource

from labpilot.accessor.profiler import tabular as tabular_module
from labpilot.accessor.profiler.source import (
    DatasetSource,
    DeclaredFacts,
    LocalFileSource,
    TableRef,
)
from labpilot.accessor.profiler.tabular import TabularProfiler
from labpilot.config import ProfilerConfig

#: Names that read data. `open` is in here because it is the one a hurried edit
#: reaches for, and it parses as a bare name rather than an attribute.
_READERS = frozenset({"read_csv", "read_parquet", "read_json", "open"})


def test_local_file_source_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(LocalFileSource(tmp_path), DatasetSource)


def test_tables_are_listed_in_one_stable_order(strong_signals_data_dir: Path) -> None:
    """`files` in the profile has always been sorted; a description that
    reorders itself between runs cannot be compared against itself."""
    source = LocalFileSource(strong_signals_data_dir)
    uris = [table.uri for table in source.tables()]

    assert uris == ["sample_submission.csv", "test.csv", "train.csv"]
    assert uris == sorted(uris)


def test_uris_are_relative_to_the_dataset_root(
    partitioned_with_template_data_dir: Path,
) -> None:
    """A uri is what the profile records, so it must not leak the machine's
    directory layout into a workspace artifact."""
    source = LocalFileSource(partitioned_with_template_data_dir)
    uris = [table.uri for table in source.tables()]

    assert "train/w001__horizontal_well.csv" in uris
    assert not any(uri.startswith("/") for uri in uris)


def test_columns_reads_the_header_without_the_rows(
    sampled_beyond_cap_data_dir: Path,
) -> None:
    source = LocalFileSource(sampled_beyond_cap_data_dir)
    train = TableRef(uri="train.csv")

    assert source.columns(train) == ["id", "feature", "label"]


def test_sample_caps_and_exact_count_does_not(sampled_beyond_cap_data_dir: Path) -> None:
    """The distinction the profile currently cannot express.

    `sample` is bounded by what a profiler can afford to read; `exact_unit_count`
    is the true number. Reporting the first as the second is the defect in
    `playground-series-s6e7/profile.json` (100,000 rows recorded for 690,088).
    """
    source = LocalFileSource(sampled_beyond_cap_data_dir)
    train = TableRef(uri="train.csv")
    real_rows = len(pd.read_csv(sampled_beyond_cap_data_dir / "train.csv"))

    assert real_rows > 10, "the fixture must exceed the cap or this proves nothing"
    assert len(source.sample(train, 10)) == 10
    assert len(source.sample(train, None)) == real_rows
    assert source.exact_unit_count(train, "id") == real_rows


def test_declared_facts_round_trip(tmp_path: Path) -> None:
    """What the environment states about itself, carried but not believed."""
    source = LocalFileSource(tmp_path, DeclaredFacts(title="Rogii", description="wellbore"))

    assert source.declared().title == "Rogii"
    assert source.declared().description == "wellbore"
    assert LocalFileSource(tmp_path).declared() == DeclaredFacts()


def test_the_profiler_reads_only_through_the_source(
    monkeypatch, partitioned_with_template_data_dir: Path
) -> None:
    """Every table read during a profile goes through the protocol's methods.

    Recorded rather than asserted from the source text: a spy subclass sees the
    reads that actually happen, including any a future edit adds. The profile it
    produces is compared against the unpatched one, so the spy cannot pass by
    changing what is read.
    """
    calls: list[tuple[str, str]] = []

    class RecordingSource(LocalFileSource):
        def columns(self, table: TableRef) -> list[str]:
            calls.append(("columns", table.uri))
            return super().columns(table)

        def sample(self, table: TableRef, limit: int | None) -> pd.DataFrame:
            calls.append(("sample", table.uri))
            return super().sample(table, limit)

        def exact_unit_count(self, table: TableRef, column: str) -> int:
            calls.append(("exact_unit_count", table.uri))
            return super().exact_unit_count(table, column)

    expected = TabularProfiler(ProfilerConfig()).profile_directory(
        partitioned_with_template_data_dir, "partitioned"
    )
    monkeypatch.setattr(tabular_module, "LocalFileSource", RecordingSource)
    actual = TabularProfiler(ProfilerConfig()).profile_directory(
        partitioned_with_template_data_dir, "partitioned"
    )

    assert json.loads(actual.model_dump_json()) == json.loads(expected.model_dump_json())
    # Every sampled training partition, and every test table's header.
    sampled = {uri for kind, uri in calls if kind == "sample"}
    headers = {uri for kind, uri in calls if kind == "columns"}
    assert "train/w001__horizontal_well.csv" in sampled
    assert "train/t001__typewell.csv" in sampled
    assert "test/w001__horizontal_well.csv" in headers
    assert "sample_submission.csv" in headers


def test_the_profiler_holds_no_read_of_its_own() -> None:
    """One owner for every read, checked structurally.

    The seam is worth nothing if the next edit adds a read beside it — which is
    how the profiler ended up with nine of them across two paths. Reads belong
    in a source; this module infers.

    Over the parse tree rather than the text, so the rule is about calls and not
    about the word: a docstring naming the function it forbids should not be a
    failure, and a call spelled `pandas.read_csv` should not be a pass.
    """
    tree = ast.parse(Path(tabular_module.__file__).read_text(encoding="utf-8"))
    # Both call shapes. Matching only `ast.Attribute` missed the likeliest
    # violation there is — a bare `open(path)`, which parses as `ast.Name` — so
    # the guard against direct reads could not see the most direct read.
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr in _READERS)
            or (isinstance(node.func, ast.Name) and node.func.id in _READERS)
        )
    ]

    assert not reads, (
        "tabular.py must read through a DatasetSource; add the read to "
        f"accessor/profiler/source.py instead (line {reads[0].lineno if reads else 0})"
    )


def test_a_uri_cannot_leave_the_dataset_root(strong_signals_data_dir: Path) -> None:
    """`root / uri` silently discards the root when the uri is absolute.

    No uri from `tables()` can do this today. The check exists because step 3
    starts building tables from operator answers and model proposals, and a
    boundary that only holds while every caller is trusted is not one.
    """
    source = LocalFileSource(strong_signals_data_dir)

    assert source.path(TableRef(uri="train.csv")).name == "train.csv"
    for escape in ("/etc/passwd", "../../../etc/passwd"):
        with pytest.raises(ValueError, match="outside the dataset root"):
            source.path(TableRef(uri=escape))


def test_a_symlinked_table_inside_the_root_is_still_readable(tmp_path: Path) -> None:
    """The boundary is about which file a uri may address, not about inodes.

    Keeping large partitions on another volume and linking them in is an
    ordinary way to lay a dataset out, and `pd.read_csv` followed such a link
    for as long as the profiler had one. A guard that resolved before comparing
    refused uris `tables()` had just listed, and the workspace layer's broad
    `except` turned that into a silent fall back to a filesystem inventory.

    Both link shapes are checked through `path()`, because whether `tables()`
    descends into a symlinked *directory* is a property of the interpreter —
    `rglob` recursed into one until 3.13 and does not from 3.13 on — while the
    uri naming it is one a caller can build on any version.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    pd.DataFrame({"id": [1, 2], "label": [0.0, 1.0]}).to_csv(elsewhere / "w001.csv", index=False)
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "train" / "w001.csv").symlink_to(elsewhere / "w001.csv")
    (root / "extra").symlink_to(elsewhere, target_is_directory=True)

    source = LocalFileSource(root)

    # The linked file is listed, and every uri `tables()` lists, `path()` must
    # accept — that is the invariant the docstring claims and the one that broke.
    uris = {table.uri for table in source.tables()}
    assert "train/w001.csv" in uris
    for uri in uris:
        assert source.columns(TableRef(uri=uri)) == ["id", "label"]
    assert len(source.sample(TableRef(uri="train/w001.csv"), None)) == 2

    # And through a linked directory, whether or not this interpreter walked it.
    assert source.columns(TableRef(uri="extra/w001.csv")) == ["id", "label"]


def test_dot_dot_is_collapsed_without_consulting_the_filesystem(tmp_path: Path) -> None:
    """`..` is refused lexically, so the guard does not depend on what exists.

    A uri naming a directory that is not there resolved to a plausible path
    under the old check too, but the point here is that the refusal is decided
    by the string: step 3's untrusted callers get the same answer whether or not
    the intermediate directories happen to have been created yet.
    """
    root = tmp_path / "dataset"
    root.mkdir()
    source = LocalFileSource(root)

    for escape in ("train/../../secrets.csv", "does/not/exist/../../../../etc/passwd"):
        with pytest.raises(ValueError, match="outside the dataset root"):
            source.path(TableRef(uri=escape))
    # Interior `..` that stays inside is fine, and is not a filesystem question.
    assert source.path(TableRef(uri="train/../train.csv")) == (root / "train.csv").resolve()


def test_a_dataset_that_is_not_a_directory_can_be_profiled() -> None:
    """The seam, exercised end to end by something that owns no files.

    This is what `profile_dataset` is for: M12's warehouse, object store or
    environment adapter is profiled by passing it here, with no edit to the
    profiler. Before it existed, `profile_directory` built its own
    `LocalFileSource` and there was no way in.
    """
    train = pd.DataFrame(
        {"Id": [1, 2, 3], "LotArea": [8450.0, 9600.0, 11250.0], "SalePrice": [1.0, 2.0, 3.0]}
    )
    source = DictSource(
        {
            "train.csv": train,
            "test.csv": train[["Id", "LotArea"]],
            "sample_submission.csv": train[["Id", "SalePrice"]],
        },
        DeclaredFacts(title="In memory", description="no files anywhere"),
    )

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(source, "in-memory")

    assert profile.target_column == "SalePrice"
    assert profile.id_column == "Id"
    assert profile.train_file == "train.csv"
    assert profile.row_count == 3
    assert profile.test_row_count == 3
    assert [column.name for column in profile.columns] == ["Id", "LotArea", "SalePrice"]
    # Skipped, and said so. A silent `modality="tabular"` would read as a
    # finding rather than as "never looked" (M14).
    assert any("modality not detected" in warning for warning in profile.warnings)
