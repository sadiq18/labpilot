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

from labpilot.accessor.profiler import tabular as tabular_module
from labpilot.accessor.profiler.source import (
    DatasetSource,
    DeclaredFacts,
    LocalFileSource,
    TableRef,
)
from labpilot.accessor.profiler.tabular import TabularProfiler
from labpilot.config import ProfilerConfig


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
    train = TableRef(id="train.csv", uri="train.csv")

    assert source.columns(train) == ["id", "feature", "label"]


def test_sample_caps_and_exact_count_does_not(sampled_beyond_cap_data_dir: Path) -> None:
    """The distinction the profile currently cannot express.

    `sample` is bounded by what a profiler can afford to read; `exact_unit_count`
    is the true number. Reporting the first as the second is the defect in
    `playground-series-s6e7/profile.json` (100,000 rows recorded for 690,088).
    """
    source = LocalFileSource(sampled_beyond_cap_data_dir)
    train = TableRef(id="train.csv", uri="train.csv")
    real_rows = len(pd.read_csv(sampled_beyond_cap_data_dir / "train.csv"))

    assert real_rows > 10, "the fixture must exceed the cap or this proves nothing"
    assert len(source.sample(train, 10)) == 10
    assert len(source.sample(train)) == real_rows
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

        def sample(self, table: TableRef, limit: int | None = None) -> pd.DataFrame:
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
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"read_csv", "read_parquet", "read_json", "open"}
    ]

    assert not reads, (
        "tabular.py must read through a DatasetSource; add the read to "
        f"accessor/profiler/source.py instead (line {reads[0].lineno if reads else 0})"
    )
