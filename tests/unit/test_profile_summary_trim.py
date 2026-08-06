"""The codegen prompt must not enumerate a partitioned dataset.

Measured on rogii 2026-08-07, after a campaign died on a token ceiling:

    profile.json      ~2,902 tokens   47% of the codegen prompt payload
      of which files  ~1,770 tokens   61% of the profile — 200 filenames
    prior train.py    ~3,376 tokens   46%

A single codegen call reached **14,437 tokens** against the 12,000 TPM ceiling
of the free tier serving it, and the run stopped. The filename list is the
cheapest thing to remove because the generated code globs
``data/raw/<split>/*.csv`` at runtime — it needs the naming *convention*, not
an inventory.
"""

from __future__ import annotations

import json

from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
    _FILE_SAMPLE,
    _summarise_profile,
)


def _profile(n_files: int = 200) -> dict:
    return {
        "competition": "rogii",
        "target_column": "TVT",
        "id_column": "row_id",
        "partitioned": True,
        "partition_kinds": {"horizontal_well": 773, "typewell": 773},
        "files": [f"test/{i:06x}__horizontal_well.csv" for i in range(n_files)],
        "columns": [
            {"name": "MD", "dtype": "float64", "null_pct": 0.0, "unique_count": 5278},
            {"name": "TVT", "dtype": "float64", "is_target_candidate": True},
        ],
    }


def test_filenames_are_replaced_by_a_count_and_a_sample():
    trimmed = _summarise_profile(_profile())
    files = trimmed["files"]

    assert files["count"] == 200, "the count must survive — it conveys dataset scale"
    assert len(files["sample"]) == _FILE_SAMPLE
    assert files["sample"][0].endswith(".csv")
    assert "glob" in files["note"], "say why the list is absent, or a reader assumes data loss"


def test_everything_the_model_actually_needs_survives():
    """The trim is only lossless if the fields driving codegen are untouched."""
    original = _profile()
    trimmed = _summarise_profile(original)

    for key in ("target_column", "id_column", "partitioned", "partition_kinds", "columns"):
        assert trimmed[key] == original[key], f"{key} must not be altered"


def test_the_trim_is_large_enough_to_matter():
    """A saving too small to change routing would not be worth the code."""
    original = _profile()
    before = len(json.dumps(original))
    after = len(json.dumps(_summarise_profile(original)))
    assert after < before * 0.5, f"expected >50% smaller, got {after}/{before}"


def test_a_short_file_list_is_left_alone():
    """A non-partitioned competition lists two or three files; summarising those
    would lose information and save nothing."""
    original = _profile(n_files=3)
    assert _summarise_profile(original) == original


def test_the_resolver_still_reads_a_trimmed_profile():
    """`_profile_summary` feeds both the prompt and technique resolution, so the
    trim must not break applicability checks — `requires: [partitioned]` and
    the numeric/categorical derivation both read this dict."""
    from labpilot.research_engine.execution.technique.resolver import resolve_technique

    trimmed = _summarise_profile(_profile())
    choice = type(
        "C", (), {"problem_type": "tabular_regression", "partitioned": True,
                  "template_name": "tabular_regression_partitioned", "validation": None}
    )()
    res = resolve_technique({"technique": "lag_features"}, {}, choice=choice, profile=trimmed)
    assert res.status == "applied", res.reason


def test_malformed_profile_does_not_raise():
    """Called on every write_code; a broken profile must not fail the run."""
    assert _summarise_profile({}) == {}
    assert _summarise_profile({"files": "not-a-list"}) == {"files": "not-a-list"}
    assert _summarise_profile(None) == {}  # type: ignore[arg-type]
