"""M19 §5's fifth check — a delta landing in the validation region is flagged.

§8 calls this the only risk *"that would hurt"*: a delta may change validation
logic, and a leaky score looks **better**, not worse, so neither the metric nor
the leaderboard says anything is wrong. Four of §5's five checks shipped in
PR #112; this one waited on a design question the plan stated and did not
answer — how to say where the region *is* without a curated list of function
names, the pattern this milestone has rejected four times.

The answer is that the workspace already declared it. `derive_validation_plan`
reads the dataset profile and writes the scheme and the excluded columns into
`baseline_choice.json`, and the region is whichever of the parent's functions
run that scheme. Nothing is maintained by hand.

Both checks are **flags**, never refusals — §8's own wording is *"the mitigation
is detection, not prohibition"*, and every check in `consistency.py` that
refused on names inferred from code has had to be walked back.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from labpilot.research_engine.execution.delta.consistency import (
    ValidationSignals,
    check_delta_consistency,
    check_leakage_discipline,
    check_validation_region,
    validation_region,
)

_ROGII = ValidationSignals(
    scheme="partition_suffix_holdout",
    exclude_features=("ANCC", "Geology", "BUDA"),
)

_PARENT = """
import pandas as pd


def partition_suffix_holdout_split(df, fraction=0.7):
    cut = int(len(df) * fraction)
    return df.iloc[:cut], df.iloc[cut:]


def get_feature_columns(df):
    excluded = {"ANCC", "Geology", "BUDA"}
    return [c for c in df.columns if c not in excluded]


def engineer_features(df):
    df["rolling"] = df["GR"].rolling(5).mean()
    return df


def main():
    df = pd.DataFrame()
    df = engineer_features(df)
    train, val = partition_suffix_holdout_split(df)
    return {"validation_scheme": "partition_suffix_holdout", "n": len(train) + len(val)}


if __name__ == "__main__":
    main()
"""


def _tree(source: str) -> ast.Module:
    return ast.parse(source)


# -- where the region is ------------------------------------------------------


def test_the_region_is_the_code_that_runs_the_declared_scheme():
    assert validation_region(_tree(_PARENT), _ROGII) == {"partition_suffix_holdout_split"}


def test_delegating_to_the_splitter_is_not_being_the_splitter():
    """`main` calls it, and `main` calls everything — leaving the orchestrator in
    the region would put a validation flag on nearly every delta, and a flag on
    everything is a flag nobody reads."""
    assert "main" not in validation_region(_tree(_PARENT), _ROGII)


def test_naming_the_scheme_in_a_string_is_reporting_not_running():
    """Measured on rogii: `main` writes `{"validation_scheme":
    "partition_suffix_holdout"}` into its metrics. That records which scheme
    ran; it does not run one."""
    reporting = 'def main():\n    return {"validation_scheme": "partition_suffix_holdout"}\n'

    assert validation_region(_tree(reporting), _ROGII) == set()


def test_a_function_that_inlines_the_scheme_is_in_the_region():
    """The carve-out must not cost the behaviour it guards."""
    inlined = (
        "def main(df):\n"
        "    partition_suffix_holdout = 0.7\n"
        "    return df.iloc[: int(len(df) * partition_suffix_holdout)]\n"
    )

    assert validation_region(_tree(inlined), _ROGII) == {"main"}


def test_the_scheme_is_matched_however_it_is_spelled():
    """`group_kfold` in a config is `GroupKFold` in code. That difference is
    spelling, not meaning."""
    source = "def split(df):\n    return GroupKFold(n_splits=5).split(df)\n"

    region = validation_region(_tree(source), ValidationSignals(scheme="group_kfold"))

    assert region == {"split"}


def test_a_workspace_that_declared_no_scheme_has_no_region():
    """Empty is the honest answer: nothing was derived, so there is no plan for
    a delta to disturb — not a silent pass on a question never asked."""
    assert validation_region(_tree(_PARENT), ValidationSignals()) == set()


# -- what gets flagged --------------------------------------------------------


def test_a_delta_to_the_split_is_flagged():
    flags = check_validation_region(
        _tree(_PARENT), _tree(_PARENT), ["partition_suffix_holdout_split"], _ROGII
    )

    assert flags
    assert "partition_suffix_holdout" in flags[0]


def test_a_feature_delta_is_not_flagged():
    """The whole value of the check is that this stays quiet — a flag on the
    ordinary case is the failure mode M20 exists for."""
    flags = check_validation_region(_tree(_PARENT), _tree(_PARENT), ["engineer_features"], _ROGII)

    assert flags == []


def test_a_delta_that_introduces_validation_logic_is_flagged():
    """Caught by the *child's* region, not the parent's. Nothing to preserve,
    nothing claimed, and a new split nobody asked for."""
    introduced = (
        "def partition_suffix_holdout_v2(df):\n    return df.iloc[:10]\n\n\n"
        "def engineer_features(df):"
    )
    child = _PARENT.replace("def engineer_features(df):", introduced)

    flags = check_validation_region(
        _tree(_PARENT), _tree(child), ["partition_suffix_holdout_v2"], _ROGII
    )

    assert flags


def test_nothing_touched_flags_nothing():
    assert check_validation_region(_tree(_PARENT), _tree(_PARENT), [], _ROGII) == []


# -- F7: the excluded columns must be excluded by something -------------------


def test_a_file_that_never_mentions_the_exclusions_is_flagged():
    """F7 was enforced structurally until M19 §2 — the Jinja pack skipped
    `column in set(EXCLUDE_FEATURES)` when deriving features. Deleting the pack
    left one bullet in a prompt and no check at all."""
    leaky = "def build(df):\n    return df[[c for c in df.columns]]\n"

    flags = check_leakage_discipline(_tree(leaky), _ROGII)

    assert flags
    assert "leaderboard" in flags[0]


def test_naming_the_columns_satisfies_the_check():
    assert check_leakage_discipline(_tree(_PARENT), _ROGII) == []


def test_reading_the_key_from_config_satisfies_the_check():
    """Code that reads `exclude_features` out of `baseline_choice.json` never
    spells the column names, and is exactly right."""
    source = (
        "def build(df, choice):\n"
        '    excluded = set(choice["validation"]["exclude_features"])\n'
        "    return df[[c for c in df.columns if c not in excluded]]\n"
    )

    assert check_leakage_discipline(_tree(source), _ROGII) == []


def test_an_explicit_allowlist_is_not_flagged():
    """Features chosen by name never touch `.columns`, so the excluded columns
    are absent by construction and there is nothing to exclude."""
    source = 'def build(df):\n    return df[["GR", "MD", "TVD"]]\n'

    assert check_leakage_discipline(_tree(source), _ROGII) == []


def test_nothing_excluded_means_nothing_to_check():
    source = "def build(df):\n    return df[[c for c in df.columns]]\n"

    assert check_leakage_discipline(_tree(source), ValidationSignals(scheme="kfold")) == []


# -- wiring -------------------------------------------------------------------


def test_both_checks_reach_the_report_as_flags():
    """Flags, not violations: they land on the evidence card and refuse
    nothing, which is §8's *"detection, not prohibition"*."""
    child = _PARENT.replace("fraction=0.7", "fraction=0.9")

    report = check_delta_consistency(_PARENT, child, validation=_ROGII)

    assert any("validation plan" in flag for flag in report.flags)
    assert report.ok is True


def test_without_signals_the_report_is_unchanged():
    """A workspace with no baseline choice must not start failing differently."""
    child = _PARENT.replace("fraction=0.7", "fraction=0.9")

    report = check_delta_consistency(_PARENT, child)

    assert not [flag for flag in report.flags if "validation plan" in flag]


def test_the_capability_passes_the_signals_in():
    """The other half of the wiring: a check nothing calls is the failure
    `00-diagnosis.md` opens with."""
    import inspect

    from labpilot.research_engine.execution.capabilities.code_engineering import capability

    assert "_validation_signals(root)" in inspect.getsource(capability.CodeEngineeringCapability)


# -- reading what the workspace declared --------------------------------------


def test_signals_come_from_the_derived_validation_plan():
    choice = {
        "validation": {
            "scheme": "partition_suffix_holdout",
            "group_key": "file_stem_entity",
            "n_splits": 5,
            "holdout_fraction": 0.73,
            "exclude_features": ["ANCC", "Geology"],
        }
    }

    signals = ValidationSignals.from_baseline_choice(choice)

    assert signals.scheme == "partition_suffix_holdout"
    assert signals.exclude_features == ("ANCC", "Geology")


@pytest.mark.parametrize("choice", [None, {}, {"validation": None}, {"validation": []}])
def test_a_missing_or_malformed_plan_yields_empty_signals(choice):
    """Empty signals flag nothing, which is the right answer for a workspace
    whose validation plan was never derived."""
    assert ValidationSignals.from_baseline_choice(choice) == ValidationSignals()


def test_the_real_rogii_workspace_flags_nothing_it_should_not():
    """The calibration this check was tuned against, kept as a test so a change
    to the matching rule has to face the file it was measured on."""
    workspace = Path("/Users/sadik/workspace/rogii-wellbore-geology-prediction")
    choice = workspace / "baseline_choice.json"
    train = workspace / "pipeline" / "train.py"
    if not (choice.is_file() and train.is_file()):
        pytest.skip("rogii workspace not present")

    signals = ValidationSignals.from_baseline_choice(json.loads(choice.read_text()))
    tree = _tree(train.read_text())

    assert validation_region(tree, signals) == {"partition_suffix_holdout_split"}
    assert check_leakage_discipline(tree, signals) == []
