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
    assert "exclude_features" in flags[0]


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


def test_the_shape_this_was_calibrated_on():
    """The file the matching rule was measured against, kept as a fixture so a
    change to that rule has to face it.

    A hardcoded `/Users/sadik/workspace/…` path did this first, which meant it
    ran on one machine and silently skipped on CI and on every other checkout —
    a calibration nobody else could break. Reported on PR #119. The fixture is
    rogii's real structure: seven top-level functions, the split named after the
    scheme, a `main` that delegates to it and records the scheme's name in its
    metrics, and feature code that names the excluded columns to drop them.

    Stored as `.py.txt` so the repo's own linters do not try to hold generated
    competition code to the standards of hand-written source. It is read, not
    imported.
    """
    signals = ValidationSignals.from_baseline_choice(
        json.loads((Path(__file__).parent / "fixtures" / "rogii_baseline_choice.json").read_text())
    )
    tree = _tree((Path(__file__).parent / "fixtures" / "rogii_train.py.txt").read_text())

    assert validation_region(tree, signals) == {"partition_suffix_holdout_split"}
    assert check_leakage_discipline(tree, signals) == []


# --- PR #119 round 2 ---------------------------------------------------------

_KFOLD = ValidationSignals(scheme="kfold", exclude_features=("Geology", "ANCC", "BUDA"))


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("a fold sanity check", "def sanity_check_folds(df):\n    return len(df)\n"),
        ("an n_kfolds parameter", "def report(n_kfolds):\n    return n_kfolds\n"),
        (
            "a benchmark_folds local",
            "def report(df):\n    benchmark_folds = 3\n    return benchmark_folds\n",
        ),
    ],
)
def test_the_scheme_matches_whole_words_not_substrings(label, source):
    """Reported on PR #119, against the *default* scheme rather than a contrived
    one: `kfold` is a substring of `sanity_check_folds` — "che**ck fold**s" — so
    a diagnostic that counts pre-existing folds joined the validation region."""
    assert validation_region(_tree(source), _KFOLD) == set(), label


def test_a_camel_case_splitter_still_matches():
    """The carve-out must not cost the behaviour it guards: `group_kfold` in a
    config is `GroupKFold` in code."""
    source = "def split(df):\n    return GroupKFold(n_splits=5).split(df)\n"

    assert validation_region(_tree(source), ValidationSignals(scheme="group_kfold")) == {"split"}


def test_a_wrapper_with_logic_of_its_own_is_in_the_region():
    """Reported on PR #119: exempting *every* caller from the region — not just
    the entry point — hid a wrapper that reseeds and reshuffles before
    delegating, which changes exactly which rows land in the holdout."""
    source = (
        "def partition_suffix_holdout_split(df):\n    return df.iloc[:5], df.iloc[5:]\n\n\n"
        "def prepare_and_split(df, seed=42):\n"
        "    df = df.sample(frac=1.0, random_state=seed)\n"
        "    return partition_suffix_holdout_split(df)\n\n\n"
        "def main():\n    return prepare_and_split(None)\n"
        '\n\nif __name__ == "__main__":\n    main()\n'
    )

    region = validation_region(_tree(source), _ROGII)

    assert region == {"partition_suffix_holdout_split", "prepare_and_split"}
    assert "main" not in region


# -- the leakage check must read direction, not presence ----------------------


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("an excluded column selected outright", 'def b(df):\n    return df[["Geology", "GR"]]\n'),
        (
            "one excluded, another re-added",
            "def b(df):\n"
            '    cols = [c for c in df.columns if c != "Geology"]\n'
            '    return df[cols + ["ANCC"]]\n',
        ),
        (
            "only one of three excluded",
            'def b(df):\n    return df[[c for c in df.columns if c != "Geology"]]\n',
        ),
        (
            "select_dtypes takes everything",
            'def b(df):\n    return df.select_dtypes(include="number")\n',
        ),
    ],
)
def test_leaking_shapes_are_flagged(label, source):
    """Reported on PR #119, five ways. The check asked whether the file
    *mentioned* an excluded column, which read explicit inclusion as evidence of
    exclusion — the exact leak class it exists to catch, scored as clean."""
    assert check_leakage_discipline(_tree(source), _KFOLD), label


@pytest.mark.parametrize(
    ("label", "source"),
    [
        (
            "names every excluded column",
            "def b(df):\n"
            '    ex = {"Geology", "ANCC", "BUDA"}\n'
            "    return df[[c for c in df.columns if c not in ex]]\n",
        ),
        (
            "reads the key from config",
            "def b(df, choice):\n"
            '    ex = set(choice["validation"]["exclude_features"])\n'
            "    return df[[c for c in df.columns if c not in ex]]\n",
        ),
        ("an explicit allowlist", 'def b(df):\n    return df[["GR", "MD"]]\n'),
    ],
)
def test_correct_shapes_are_not_flagged(label, source):
    assert check_leakage_discipline(_tree(source), _KFOLD) == [], label


def test_writing_a_column_is_not_selecting_it_as_a_feature():
    """`df["Geology"] = …` creates a column; it does not choose one as a
    feature, and Store context is how the difference is read."""
    source = (
        "def b(df):\n"
        '    df["Geology"] = 0\n'
        '    ex = {"Geology", "ANCC", "BUDA"}\n'
        "    return df[[c for c in df.columns if c not in ex]]\n"
    )

    assert check_leakage_discipline(_tree(source), _KFOLD) == []


# -- tolerating anything ------------------------------------------------------


@pytest.mark.parametrize("choice", [["a"], "x", 5, 0.5, True])
def test_a_non_dict_baseline_choice_does_not_raise(choice):
    """Reported on PR #119: `(choice or {}).get(...)` assumed a dict for
    anything truthy, so a top-level list raised `AttributeError` past a caller
    catching `(ValueError, TypeError)` and took the whole write with it."""
    assert ValidationSignals.from_baseline_choice(choice) == ValidationSignals()


@pytest.mark.parametrize("scheme", [True, 5, 0.5, ["kfold"], {"a": 1}])
def test_a_non_string_scheme_yields_no_region(scheme):
    """`str(True)` is `"True"`, which matched `is_true_positive_rate`. A
    malformed value should flag nothing, not match by luck."""
    signals = ValidationSignals.from_baseline_choice({"validation": {"scheme": scheme}})

    assert signals.scheme == ""


def test_a_non_utf8_baseline_choice_does_not_break_the_write(tmp_path):
    """`UnicodeDecodeError` is a `ValueError`, not an `OSError`, so guarding the
    read with `except OSError` let it escape. Reported on PR #119."""
    from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
        _validation_signals,
    )

    (tmp_path / "baseline_choice.json").write_bytes(b"\xff\xfe not utf-8 at all")

    assert _validation_signals(tmp_path) == ValidationSignals()


def test_a_malformed_baseline_choice_does_not_break_the_write(tmp_path):
    from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
        _validation_signals,
    )

    (tmp_path / "baseline_choice.json").write_text("[1, 2, 3]", encoding="utf-8")

    assert _validation_signals(tmp_path) == ValidationSignals()


# -- the flags have to reach a reader -----------------------------------------


def test_flags_travel_from_task_evidence_onto_the_evidence_card(tmp_path):
    """Reported on PR #119, and it made every flag in the system decorative.

    `check_confinement` has written `delta_flags` into the write-code task's own
    `TaskEvidence` since PR #112, and `build_evidence_card` builds from
    `metrics.json` and plan metadata — so nothing, human or system, ever read
    one. A delta that silently moved the validation split was flagged in a file
    no part of the pipeline opens, and then confirmed off the gain that move
    produced. The whole argument for flagging rather than refusing is that a
    reader can discount the result; no reader was being shown anything.
    """
    from labpilot.research_engine.evidence.builder import delta_flags_for
    from labpilot.research_engine.execution.evidence import evidence_dir
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    paths = ResearchPaths(tmp_path, "demo").ensure()
    directory = evidence_dir(paths, "E-001")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "T-write.json").write_text(
        json.dumps({"metadata": {"delta_flags": ["the delta changed the split"]}}),
        encoding="utf-8",
    )

    assert delta_flags_for(tmp_path, "demo", "E-001") == ["the delta changed the split"]


def test_an_execution_with_no_flags_reports_none(tmp_path):
    from labpilot.research_engine.evidence.builder import delta_flags_for

    assert delta_flags_for(tmp_path, "demo", "E-nothing") == []


def test_an_unreadable_evidence_file_does_not_cost_the_card(tmp_path):
    """One corrupt file must not take the other flags — or the card — with it."""
    from labpilot.research_engine.evidence.builder import delta_flags_for
    from labpilot.research_engine.execution.evidence import evidence_dir
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    paths = ResearchPaths(tmp_path, "demo").ensure()
    directory = evidence_dir(paths, "E-002")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "T-bad.json").write_text("{not json", encoding="utf-8")
    (directory / "T-good.json").write_text(
        json.dumps({"metadata": {"delta_flags": ["kept"]}}), encoding="utf-8"
    )

    assert delta_flags_for(tmp_path, "demo", "E-002") == ["kept"]


def test_the_card_carries_the_flags_and_says_so_in_its_reason():
    """Stored *and* surfaced: metadata is not where a verdict is read, and a
    confirmed hypothesis is where a validation-region flag matters most."""
    import inspect

    from labpilot.research_engine.evidence import builder

    source = inspect.getsource(builder.build_evidence_card)

    assert "delta_flags_for(" in source
    assert 'metadata={"delta_flags": flags}' in source


def test_a_baseline_is_not_a_delta_landing_in_the_region():
    """Raised on PR #119 as the region check being inert on a first-ever write.

    It is inert there on purpose: writing the validation split is what a
    baseline *is*, and flagging every baseline for defining one would flag every
    baseline. The question this check asks — did a change land somewhere it did
    not claim to go — has no meaning without a parent.
    """
    baseline = (
        "def partition_suffix_holdout_split(df):\n    return df.iloc[:5], df.iloc[5:]\n\n\n"
        "def main():\n    return partition_suffix_holdout_split(None)\n"
        '\n\nif __name__ == "__main__":\n    main()\n'
    )

    report = check_delta_consistency("", baseline, validation=_ROGII)

    assert not [flag for flag in report.flags if "validation plan" in flag]


def test_leakage_is_still_checked_on_a_first_ever_write():
    """What *does* need saying about a baseline: whether it trains on columns
    the test set will not carry. That needs no parent, so it runs on every
    write."""
    leaky = (
        'def build(df):\n    return df[["Geology", "GR"]]\n\n\n'
        "def main():\n    return build(None)\n"
        '\n\nif __name__ == "__main__":\n    main()\n'
    )

    report = check_delta_consistency("", leaky, validation=_ROGII)

    assert any("Geology" in flag for flag in report.flags)


# --- PR #119 round 3 ---------------------------------------------------------


def test_the_flag_summary_survives_a_leaderboard_patch():
    """Reported on PR #119, against the previous round's own headline fix.

    Three writers recompute `decision_reason` after a card is built —
    `submit_learn` when leaderboard results land, `repair` twice — and each
    overwrote the appended flag text wholesale. It vanished exactly when a
    hypothesis reached the leaderboard, which is the confirmed case the fix
    named as mattering most. The flags live in metadata now and the summary is
    derived, so no writer can drop them by rewriting a sentence.
    """
    from labpilot.research_engine.evidence.models import EvidenceCard

    card = EvidenceCard(
        decision_reason="cv_gain_positive",
        metadata={"delta_flags": ["the delta landed in the validation region"]},
    )
    patched = card.model_copy(update={"decision_reason": "lb_gain_non_negative"})

    assert "validation region" in card.decision_summary
    assert "validation region" in patched.decision_summary
    assert patched.decision_summary.startswith("lb_gain_non_negative")


def test_a_card_without_flags_reads_exactly_as_before():
    from labpilot.research_engine.evidence.models import EvidenceCard

    card = EvidenceCard(decision_reason="cv_gain_positive")

    assert card.decision_summary == "cv_gain_positive"
    assert card.delta_flags == []


def test_the_readers_use_the_summary_not_the_raw_field():
    """A check nothing reads is the failure this milestone exists to end, and
    the summary is only worth deriving if the places a verdict is read use it."""
    import inspect

    from labpilot.research_engine.context.providers import experiments
    from labpilot.research_engine.evidence import apply

    assert "decision_summary" in inspect.getsource(experiments)
    assert "decision_summary" in inspect.getsource(apply)


def test_a_notebook_shaped_script_grants_no_delegation_exemption():
    """Reported on PR #119: "called at module level" made every top-level call
    its own entry point, so a script with no `main()` handed `prepare_and_split`
    back the exemption the same round had just taken away."""
    source = (
        "def partition_suffix_holdout_split(df):\n    return df.iloc[:5], df.iloc[5:]\n\n\n"
        "def prepare_and_split(df, seed=42):\n"
        "    df = df.sample(frac=1.0, random_state=seed)\n"
        "    return partition_suffix_holdout_split(df)\n\n\n"
        "def train_model(a):\n    return a\n\n\n"
        "tr, va = prepare_and_split(None)\ntrain_model(tr)\n"
    )

    assert validation_region(_tree(source), _ROGII) == {
        "partition_suffix_holdout_split",
        "prepare_and_split",
    }


def test_the_guarded_entry_point_is_still_exempt():
    """The carve-out must not cost the behaviour it guards: `main` calls
    everything, so calling says nothing about it."""
    source = (
        "def partition_suffix_holdout_split(df):\n    return df.iloc[:5], df.iloc[5:]\n\n\n"
        "def main():\n    return partition_suffix_holdout_split(None)\n"
        '\n\nif __name__ == "__main__":\n    main()\n'
    )

    assert validation_region(_tree(source), _ROGII) == {"partition_suffix_holdout_split"}


def test_a_boolean_mask_is_not_feature_selection():
    """Reported on PR #119: `df[df["Geology"] > 0]` filters rows, and `ast.walk`
    separated the mask's inner subscript from the `Compare` it belongs to — so
    row filtering read as "the code selects 'Geology' as a feature", a false
    positive in the direction this check was rebuilt to remove."""
    source = (
        "def b(df):\n"
        '    ex = {"Geology", "ANCC", "BUDA"}\n'
        '    d = df[df["Geology"] > 0]\n'
        "    return d[[c for c in d.columns if c not in ex]]\n"
    )

    assert check_leakage_discipline(_tree(source), _KFOLD) == []


def test_a_loc_selection_is_the_same_leak_as_a_plain_one():
    """`df.loc[:, ["Geology", "GR"]]` — a `Tuple` holding the `List`, which the
    first version only looked one level into. Reported on PR #119."""
    source = 'def b(df):\n    return df.loc[:, ["Geology", "GR"]]\n'

    assert check_leakage_discipline(_tree(source), _KFOLD)


def test_frame_detection_is_a_known_miss_on_keys():
    """`df.keys()` is columns by another name, and it was detected for one
    round. It is a miss now, on purpose: `keys` and `items` are dict methods,
    and matching them by name flagged files with no DataFrame in them at all. A
    false positive that fires on ordinary dict code is worse than a miss —
    it is the one that gets the check turned off. Written as a test so the trade
    is visible rather than inferred from a frozenset."""
    source = "def b(df):\n    return df[[c for c in df.keys()]]\n"

    assert check_leakage_discipline(_tree(source), _KFOLD) == []


def test_an_unrelated_items_call_is_not_a_dataframe():
    """Reported on PR #119: `Counter(words).items()` made a file that never
    touches a frame read as deriving features from one."""
    source = (
        "def b(words):\n"
        "    from collections import Counter\n"
        "    return [w for w, n in Counter(words).items()]\n"
    )

    assert check_leakage_discipline(_tree(source), _KFOLD) == []


def test_an_allowlist_survives_an_unrelated_items_call():
    """The worse half of the same bug: a file correctly selecting features by
    name was flagged for a `.items()` call elsewhere in the module."""
    source = (
        "def b(df, mapping):\n"
        "    for key, value in mapping.items():\n"
        "        pass\n"
        '    return df[["GR", "MD"]]\n'
    )

    assert check_leakage_discipline(_tree(source), _KFOLD) == []
