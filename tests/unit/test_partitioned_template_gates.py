"""Technique gates in the template rogii actually uses.

Until now `tabular_regression_partitioned` had **zero** `{% if %}` blocks, so
every hypothesis rendered byte-identical training code and scored MSE 194.80
twelve times. These tests cover the two things that can go wrong now that it
has gates: the code not differing, and the code differing *because it leaks*.

The second is the dangerous one. rogii is **predict-forward** — the scored tail
of each partition has no observed target — so a lag or rolling statistic over
the target would read observed values during training and be NaN for exactly
the rows being scored. That produces a better-looking validation score with no
transfer, which is worse than the flat baseline because it is believable.
"""

from __future__ import annotations

import ast
import hashlib

import pytest

from labpilot.config import TrainingConfig
from labpilot.research_engine.execution.baseline.registry import get_template
from labpilot.research_engine.execution.baseline.selector import BaselineChoice, ValidationPlan
from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen.renderer import (  # noqa: E501
    CodeRenderer,
)
from labpilot.research_engine.execution.technique.resolver import resolve_technique

TEMPLATE = "tabular_regression_partitioned"
GATED = ["lag_features", "rolling_features", "aggregation_features"]

# rogii's real shape, including the columns the validation plan excludes.
ROGII_PROFILE = {
    "target_column": "TVT",
    "id_column": "row_id",
    "partitioned": True,
    "columns": [
        {"name": "MD", "dtype": "float64"},
        {"name": "GR", "dtype": "float64"},
        {"name": "ANCC", "dtype": "float64"},
        {"name": "TVT", "dtype": "float64", "is_target": True},
    ],
}


def _choice() -> BaselineChoice:
    return BaselineChoice(
        problem_type="tabular_regression",
        template_name=TEMPLATE,
        rationale="test",
        target_column="TVT",
        id_column="row_id",
        metric_name="mse",
        sample_submission_file="sample_submission.csv",
        partitioned=True,
        validation=ValidationPlan(
            scheme="partition_suffix_holdout",
            holdout_fraction=0.5,
            exclude_features=["ANCC", "ASTNU", "BUDA"],
        ),
    )


def _render(tmp_path, **kwargs) -> str:
    """One directory always — `run_dir` is baked into the output, so rendering
    variants elsewhere would differ regardless of the technique."""
    run_dir = tmp_path / "run"
    (run_dir / "pipeline").mkdir(parents=True, exist_ok=True)
    choice = _choice()
    template = get_template(choice.problem_type, template_name=TEMPLATE)
    assert template is not None
    CodeRenderer(TrainingConfig()).render(template, choice, run_dir, **kwargs)
    return (run_dir / "pipeline" / "train.py").read_text(encoding="utf-8")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the contract: rogii's hypotheses can finally differ --------------------


@pytest.mark.parametrize("recipe", GATED)
def test_each_gate_changes_the_rendered_code(tmp_path, recipe):
    baseline = _render(tmp_path)
    treated = _render(tmp_path, feature_recipes=[recipe])
    assert _digest(baseline) != _digest(treated), f"{recipe} rendered identical code"


def test_two_techniques_differ_from_each_other(tmp_path):
    """The headline: two hypotheses on rogii now produce two different models."""
    digests = {_digest(_render(tmp_path, feature_recipes=[r])) for r in GATED}
    assert len(digests) == len(GATED), "gated techniques collapsed to the same code"


def test_no_technique_still_renders_the_baseline_byte_for_byte(tmp_path):
    """N5. Also the control: without it the tests above could pass on
    nondeterminism rather than on the technique."""
    assert _digest(_render(tmp_path)) == _digest(_render(tmp_path))


@pytest.mark.parametrize("recipe", [None, *GATED])
def test_rendered_code_is_valid_python(tmp_path, recipe):
    src = _render(tmp_path, feature_recipes=[recipe] if recipe else None)
    ast.parse(src)


# --- the leakage discipline -------------------------------------------------


@pytest.mark.parametrize("recipe", GATED)
def test_gates_never_derive_from_the_target(tmp_path, recipe):
    """The predict-forward trap.

    Every gate iterates `_driver_columns`, which drops TARGET_COLUMN. If a gate
    were ever written to loop over `frame.columns` directly it would derive
    from the target, and this asserts against that at the source level.
    """
    src = _render(tmp_path, feature_recipes=[recipe])
    body = src.split("def _add_partition_features", 1)[1]
    gate_lines = [
        line
        for line in body.splitlines()
        if "__lag" in line or "__roll" in line or "__part_" in line
    ]
    assert gate_lines, f"{recipe} produced no feature lines to check"
    for line in gate_lines:
        assert "TARGET_COLUMN" not in line, f"gate derives from the target: {line.strip()}"


def test_driver_columns_excludes_target_and_excluded_features(tmp_path):
    """`_driver_columns` is where F7 is actually enforced, so its drop set is
    asserted directly rather than trusted."""
    src = _render(tmp_path, feature_recipes=GATED)
    body = src.split("def _driver_columns", 1)[1].split("def ", 1)[0]
    assert "EXCLUDE_FEATURES" in body
    assert "TARGET_COLUMN" in body


def test_excluded_columns_are_named_in_the_rendered_constant(tmp_path):
    """The validation plan's exclusions must reach the file, or the drop set
    above has nothing to act on.

    Quote-agnostic on purpose: the constant is rendered by a Jinja filter, and
    whether it emits JSON or a Python literal is that filter's business. An
    earlier version asserted double quotes and broke when the templates moved
    to `| py` — testing the serialiser's punctuation rather than the property.
    """
    src = _render(tmp_path, feature_recipes=GATED)
    assert "ANCC" in src and "BUDA" in src


# --- resolver → renderer, end to end on rogii's shape -----------------------


@pytest.mark.parametrize("technique", GATED)
def test_resolution_applies_and_changes_bytes(tmp_path, technique):
    """Before the gates these resolved `not_applicable` with reason "no gate"."""
    res = resolve_technique(
        {"technique": technique}, {}, choice=_choice(), profile=ROGII_PROFILE
    )
    assert res.status == "applied", res.reason

    baseline = _render(tmp_path)
    treated = _render(tmp_path, feature_recipes=res.feature_recipes)
    assert _digest(baseline) != _digest(treated)


# --- it must run, not merely differ (§10.5) ---------------------------------


def _partitioned_data(root):
    """Minimal predict-forward layout: <entity>__main.csv per split."""
    import pandas as pd

    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    def frame(seed, n, labelled):
        data = {
            "MD": [float(seed + i) for i in range(n)],
            "GR": [float((i * 3) % 11) for i in range(n)],
            "ANCC": [float(i) for i in range(n)],  # excluded by the validation plan
        }
        if labelled:
            data["TVT"] = [float(i) * 1.5 + seed for i in range(n)]
        return pd.DataFrame(data)

    for i in range(12):
        frame(i * 100, 40, True).to_csv(root / "train" / f"e{i:03d}__main.csv", index=False)
    for i in range(2):
        frame(i * 100, 40, False).to_csv(root / "test" / f"t{i:03d}__main.csv", index=False)
    pd.DataFrame({"row_id": ["t000_1"], "TVT": [0.0]}).to_csv(
        root / "sample_submission.csv", index=False
    )


@pytest.mark.parametrize("recipe", GATED)
def test_generated_code_runs_and_the_features_reach_the_model(tmp_path, recipe):
    """The check that source inspection cannot make.

    Asserting the recipe's columns appear in `metrics.json["features"]` is what
    proves the technique reached the *model*, not merely the rendered file — a
    gate that renders but produces columns the feature selector drops would
    otherwise look like success.
    """
    import json
    import subprocess
    import sys

    run_dir = tmp_path / "run"
    (run_dir / "pipeline").mkdir(parents=True, exist_ok=True)
    _partitioned_data(run_dir / "data" / "raw")

    choice = _choice()
    template = get_template(choice.problem_type, template_name=TEMPLATE)
    CodeRenderer(TrainingConfig()).render(
        template, choice, run_dir, feature_recipes=[recipe]
    )

    proc = subprocess.run(
        [sys.executable, str(run_dir / "pipeline" / "train.py")],
        capture_output=True,
        text=True,
        timeout=600,
        env={**__import__("os").environ, "LABPILOT_SMOKE": "1"},
    )
    assert proc.returncode == 0, f"rendered {recipe} failed to run:\n{proc.stderr[-2500:]}"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    suffix = {
        "lag_features": "__lag",
        "rolling_features": "__roll",
        "aggregation_features": "__part_",
    }[recipe]
    derived = [f for f in metrics["features"] if suffix in f]
    assert derived, f"{recipe} produced no features the model used: {metrics['features']}"

    # Leakage, asserted on what the model actually consumed rather than on source.
    assert not [f for f in derived if f.startswith("TVT")], f"target-derived: {derived}"
    assert not [f for f in derived if f.startswith("ANCC")], f"excluded-derived: {derived}"
