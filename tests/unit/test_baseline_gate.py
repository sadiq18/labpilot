"""M23 steps 5 and 6: the nine states, and a report that cites its evidence.

Observe-only by design. Enforcement is step 8, and it is last because the plan's
own trap records that `_observe_delta` was *"calibrated against hand-written
samples, and that is precisely how the two bugs got in"* — one campaign's worth
of recorded verdicts is what turns "the gate is right" from an argument into a
false-positive rate.

Checks 4-8 from the plan live here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labpilot.research_engine.execution.baseline.baseline_one import (
    fit_baseline_one,
    write_baseline_one,
)
from labpilot.research_engine.execution.baseline.floor import (
    FloorReading,
    compute_floor,
    write_floor,
)
from labpilot.research_engine.execution.baseline.gate import (
    GATE_STATES,
    Waiver,
    blocks_research,
    evaluate_gate,
    reading_fingerprint,
    write_waiver,
)
from labpilot.research_engine.execution.baseline.report import CAUSES, build_report
from labpilot.research_engine.execution.baseline.selector import ValidationPlan

PLAN = ValidationPlan(scheme="kfold", n_splits=4)


def _workspace(tmp_path: Path, **choice) -> Path:
    body = {
        "problem_type": "tabular_regression",
        "template_name": "t",
        "rationale": "",
        "metric_name": "rmse",
        "target_column": "y",
        "train_file": "train.csv",
        "validation": {"scheme": "kfold", "n_splits": 4},
    }
    body.update(choice)
    (tmp_path / "baseline_choice.json").write_text(json.dumps(body), encoding="utf-8")
    (tmp_path / "profile.json").write_text(
        json.dumps({"competition": "demo", "schema_version": 4, "target_column": "y"}),
        encoding="utf-8",
    )
    return tmp_path


def _readings(tmp_path: Path, *, model_better: bool) -> None:
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    y = 3 * x1 + rng.normal(0, 0.3, n) if model_better else rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "y": y})
    floor = compute_floor(frame, target="y", plan=PLAN, metric_name="rmse", direction="minimize")
    model = fit_baseline_one(
        frame,
        target="y",
        plan=PLAN,
        metric_name="rmse",
        target_type="continuous",
        feature_columns=["x1", "x2"],
    )
    stamp = reading_fingerprint(tmp_path)
    floor.workspace_fingerprint = stamp
    model.workspace_fingerprint = stamp
    write_floor(tmp_path, floor)
    write_baseline_one(tmp_path, model)


# --- check 5: every target_type maps to a state, never silently to passed -----


@pytest.mark.parametrize(
    "target_type",
    ["binary", "multiclass", "multilabel", "continuous", "count", "ordinal", "none", "unknown"],
)
def test_no_target_type_reaches_passed_without_a_reading(tmp_path: Path, target_type: str) -> None:
    """Goal 5, parametrized. The undefined shapes must land on a state that is
    not `passed`, and the defined ones must not reach `passed` either without
    something having actually been measured."""
    _workspace(tmp_path)
    (tmp_path / "profile.json").write_text(
        json.dumps(
            {
                "competition": "d",
                "schema_version": 4,
                "target_column": "y",
                "columns": [{"name": "y", "dtype": "int64", "unique_count": 2, "is_numeric": True}],
            }
        ),
        encoding="utf-8",
    )

    verdict = evaluate_gate(tmp_path)

    assert verdict.state in GATE_STATES
    assert verdict.state != "passed"
    assert verdict.reason, "every state explains itself"


def test_an_undefined_floor_is_not_a_pass(tmp_path: Path) -> None:
    """`floor_undefined` and `passed` are different answers to different
    questions, and collapsing them is how eight gates reported `pass` on things
    that could not run."""
    _workspace(tmp_path)
    write_floor(
        tmp_path,
        FloorReading(
            metric_name="spearman",
            undefined_reason="no floor strategy is defined for metric 'spearman'",
            workspace_fingerprint=reading_fingerprint(tmp_path),
        ),
    )

    verdict = evaluate_gate(tmp_path)

    assert verdict.state == "floor_undefined"
    assert verdict.blocks_research


# --- check 7: an uncertain schema is not a failed baseline --------------------


def test_an_open_schema_question_blocks_uncertain_not_failed(tmp_path: Path) -> None:
    """Goal 8, which M22 made reachable.

    `blocked_uncertain` was in the plan's nine states with no defined trigger.
    A floor computed against a guessed target is worse than no floor, and the
    operator's action here is to answer the question — not to debug a baseline.
    """
    from labpilot.accessor.profiler.evidence import Inference, Signal

    _workspace(tmp_path)
    # A real uncertain inference, built through the shipped model rather than
    # hand-written: `Inference` re-checks that `confidence == combine(signals)`
    # on load, so a fabricated one would not survive validation — and a fixture
    # the product cannot produce proves nothing.
    weak = Inference.of([Signal(id="positional_template_overlap")])
    assert weak.band == "uncertain"
    (tmp_path / "profile.json").write_text(
        json.dumps(
            {
                "competition": "d",
                "schema_version": 4,
                "target_column": "y",
                "inferences": {"target_column": json.loads(weak.model_dump_json())},
            }
        ),
        encoding="utf-8",
    )
    _readings(tmp_path, model_better=False)

    verdict = evaluate_gate(tmp_path)

    assert verdict.state == "blocked_uncertain"
    assert "guessed target" in verdict.reason
    assert verdict.state != "failed", "the operator answers; they do not debug a baseline"


# --- the ordinary verdicts ----------------------------------------------------


def test_a_model_that_beats_the_floor_passes(tmp_path: Path) -> None:
    _workspace(tmp_path)
    _readings(tmp_path, model_better=True)

    verdict = evaluate_gate(tmp_path)

    assert verdict.state == "passed"
    assert not verdict.blocks_research
    assert verdict.comparison.beats_floor


def test_a_model_that_loses_to_a_constant_fails(tmp_path: Path) -> None:
    """The first failure in this system for being *worse* rather than for
    crashing."""
    _workspace(tmp_path)
    _readings(tmp_path, model_better=False)

    verdict = evaluate_gate(tmp_path)

    assert verdict.state == "failed"
    assert verdict.blocks_research


def test_no_floor_on_disk_is_missing_not_passed(tmp_path: Path) -> None:
    _workspace(tmp_path)

    assert evaluate_gate(tmp_path).state == "floor_missing"


def test_a_floor_without_a_model_is_awaiting_ml(tmp_path: Path) -> None:
    """The floor stands; nothing has been compared to it. Not `passed`."""
    _workspace(tmp_path)
    write_floor(
        tmp_path,
        FloorReading(
            metric_name="rmse",
            score=1.0,
            best_strategy="mean",
            workspace_fingerprint=reading_fingerprint(tmp_path),
        ),
    )

    verdict = evaluate_gate(tmp_path)

    assert verdict.state == "awaiting_ml"
    assert verdict.blocks_research


# --- staleness ----------------------------------------------------------------


def test_a_changed_answer_makes_the_readings_stale(tmp_path: Path) -> None:
    """ "An operator answering 'the label is Depth' invalidates every reading
    that described Zone_Depth."

    Without this the gate keeps reporting `passed` over a measurement of the
    wrong column, which is the most convincing wrong verdict it could give.
    """
    _workspace(tmp_path)
    _readings(tmp_path, model_better=True)
    assert evaluate_gate(tmp_path).state == "passed"

    profile = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    profile["answers_fingerprint"] = "someone-answered-since"
    (tmp_path / "profile.json").write_text(json.dumps(profile), encoding="utf-8")

    verdict = evaluate_gate(tmp_path)

    assert verdict.state == "stale"
    assert "answers" in verdict.reason


def test_an_unstamped_reading_is_not_stale(tmp_path: Path) -> None:
    """Every reading written before the field existed has an empty stamp.

    Re-measuring every workspace on upgrade would train an operator to ignore
    the state, and `stale` matters as much as `failed` only while it is rare.
    """
    _workspace(tmp_path)
    write_floor(tmp_path, FloorReading(metric_name="rmse", score=1.0, best_strategy="mean"))

    assert evaluate_gate(tmp_path).state != "stale"


# --- the waiver ---------------------------------------------------------------


def test_a_waiver_is_recorded_and_specific(tmp_path: Path) -> None:
    """Durable and fingerprinted, because an env-var kill switch gets set once
    during a frustrating afternoon and never unset, and nothing records that it
    happened."""
    _workspace(tmp_path)
    _readings(tmp_path, model_better=False)
    assert evaluate_gate(tmp_path).state == "failed"

    write_waiver(
        tmp_path,
        Waiver(
            reason="known-bad baseline, shipping anyway",
            fingerprint=reading_fingerprint(tmp_path),
        ),
    )

    verdict = evaluate_gate(tmp_path)

    assert verdict.state == "waived"
    assert not verdict.blocks_research
    assert "shipping anyway" in verdict.reason


def test_a_waiver_granted_against_other_inputs_does_not_apply(tmp_path: Path) -> None:
    """A waiver that outlived its cause is the gate quietly switching itself off.

    Reported rather than ignored: an operator who granted one deserves to know
    it no longer applies, instead of believing they had already dealt with this.
    """
    _workspace(tmp_path)
    _readings(tmp_path, model_better=False)
    write_waiver(tmp_path, Waiver(reason="stale waiver", fingerprint="from-another-dataset"))

    verdict = evaluate_gate(tmp_path)

    assert verdict.state == "failed"
    assert "different fingerprint" in verdict.reason


# --- check 8: observe-only ------------------------------------------------------


def test_a_failing_verdict_withholds_nothing_while_observing(tmp_path: Path) -> None:
    """The rollout's first stage does what it claims.

    `blocks_research` and `withholds_anything` are two facts, not one: "this
    campaign has not shown a working baseline" and "the system is refusing it
    something" are different, and conflating them is how an observe-only rollout
    quietly becomes an enforcing one.
    """
    _workspace(tmp_path)
    _readings(tmp_path, model_better=False)

    observing = evaluate_gate(tmp_path)
    enforcing = evaluate_gate(tmp_path, enforced=True)

    assert observing.state == enforcing.state == "failed"
    assert observing.blocks_research and enforcing.blocks_research
    assert not observing.withholds_anything, "nothing is withheld in step 5"
    assert enforcing.withholds_anything


def test_a_tenth_state_would_block_by_default() -> None:
    """Written as the complement of passed/waived rather than as a list, so the
    direction of a future mistake is toward refusing rather than permitting."""
    assert not blocks_research("passed") and not blocks_research("waived")
    assert all(blocks_research(s) for s in GATE_STATES if s not in ("passed", "waived"))
    assert blocks_research("a_state_nobody_has_written_yet")  # type: ignore[arg-type]


# --- check 6: every cause cites an artifact ---------------------------------------


def test_every_observed_cause_carries_a_citation(tmp_path: Path) -> None:
    """Goal 6. A cause without a citation is an opinion, and `Cause` has no way
    to express one."""
    _workspace(tmp_path, metric_substituted_from="balanced_accuracy")
    _readings(tmp_path, model_better=False)

    report = build_report(tmp_path, evaluate_gate(tmp_path), competition="demo")

    assert report.observed, "the substituted metric should have fired"
    for cause in report.observed:
        assert cause.citation, f"{cause.name} has no artifact behind it"


def test_a_report_with_nothing_to_say_says_so(tmp_path: Path) -> None:
    """ "A list that prints identically on every failure is a list nobody reads."

    This repository has paid for that twice, in `check_confinement` and in
    `validation_region`. When nothing fires the report says the detectors ran,
    which is more useful than six bullets.
    """
    _workspace(tmp_path)
    _readings(tmp_path, model_better=False)

    report = build_report(tmp_path, evaluate_gate(tmp_path))

    assert report.observed == []
    assert "Every detector ran and none of them fired" in report.render()
    assert set(report.not_ruled_out) == set(CAUSES)


# --- check 4: rogii's shape ---------------------------------------------------------


def test_the_anchor_cause_fires_and_cites_the_profile(tmp_path: Path) -> None:
    """Check 4. rogii's whole story, and the detector cites `profile.json`.

    `TVT_input` equals the target wherever present and is absent exactly on the
    scored rows, so carrying it forward scores 15.1 against the pipeline's 1380.
    The profiler has named it since 2026-08-13 and nothing read it.
    """
    _workspace(tmp_path)
    (tmp_path / "profile.json").write_text(
        json.dumps(
            {
                "competition": "rogii",
                "schema_version": 4,
                "target_column": "TVT",
                "anchor_column": "TVT_input",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text(
        "import pandas as pd\ndf = pd.read_csv('train.csv')\nmodel.fit(df[['md']], df['TVT'])\n",
        encoding="utf-8",
    )
    _readings(tmp_path, model_better=False)

    report = build_report(tmp_path, evaluate_gate(tmp_path), competition="rogii")

    anchor = next(c for c in report.observed if c.name == "leakage/ID handling")
    assert "TVT_input" in anchor.detail
    assert "profile.json:anchor_column" in anchor.citation
    assert "pipeline/train.py" in anchor.citation
    assert "leakage/ID handling" not in report.not_ruled_out


def test_the_anchor_cause_stays_quiet_when_the_pipeline_uses_it(tmp_path: Path) -> None:
    """The detector must be able to *not* fire, or it is a constant."""
    _workspace(tmp_path)
    (tmp_path / "profile.json").write_text(
        json.dumps(
            {
                "competition": "r",
                "schema_version": 4,
                "target_column": "TVT",
                "anchor_column": "TVT_input",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text(
        "residual = df['TVT'] - df['TVT_input'].ffill()\n", encoding="utf-8"
    )
    _readings(tmp_path, model_better=False)

    report = build_report(tmp_path, evaluate_gate(tmp_path))

    assert [c for c in report.observed if c.name == "leakage/ID handling"] == []
    assert "leakage/ID handling" in report.not_ruled_out


def test_a_declared_scheme_no_function_performs_fires(tmp_path: Path) -> None:
    _workspace(tmp_path, validation={"scheme": "group_kfold", "group_key": "w", "n_splits": 4})
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text(
        "from sklearn.model_selection import KFold\nfor tr, va in KFold(5).split(X):\n    pass\n",
        encoding="utf-8",
    )
    _readings(tmp_path, model_better=False)

    report = build_report(tmp_path, evaluate_gate(tmp_path))

    mismatch = next(c for c in report.observed if c.name == "validation mismatch")
    assert "group_kfold" in mismatch.detail
    assert "baseline_choice.json" in mismatch.citation


def test_the_scheme_detector_recognises_the_camel_case_spelling(tmp_path: Path) -> None:
    """`group_kfold` and `GroupKFold` are the same word spelled twice.

    Reuses `delta/consistency.py`'s matcher rather than a second one — a fresh
    substring check here would disagree with the delta checks about whether the
    pipeline honours its own plan.
    """
    _workspace(tmp_path, validation={"scheme": "group_kfold", "group_key": "w", "n_splits": 4})
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text(
        "from sklearn.model_selection import GroupKFold\nGroupKFold(4).split(X, groups=g)\n",
        encoding="utf-8",
    )
    _readings(tmp_path, model_better=False)

    report = build_report(tmp_path, evaluate_gate(tmp_path))

    assert [c for c in report.observed if c.name == "validation mismatch"] == []
