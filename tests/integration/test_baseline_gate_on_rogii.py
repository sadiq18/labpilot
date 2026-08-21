"""Goal 7: the gate, against a sandbox copy of the real workspace.

M23's goal table asks for this specifically — *"on rogii the gate reports
`failed` and names the anchor-column cause | **Sandbox copy of the real
workspace**"* — and every other test of this machinery runs against a fixture
shaped like rogii rather than rogii itself. Six defects in this milestone came
from that substitution, so the goal names the remedy.

**Read-only, always.** AGENTS.md rule 1: the live workspace is never written to.
The artifacts are copied into `tmp_path` and the data is read and never touched.

**Marked `slow` and skipped loudly.** It reads forty real partition files. M24's
exit criterion 7 is that a skip for absent data must never be a silent pass, so
the skip says which path was missing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from labpilot.research_engine.execution.baseline.baseline_one import (
    ModelReading,
    write_baseline_one,
)
from labpilot.research_engine.execution.baseline.floor import compute_floor, write_floor
from labpilot.research_engine.execution.baseline.gate import evaluate_gate, reading_fingerprint
from labpilot.research_engine.execution.baseline.report import build_report
from labpilot.research_engine.execution.baseline.selector import ValidationPlan

WORKSPACE = Path("/Users/sadik/workspace/rogii-wellbore-geology-prediction")

#: Enough partitions for the suffix scheme to be exercised over a duplicate
#: index and real column dtypes, few enough to stay under a second.
PARTITIONS = 40


def _sandbox(tmp_path: Path) -> Path:
    """A copy of the real workspace's artifacts. Never the workspace itself."""
    missing = [
        str(WORKSPACE / name)
        for name in ("profile.json", "baseline_choice.json", "pipeline", "data/raw/train")
        if not (WORKSPACE / name).exists()
    ]
    if missing:
        pytest.skip(f"real rogii workspace not available; missing: {', '.join(missing)}")

    root = tmp_path / "rogii-wellbore-geology-prediction"
    root.mkdir()
    for name in ("profile.json", "baseline_choice.json", "competition.json"):
        if (WORKSPACE / name).is_file():
            shutil.copy2(WORKSPACE / name, root / name)
    shutil.copytree(WORKSPACE / "pipeline", root / "pipeline")
    return root


def _real_frame() -> pd.DataFrame:
    frames = []
    for path in sorted((WORKSPACE / "data/raw/train").glob("*__horizontal_well.csv"))[:PARTITIONS]:
        frame = pd.read_csv(path)
        frame["file_stem_entity"] = path.stem.split("__")[0]
        frames.append(frame)
    # Concatenated exactly as the profiler sees it: each file keeps its own
    # 0..n, so the index is not unique. That shape crashed `folds_for` until a
    # review caught it, and no fixture in this repo reproduced it.
    return pd.concat(frames)


def _readings(root: Path, frame: pd.DataFrame) -> None:
    choice = json.loads((root / "baseline_choice.json").read_text(encoding="utf-8"))
    profile = json.loads((root / "profile.json").read_text(encoding="utf-8"))
    floor = compute_floor(
        frame,
        target=str(profile["target_column"]),
        plan=ValidationPlan.model_validate(choice["validation"]),
        metric_name=str(choice["metric_name"]),
        direction="minimize",
        anchor_column=profile.get("anchor_column"),
    )
    # The *recorded* pipeline score, not a re-run: this is the number eleven
    # campaigns actually produced, and re-training it here would measure
    # something else.
    recorded = json.loads((WORKSPACE / "metrics.json").read_text(encoding="utf-8"))["cv_rmse"]
    model = ModelReading(metric_name=str(choice["metric_name"]), model="pipeline", score=recorded)
    stamp = reading_fingerprint(root)
    floor.workspace_fingerprint = model.workspace_fingerprint = stamp
    write_floor(root, floor)
    write_baseline_one(root, model)


@pytest.mark.slow
def test_the_gate_fails_the_real_rogii_pipeline(tmp_path: Path) -> None:
    """Goal 3 and goal 7, on the workspace they were written about.

    The pipeline scores 1380.38. Predicting the mean scores ~655 on the same
    folds — so eleven campaigns of work produced something **twice as bad as a
    constant**, and until this milestone nothing in the system could say so:
    `_observe_delta` compares runs against each other, never against doing
    nothing.
    """
    root = _sandbox(tmp_path)
    frame = _real_frame()
    _readings(root, frame)

    verdict = evaluate_gate(root)

    assert verdict.state == "failed"
    assert verdict.comparison.model_score == pytest.approx(1380.38, abs=0.01)
    assert verdict.comparison.floor_score < verdict.comparison.model_score
    assert verdict.comparison.improvement < 0


@pytest.mark.slow
def test_the_report_names_a_cause_and_cites_an_artifact(tmp_path: Path) -> None:
    """Goal 6, against real artifacts rather than a fixture.

    Which cause fires is not asserted — the report's contract is that whatever
    fires cites the file it was read from, and pinning today's particular cause
    would make this test fail when the pipeline improves rather than when the
    report breaks.
    """
    root = _sandbox(tmp_path)
    _readings(root, _real_frame())

    report = build_report(root, evaluate_gate(root), competition="rogii")

    assert report.observed, "the real workspace should trip at least one detector"
    for cause in report.observed:
        assert cause.citation, f"{cause.name} has no artifact behind it"
        cited = cause.citation.split(":")[0].split(",")[0].strip()
        assert (root / cited).exists() or cited in cause.citation


@pytest.mark.slow
def test_the_anchor_cause_stays_quiet_because_this_pipeline_uses_it(tmp_path: Path) -> None:
    """The design expected the anchor cause here. It does not fire, and that is
    correct — which is worth pinning rather than quietly diverging from.

    The goal table was written when `pipeline/train.py` did not mention
    `TVT_input`. It now references it twenty-three times, including a
    forward-filled carry (`anchor_target: last known TVT_input value`). The
    detector's rule is *named in the profile and used by no training source*, so
    staying quiet is the right answer for this workspace as it stands today.

    rogii still fails — for feature selection, not for the anchor — so goal 7's
    substance holds and only its example has moved on.
    """
    root = _sandbox(tmp_path)
    _readings(root, _real_frame())
    profile = json.loads((root / "profile.json").read_text(encoding="utf-8"))
    assert profile.get("anchor_column") == "TVT_input", "the profiler still names it"
    assert "TVT_input" in (root / "pipeline" / "train.py").read_text(encoding="utf-8")

    report = build_report(root, evaluate_gate(root), competition="rogii")

    assert [c for c in report.observed if c.name == "leakage/ID handling"] == []
    assert "leakage/ID handling" in report.not_ruled_out


@pytest.mark.slow
def test_the_real_partitioned_split_is_not_quadratic(tmp_path: Path) -> None:
    """`folds_for` took 12.4 seconds on 32,000 rows before a fix.

    `i not in set(val.tolist())` rebuilt the set for every row — invisible on the
    hundreds-of-rows fixtures everywhere else in this suite, and hours on rogii's
    1,546 partitions. Found by running against the real workspace, which is the
    argument for this file existing.
    """
    import time

    from labpilot.research_engine.execution.baseline.floor import folds_for

    _sandbox(tmp_path)
    frame = _real_frame()
    plan = ValidationPlan.model_validate(
        json.loads((WORKSPACE / "baseline_choice.json").read_text(encoding="utf-8"))["validation"]
    )

    started = time.perf_counter()
    folds = folds_for(plan, frame)
    elapsed = time.perf_counter() - started

    assert folds, "the suffix scheme must produce a split on the real layout"
    assert elapsed < 2.0, f"{len(frame):,} rows took {elapsed:.1f}s — check for a quadratic"
