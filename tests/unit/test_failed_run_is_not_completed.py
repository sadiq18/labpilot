"""A failed experiment must not publish a completed one's numbers.

Measured on rogii 2026-08-08. E-147 died on `import catboost`. `metrics` in the
event payload is whatever sits on disk, which after a crash is the *previous*
run's file — so `ExperimentCompleted` carried `rmse 13.957107`, E-003's figure
from six days earlier.

That note is read by `build_observe_bundle` straight into the Conductor's
observation, so the next decision saw a completed experiment with a real score
and re-ran the same broken file. Sixteen dispatches, zero `write_code`. This is
the artifact that fuelled the loop, and unlike a stale file a human might read,
it steers what the system does next.
"""

from __future__ import annotations

import json

from labpilot.research_engine.agents.events import (
    EVIDENCE_UPDATED,
    EXPERIMENT_COMPLETED,
)
from labpilot.research_engine.agents.subscribers import (
    install_evidence_refresh_subscriber,
)


class _Bus:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.published: list[tuple[str, dict]] = []

    def subscribe(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)

    def publish(self, event, payload):
        self.published.append((event, payload))
        for handler in self.handlers.get(event, []):
            handler(event, payload)


_STALE = {"cv_rmse": 194.80084243002463, "rmse": 13.957107237175784}


def _payload(tmp_path, **over):
    base = {
        "experiment_id": "exp_E-147",
        "execution_id": "E-147",
        "plan_id": "P-019",
        "competition": "rogii-wellbore-geology-prediction",
        "workspace_root": str(tmp_path),
        "metrics": _STALE,
    }
    base.update(over)
    return base


def _note(tmp_path):
    return tmp_path / "artifacts" / "evidence_refresh_rogii-wellbore-geology-prediction.json"


def test_a_failed_run_writes_no_evidence_note(tmp_path):
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    bus = _Bus()
    install_evidence_refresh_subscriber(bus)

    bus.publish(EXPERIMENT_COMPLETED, _payload(tmp_path, status="failed"))

    assert not _note(tmp_path).is_file()


def test_an_errored_run_writes_no_evidence_note(tmp_path):
    """Status may be absent while `error` carries the traceback."""
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    bus = _Bus()
    install_evidence_refresh_subscriber(bus)

    bus.publish(
        EXPERIMENT_COMPLETED,
        _payload(tmp_path, error="ModuleNotFoundError: No module named 'catboost'"),
    )

    assert not _note(tmp_path).is_file()


def test_a_successful_run_still_writes_its_note(tmp_path):
    """The carve-out must not cost the behaviour it guards."""
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    bus = _Bus()
    install_evidence_refresh_subscriber(bus)

    bus.publish(EXPERIMENT_COMPLETED, _payload(tmp_path, status="succeeded"))

    note = json.loads(_note(tmp_path).read_text(encoding="utf-8"))
    assert note["execution_id"] == "E-147"
    assert note["metrics"]["rmse"] == _STALE["rmse"]
    assert any(e == EVIDENCE_UPDATED for e, _ in bus.published)
