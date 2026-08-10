"""M20, one layer up: an error must not be indistinguishable from an answer.

The milestone found gates that could not fail. The same shape sits above them,
in the reads the conductor *decides* on. Seven of them wrapped a store access in
`except Exception:` and returned the negative — `None`, `False`, `0`,
`(0, 0.0)`, `""` — under a comment saying *"absent store means nothing yet"*.
True of the case each author had in mind. False of every other one: a locked
database, a schema the code no longer matches, a permissions problem all
returned exactly what an empty workspace returns, and said nothing at all.

The fix is the same everywhere and is not a wider `except`. **Absence is asked
first**, so the negative answer is reached without an exception, and the handler
is left holding only genuine faults — which are then logged with their
traceback, because the value returned after one is a guess and the log is the
only place that admits it.

These tests hold both halves: absence still answers cleanly, and a fault is
still visible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from labpilot.research_engine.intelligence.paths import ResearchPaths, store_is_absent


class _Workspace:
    def __init__(self, knowledge_dir: Path, competition: str = "demo") -> None:
        self.knowledge_dir = knowledge_dir
        self.competition = competition
        self.effective_runs_dir = knowledge_dir / "runs"


def _with_a_store(tmp_path: Path) -> Path:
    """A workspace that is present but broken, so the absence check says "there
    is something here" and the code under test has to reach it — and fail.

    Both stores, because they are different stores: plans live in SQLite and
    hypotheses in a directory of JSON. Asking `knowledge.db` about hypotheses
    was the first version of the absence check and would have answered *"no
    hypotheses"* for a workspace full of them — the same mistake in the other
    direction.
    """
    from labpilot.workspace import competition_data_root

    knowledge = tmp_path / "knowledge"
    paths = ResearchPaths(knowledge, "demo").ensure()
    paths.db_path.write_text("not a database at all", encoding="utf-8")
    hypotheses = competition_data_root(knowledge, "demo") / "hypotheses"
    hypotheses.mkdir(parents=True, exist_ok=True)
    return knowledge


@pytest.fixture
def broken_hypothesis_store(monkeypatch):
    """`HypothesisStore` is file-backed, so a corrupt SQLite file does not
    disturb it. Patch the read itself — the handler under test is the one that
    decides what a failed read means, not the storage layer's."""
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    def _raise(self, *args, **kwargs):
        raise OSError("hypotheses directory is unreadable")

    monkeypatch.setattr(HypothesisStore, "list", _raise)


# -- the distinction itself ----------------------------------------------------


def test_an_empty_workspace_is_absent(tmp_path):
    assert store_is_absent(tmp_path / "never-written", "demo") is True


def test_a_workspace_with_a_store_is_not_absent(tmp_path):
    assert store_is_absent(_with_a_store(tmp_path), "demo") is False


def test_no_knowledge_dir_at_all_is_absent():
    """`Path(None)` raises, and a `TypeError` escaping a question this calm
    would crash the conductor."""
    assert store_is_absent(None, "demo") is True


# -- absence still answers cleanly ---------------------------------------------


def test_absence_needs_no_exception(tmp_path, caplog):
    """The point of asking first: an empty workspace must not travel through an
    error path, or the log fills with tracebacks that mean nothing and the real
    ones stop being read."""
    from labpilot.research_engine.conductor import loop, policy

    workspace = _Workspace(tmp_path / "never-written")
    with caplog.at_level(logging.ERROR):
        assert loop._latest_plan_id(workspace) is None
        assert loop._next_hypothesis_id(workspace) is None
        assert loop._baseline_plan_exists(workspace) is False
        assert policy.has_runnable_plan(workspace) is False
        assert policy.untested_hypothesis_count(workspace) == 0

    assert caplog.records == []


# -- a fault is loud, and still answers --------------------------------------


@pytest.mark.parametrize(
    ("name", "call", "negative"),
    [
        ("_latest_plan_id", "loop._latest_plan_id", None),
        ("has_runnable_plan", "policy.has_runnable_plan", False),
    ],
)
def test_a_broken_store_is_logged_with_its_traceback(tmp_path, caplog, name, call, negative):
    """The value is unchanged — the conductor keeps running — but the failure is
    no longer invisible. Before this, a corrupted store and an empty one were the
    same event, and neither left a trace."""
    from labpilot.research_engine.conductor import loop, policy  # noqa: F401

    workspace = _Workspace(_with_a_store(tmp_path))
    function = eval(call)  # noqa: S307 - parametrised over module attributes

    with caplog.at_level(logging.ERROR):
        assert function(workspace) == negative

    assert caplog.records, f"{name} swallowed a broken store silently"
    assert any(record.exc_info for record in caplog.records), (
        f"{name} logged without the traceback, which is the part that says what broke"
    )


@pytest.mark.parametrize(
    ("name", "call", "negative"),
    [
        ("_next_hypothesis_id", "loop._next_hypothesis_id", None),
        ("untested_hypothesis_count", "policy.untested_hypothesis_count", 0),
    ],
)
def test_a_broken_hypothesis_store_is_logged_too(
    tmp_path, caplog, broken_hypothesis_store, name, call, negative
):
    """Same rule, the other storage layer."""
    from labpilot.research_engine.conductor import loop, policy  # noqa: F401

    workspace = _Workspace(_with_a_store(tmp_path))
    function = eval(call)  # noqa: S307 - parametrised over module attributes

    with caplog.at_level(logging.ERROR):
        assert function(workspace) == negative

    assert any(record.exc_info for record in caplog.records), f"{name} swallowed it silently"


def test_a_read_whose_negative_destroys_work_raises_instead(tmp_path):
    """`_baseline_plan_exists` is the exception to the rule above, and PR #120 is
    why. The other reads degrade to a negative because being wrong costs a step;
    this one's negative means *compile a baseline* over whatever is already
    there, so a fault that reads as "no baseline" destroys work rather than
    delaying it."""
    from labpilot.research_engine.conductor import loop

    workspace = _Workspace(_with_a_store(tmp_path))

    with pytest.raises(Exception, match="not a database"):
        loop._baseline_plan_exists(workspace)


def test_viability_reports_a_fault_rather_than_an_empty_backlog(
    tmp_path, caplog, broken_hypothesis_store
):
    """This count opens M21's gathering gate. A fault reporting zero is the gate
    stuck shut — the failure that module's own docstring exists to prevent,
    arriving through its error path."""
    from labpilot.research_engine.intelligence.hypothesis.viability import (
        viable_hypothesis_count,
    )

    knowledge = _with_a_store(tmp_path)

    with caplog.at_level(logging.ERROR):
        assert viable_hypothesis_count(knowledge, "demo") == 0

    assert any(record.exc_info for record in caplog.records)


def test_unmeasured_is_not_the_same_as_measured_zero(tmp_path, caplog):
    """`(0, 0.0)` does not mean "unknown", it means *measured, and it was zero* —
    a claim about evidence, made because the evidence could not be read."""
    from labpilot.research_engine.evidence.store import EvidenceCardStore
    from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter

    knowledge = tmp_path / "knowledge"
    store = EvidenceCardStore(knowledge, "demo")

    class _Broken:
        def list(self):
            raise OSError("evidence directory is unreadable")

    promoter = ClaimPromoter.__new__(ClaimPromoter)
    promoter._evidence = _Broken()

    with caplog.at_level(logging.ERROR):
        assert promoter.measured_effect("swa") == (0, 0.0)

    assert any(record.exc_info for record in caplog.records)
    store.dir.exists()


def test_a_corrupt_evidence_card_does_not_vanish_quietly(tmp_path, caplog):
    """A card that will not parse is not a card that does not exist. Returning
    `None` for both means a corrupted verdict reads as *no verdict*, and the
    promoter, the belief updater and the planner all act on that difference."""
    from labpilot.research_engine.evidence.store import EvidenceCardStore

    store = EvidenceCardStore(tmp_path / "knowledge", "demo")
    (store.dir / "EV-broken.json").write_text("{ not json", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        assert store.get("EV-broken") is None

    assert any(record.exc_info for record in caplog.records)


def test_an_unreadable_parent_does_not_become_a_baseline(tmp_path):
    """The most consequential of the seven. `prior_train == ""` is how the
    capability decides a run is a baseline: `_propose_delta` declines without it
    and the whole-file agent rewrites `train.py` from scratch. So a permissions
    problem turned an experiment into a fresh start, on a card that said the step
    passed — M19's premise lost to an `except` clause.
    """
    from unittest import mock

    from helpers.capability_context import capability_context

    from labpilot.research_engine.execution.capabilities.code_engineering import (
        CodeEngineeringCapability,
    )
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    context = capability_context(tmp_path, task_type=TaskType.WRITE_CODE)
    train = context.workspace_root / "pipeline" / "train.py"
    train.write_text("def main():\n    return 1\n", encoding="utf-8")

    real_read = Path.read_text

    def _unreadable(self, *args, **kwargs):
        if self.name == "train.py":
            raise OSError("permission denied")
        return real_read(self, *args, **kwargs)

    with mock.patch.object(Path, "read_text", _unreadable):
        result = CodeEngineeringCapability().execute(context)

    assert result.passed is False
    assert "could not be read" in (result.error or "")


# --- PR #120 review ----------------------------------------------------------


def test_an_unmigrated_client_workspace_is_not_absent(tmp_path):
    """Reported on PR #120. `ResearchPaths` reads the *migrated* location and
    `ensure()` is what moves things there — but a read must not migrate, so a
    client workspace still holding `knowledge/<slug>/research/knowledge.db`
    reported absent and the conductor rebuilt a baseline over live data."""
    from labpilot.research_engine.intelligence.paths import (
        hypotheses_are_absent,
        store_is_absent,
    )

    knowledge = tmp_path / "knowledge"
    (knowledge / "demo" / "research").mkdir(parents=True)
    (knowledge / "demo" / "research" / "knowledge.db").write_text("x", encoding="utf-8")
    (knowledge / "demo" / "hypotheses").mkdir()

    assert store_is_absent(knowledge, "demo") is False
    assert hypotheses_are_absent(knowledge, "demo") is False


def test_a_genuinely_empty_workspace_is_still_absent(tmp_path):
    """The carve-out must not cost the behaviour it guards."""
    from labpilot.research_engine.intelligence.paths import (
        hypotheses_are_absent,
        store_is_absent,
    )

    assert store_is_absent(tmp_path, "demo") is True
    assert hypotheses_are_absent(tmp_path, "demo") is True


def test_a_fault_reading_plans_stops_rather_than_rebuilding(tmp_path):
    """Reported on PR #120. The other reads here degrade to a negative because
    being wrong costs a step. This one's negative means *compile a baseline*,
    over whatever is already there — so a fault that reads as "no baseline"
    destroys work rather than delaying it, and it raises instead."""
    from unittest import mock

    from labpilot.accessor.sqlite import SqliteClient
    from labpilot.research_engine.artifacts.plan import PlanArtifacts
    from labpilot.research_engine.conductor import loop

    knowledge = tmp_path / "knowledge"
    paths = ResearchPaths(knowledge, "demo").ensure()
    # A *real* database, so `store_is_absent` says the workspace is live and the
    # only fault in play is the mocked one. `ensure()` makes directories; the
    # file appears when a client connects.
    SqliteClient(paths.db_path, allow_cross_thread=True).close()
    workspace = _Workspace(knowledge)

    with mock.patch.object(PlanArtifacts, "list", side_effect=OSError("plan store unreadable")):
        with pytest.raises(OSError, match="unreadable"):
            loop._baseline_plan_exists(workspace)


def test_the_conductor_path_forwards_kaggle_credentials():
    """Reported on PR #120: `run_plan` never plumbed `config.kaggle`, so
    `prepare_workspace` reached its download step with no credentials on every
    conductor-driven campaign and skipped it — silently, while a skip and a
    success were the same answer."""
    import inspect

    from labpilot.research_engine.tools.handlers import run

    assert '"kaggle": _kaggle_config(workspace)' in inspect.getsource(run.run_plan)


def test_the_card_agrees_with_its_own_verdict(tmp_path):
    """Reported on PR #120. A failing PREPARE_WORKSPACE card read *"workspace
    prepared; download skipped"* beside `passed=False` — the summary describing
    the old, silent behaviour while the verdict described the new one. A card
    that argues with itself is worse than either half alone, because a reader
    believes the sentence and the pipeline believes the boolean."""
    from helpers.capability_context import capability_context

    from labpilot.research_engine.execution.capabilities.workspace import (
        WorkspaceCapability,
    )
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    context = capability_context(
        tmp_path,
        task_type=TaskType.PREPARE_WORKSPACE,
        constraints={"skip_download": False, "dry_run": False},
    )

    result = WorkspaceCapability().execute(context)

    assert result.passed is False
    assert "not prepared" in result.summary
    assert "prepared;" not in result.summary


def test_a_requested_skip_still_reads_as_prepared(tmp_path):
    """The carve-out: asking not to download is not a failure, and the card
    should keep saying so."""
    from helpers.capability_context import capability_context

    from labpilot.research_engine.execution.capabilities.workspace import (
        WorkspaceCapability,
    )
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    context = capability_context(
        tmp_path,
        task_type=TaskType.PREPARE_WORKSPACE,
        constraints={"skip_download": True, "dry_run": False},
    )

    result = WorkspaceCapability().execute(context)

    assert result.passed is True
    assert "download skipped" in result.summary


def test_the_agent_path_inherits_the_kaggle_credentials():
    """`agents/experiment.py` passes its own constraints to `run_plan`, which
    merges them *after* its own — so a caller that says nothing about Kaggle
    inherits what `run_plan` read from the workspace. Raised on PR #120 as a
    third unplumbed call site; it is covered by the second."""
    import inspect

    from labpilot.research_engine.tools.handlers import run

    source = inspect.getsource(run.run_plan)
    kaggle_at = source.index('"kaggle": _kaggle_config(workspace)')
    spread_at = source.index("**(constraints or {})")

    assert kaggle_at < spread_at, "a caller's constraints must be able to override, not precede"
