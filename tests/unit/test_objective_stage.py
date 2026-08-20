"""M23 step 0: the objective is a stage, not an argument list.

Three properties, and the third is the one the milestone is about:

* it reads the *stage before it* — the target comes from `profile.json`, not
  from whatever a caller happened to pass;
* it is written down, so `contradiction` and `unresolved` outlive the console;
* it is reused only while its inputs still hold, and the inputs are stored, so
  "why was this re-resolved" has an answer on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from labpilot.research_engine.intelligence.competition.objective_stage import (
    OBJECTIVE_FILENAME,
    OBJECTIVE_SCHEMA_VERSION,
    StoredObjective,
    ensure_objective,
    load_objective,
    objective_state,
    read_inputs,
    resolve_workspace_objective,
    write_objective,
)


def _contract(root: Path, **metric: object) -> None:
    spec: dict[str, object] = {"slug": "demo", "problem_type": "tabular_regression"}
    spec["evaluation_metric"] = metric or None
    (root / "competition.json").write_text(json.dumps(spec), encoding="utf-8")


def _profile(root: Path, **fields: object) -> None:
    (root / "profile.json").write_text(
        json.dumps({"competition": "demo", "schema_version": 3, **fields}), encoding="utf-8"
    )


# --- it reads the stage before it -------------------------------------------


def test_the_target_comes_from_the_profile(tmp_path: Path) -> None:
    """M22 resolved the target with evidence; the objective reads that answer.

    `competition.json` carries no target at all, so an objective that did not
    read the profile would carry `target: None` while reading as though it were
    populated — which it did, for every campaign, until `_stated_target` was
    added as a patch on the CLI side.
    """
    _contract(tmp_path, name="rmse", direction="minimize")
    _profile(tmp_path, target_column="SalePrice")

    stored = resolve_workspace_objective(tmp_path, "demo")

    assert stored.inputs.target == "SalePrice"
    assert stored.spec.target == "SalePrice"
    assert stored.inputs.profile_read is True


def test_the_metric_is_read_from_the_profile_before_the_contract(tmp_path: Path) -> None:
    """The profile's `MetricRef` is what dataset understanding was handed.

    Re-parsing `competition.json` here would put a second reader of the same
    file on the other side of the seam, free to disagree with the first — and
    the disagreement would surface as the CLI and `objective.json` naming
    different metrics for one campaign.
    """
    _contract(tmp_path, name="rmse", direction="minimize")
    _profile(
        tmp_path,
        target_column="y",
        metric={"name": "mae", "key": "mae", "direction": "minimize"},
    )

    facts = read_inputs(tmp_path)

    assert facts.metric_raw == "mae"
    assert facts.metric_from == "profile"


def test_the_contract_is_the_fallback_for_a_profile_that_names_none(tmp_path: Path) -> None:
    """A pre-M22 profile, or one built by the inventory path, has no metric."""
    _contract(tmp_path, name="rmse", direction="minimize")
    _profile(tmp_path, target_column="y")

    facts = read_inputs(tmp_path)

    assert facts.metric_raw == "rmse"
    assert facts.metric_from == "competition.json"
    assert facts.declared_direction == "minimize"


def test_a_benchmark_states_its_own_objective(tmp_path: Path) -> None:
    """No `competition.json`, and still an objective.

    Exit criterion 3: the same command has to start a campaign in the other
    domain. A stage that could only read a Kaggle contract would re-introduce
    the refusal the preflight already removed.
    """
    (tmp_path / "harness.json").write_text(
        json.dumps({"metric": "pass_rate", "direction": "maximize"}), encoding="utf-8"
    )

    stored = resolve_workspace_objective(tmp_path, "bench")

    assert stored.inputs.metric_raw == "pass_rate"
    assert stored.inputs.externally_scored is True
    assert stored.spec.is_actionable, stored.spec.why_blocked()


def test_a_workspace_that_states_nothing_resolves_to_saying_so(tmp_path: Path) -> None:
    """Not an exception, and not `None` — a written objective naming the gap.

    `None` would push "we do not know" back into the caller's absence handling,
    where it is indistinguishable from "not asked yet".
    """
    stored = resolve_workspace_objective(tmp_path, "demo")

    assert stored.spec.unresolved == ["metric"]
    assert stored.spec.blocks_launch


# --- it is written down ------------------------------------------------------


def test_a_contradiction_survives_the_console(tmp_path: Path) -> None:
    """The whole point of step 0.

    A contract declaring `rmse` is `maximize` contradicts the probe, which
    *measures* the scorer that will be used. That verdict reached a console line
    and nothing else; now it is a field in a file, which is what a later stage
    can read and what an operator can find tomorrow.
    """
    _contract(tmp_path, name="rmse", direction="maximize")
    _profile(tmp_path, target_column="y")

    stored, how = ensure_objective(tmp_path, "demo")
    on_disk = json.loads((tmp_path / OBJECTIVE_FILENAME).read_text(encoding="utf-8"))

    assert how == "resolved"
    assert stored.spec.contradiction is not None
    assert on_disk["spec"]["contradiction"] == stored.spec.contradiction
    assert on_disk["spec"]["alternatives"] == ["maximize", "minimize"]


def test_the_file_is_stamped_by_its_writer(tmp_path: Path) -> None:
    """A default on the model would make every unstamped legacy file current.

    Same reason `write_profile` stamps rather than defaults, and the same defect
    avoided.
    """
    stored = StoredObjective()
    assert stored.schema_version == 0

    write_objective(tmp_path, stored)

    assert load_objective(tmp_path).schema_version == OBJECTIVE_SCHEMA_VERSION


# --- it is reused only while its inputs hold ---------------------------------


def test_an_unchanged_workspace_reuses_its_objective(tmp_path: Path) -> None:
    _contract(tmp_path, name="rmse", direction="minimize")
    _profile(tmp_path, target_column="y")

    _first, how_first = ensure_objective(tmp_path, "demo")
    _second, how_second = ensure_objective(tmp_path, "demo")

    assert how_first == "resolved"
    assert how_second == "reused"
    assert objective_state(tmp_path) == "current"


def test_a_changed_target_makes_the_objective_stale(tmp_path: Path) -> None:
    """The failure this prevents: an objective outliving the schema it read.

    A person answers `research schema answer target_column`, the profile is
    rebuilt naming a different column, and a cached objective would keep
    optimising toward the one that was rejected — silently, because nothing
    about the file would look wrong.
    """
    _contract(tmp_path, name="rmse", direction="minimize")
    _profile(tmp_path, target_column="y")
    ensure_objective(tmp_path, "demo")

    _profile(tmp_path, target_column="SalePrice")

    assert objective_state(tmp_path) == "stale"
    stored, how = ensure_objective(tmp_path, "demo")
    assert how == "resolved"
    assert stored.spec.target == "SalePrice"


def test_a_version_bump_re_resolves(tmp_path: Path) -> None:
    """An older resolver's answer is re-resolved, never migrated.

    Resolution is cheap; a migration would be code asserting what a previous
    version meant by a field it no longer writes.
    """
    _contract(tmp_path, name="rmse", direction="minimize")
    ensure_objective(tmp_path, "demo")
    path = tmp_path / OBJECTIVE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = OBJECTIVE_SCHEMA_VERSION - 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert objective_state(tmp_path) == "stale"


def test_unreadable_bytes_are_not_stale(tmp_path: Path) -> None:
    """`stale` and `unusable` are different situations.

    One is a file worth re-resolving from a workspace that has since changed;
    the other is bytes no reader can use. `_profile_state` keeps them apart for
    the same reason, and a truncated write is how both arise.
    """
    (tmp_path / OBJECTIVE_FILENAME).write_text('{"schema_version": 1, "spe', encoding="utf-8")

    assert objective_state(tmp_path) == "unusable"
    assert load_objective(tmp_path) is None
    # And it is replaced rather than served.
    _contract(tmp_path, name="rmse", direction="minimize")
    stored, how = ensure_objective(tmp_path, "demo")
    assert how == "resolved"
    assert stored.spec.metric_name == "rmse"


def test_staleness_is_the_stored_inputs_not_a_second_copy(tmp_path: Path) -> None:
    """The inputs are what is compared, so the file says *why* it went stale.

    A recorded fingerprint would be a second copy of the same fact, free to
    disagree with the inputs beside it — and when it did, the file would report
    "current" over inputs that plainly are not.
    """
    _contract(tmp_path, name="rmse", direction="minimize")
    _profile(tmp_path, target_column="y")
    stored, _ = ensure_objective(tmp_path, "demo")

    assert stored.inputs == read_inputs(tmp_path)
    on_disk = json.loads((tmp_path / OBJECTIVE_FILENAME).read_text(encoding="utf-8"))
    assert "fingerprint" not in json.dumps(on_disk)
    assert on_disk["inputs"]["target"] == "y"
    assert on_disk["inputs"]["metric_from"] == "competition.json"


def test_a_corrupt_profile_does_not_stop_the_objective(tmp_path: Path) -> None:
    """One broken file should not become two.

    A profile nothing can parse is the profile stage's problem to report;
    refusing to resolve the metric over it would withhold an answer the contract
    states plainly.
    """
    _contract(tmp_path, name="rmse", direction="minimize")
    (tmp_path / "profile.json").write_text("{not json", encoding="utf-8")

    stored = resolve_workspace_objective(tmp_path, "demo")

    assert stored.spec.metric_name == "rmse"
    assert stored.inputs.target is None
    assert stored.inputs.profile_read is True, "the file is there; it is what is in it"


# --- it runs where the pipeline runs -----------------------------------------


def test_preparing_a_workspace_resolves_its_objective(tmp_path: Path) -> None:
    """`prepare_workspace` writes the objective beside the profile.

    This is what makes it a stage rather than a CLI convenience: the pipeline
    step that produces the dataset understanding is the one that produces the
    objective read from it. Resolving only in `research conduct` left every
    other entry point — the conductor, the agent path — with no objective at
    all.
    """
    from helpers.capability_context import capability_context

    from labpilot.research_engine.execution.capabilities.workspace import (
        WorkspaceCapability,
    )
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    context = capability_context(
        tmp_path, task_type=TaskType.PREPARE_WORKSPACE, constraints={"skip_download": True}
    )
    root = Path(context.workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    _contract(root, name="rmse", direction="minimize")

    result = WorkspaceCapability().execute(context)

    assert "objective_written" in result.checks
    assert result.metadata["objective"] == str(root / OBJECTIVE_FILENAME)
    stored = load_objective(root)
    assert stored is not None and stored.spec.metric_name == "rmse"


def test_an_unresolvable_objective_is_recorded_not_raised(tmp_path: Path) -> None:
    """A blocked objective is a finding, not a broken workspace step.

    The CLI preflight is where a campaign is refused, and it refuses with the
    operator still at the keyboard and advice from their own domain. Failing the
    workspace step too would refuse the same campaign twice, with the less
    actionable message of the two — and would make `passed=False` mean two
    different things.
    """
    from helpers.capability_context import capability_context

    from labpilot.research_engine.execution.capabilities.workspace import (
        WorkspaceCapability,
    )
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    context = capability_context(
        tmp_path, task_type=TaskType.PREPARE_WORKSPACE, constraints={"skip_download": True}
    )
    root = Path(context.workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    _contract(root)  # a contract stating no metric at all

    result = WorkspaceCapability().execute(context)

    assert result.passed is True, "the workspace is prepared; the objective is not resolved"
    assert "objective_unresolved" in result.checks
    assert result.metadata["objective_unresolved"] == ["metric"]
    assert "metric" in result.metadata["objective_blocked"]
