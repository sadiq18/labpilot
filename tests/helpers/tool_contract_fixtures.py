"""Per-tool fixtures for M15's contract-test harness.

Design: docs/research-os/design/12-capability-audit.md §6.2, §6.2.1, §6.2.2.
Not the harness itself (that's test_tool_contracts.py, M15 task #7) — this
module answers the question the design's §6.2 discussion says costs real
time: for each catalog tool, what genuine, non-vacuous fixture proves (or
disproves) its declared `varies_by`.

All ten catalog tools are covered, each verified against the real handler in
`test_tool_contract_fixtures.py` rather than asserted from the design doc.

Building these fixtures found three things reading the code had missed, all
recorded in the 2026-08-11 re-audit
(docs/research-os/autonomy-roadmap/10-capability-audit.md):

* `implement`'s `prefer_patch` silent no-op;
* `implement`'s declared `varies_by` being wrong — `technique` never reaches
  codegen, `description` does;
* `run_plan`/`run_experiment` needing an **evidence-set** comparison rather
  than a payload digest, because their `ToolResult.data` differs on every
  call purely from incrementing ids (§6.2.1's trap). The design doc's guess
  that these two needed a real `dry_run=False` training run was wrong; two
  different task *graphs* under a dry run already prove `plan_id` variance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from helpers.fake_codegen import FakeCodegenLLM
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.workspace_facade import Workspace


@dataclass
class ToolFixture:
    """Everything one contract-test invocation needs for one tool.

    Not every field applies to every branch — tools with a non-empty
    `varies_by` use `inputs_a`/`inputs_b`/`observe`; `partial` tools with no
    `varies_by` use `degraded_inputs`/`assert_degraded`; `fixed` tools use
    none of it (the harness's fixed branch never calls the handler — §6.2).

    `observe` is the load-bearing field. It must return something that
    differs *because the tool did different work*, not because an id or
    timestamp incremented — §6.2.1, and see `run_plan`, where the whole
    payload is id-noise and only the evidence set carries signal.
    """

    workspace: Workspace
    inputs_a: dict[str, Any] = field(default_factory=dict)
    inputs_b: dict[str, Any] = field(default_factory=dict)
    degraded_inputs: dict[str, Any] = field(default_factory=dict)
    #: Applied to every invocation of this tool (test doubles, gates).
    common_kwargs: dict[str, Any] = field(default_factory=dict)
    #: `(dotted.target, replacement)` pairs the harness monkeypatches in.
    patches: list[tuple[str, Any]] = field(default_factory=list)
    #: `(workspace, ToolResult) -> comparable`, id-free. Defaults to a
    #: digest of the payload, which is correct only for tools whose payload
    #: carries no auto-generated id — most override it.
    observe: Any = None
    #: Checked by the `partial`, no-`varies_by` branch.
    assert_degraded: Any = None


def _base_workspace(tmp_path: Path, name: str) -> Workspace:
    ws = Workspace.from_competition(
        tmp_path / "knowledge", f"fixture-{name}", code_root=tmp_path / "ws"
    )
    ws.ensure_roots()
    return ws


class _FakeAnalyzer:
    """Offline analyzer double — same shape the real registry expects.

    Mirrors `test_research_intelligence.py::FakeAnalyzer`; kept here so the
    fixture is self-contained rather than importing another test module's
    internals.
    """

    def __init__(self, name: str, *, items: list[Any] | None = None) -> None:
        self.name = name
        self.default_enabled = True
        self._items = items or []

    def analyze(self, context: Any) -> Any:
        from labpilot.research_engine.intelligence.models import ResearchArtifacts

        return ResearchArtifacts(analyzer=self.name, items=self._items, notes=[])


def _fake_analyzer_registry() -> Any:
    """Two named analyzers, no network, no LLM — patched over
    `tools.handlers.analyze.build_default_registry`."""
    from labpilot.research_engine.intelligence.models import (
        ResearchArtifact,
        ResearchArtifactType,
    )
    from labpilot.research_engine.intelligence.registry import AnalyzerRegistry

    def _artifact(artifact_id: str, title: str) -> Any:
        return ResearchArtifact(
            id=artifact_id,
            type=ResearchArtifactType.PAPER,
            source="fixture",
            title=title,
            confidence=0.9,
        )

    registry = AnalyzerRegistry()
    registry.register(_FakeAnalyzer("competition", items=[_artifact("c:1", "competition")]))
    registry.register(_FakeAnalyzer("dataset", items=[_artifact("d:1", "dataset")]))
    return registry


def _seed_analyze_competition(tmp_path: Path) -> ToolFixture:
    """Two single-analyzer selections against a FakeAnalyzer-stubbed registry.

    Reuses the pattern `test_research_intelligence.py::FakeAnalyzer` already
    exercises for the CLI path — same double, same monkeypatch target the
    handler itself imports (`tools.handlers.analyze.build_default_registry`).
    """
    ws = _base_workspace(tmp_path, "analyze")
    return ToolFixture(
        workspace=ws,
        inputs_a={"only": "competition"},
        inputs_b={"only": "dataset"},
        common_kwargs={"verify_auto": True},
        patches=[
            (
                "labpilot.research_engine.tools.handlers.analyze.build_default_registry",
                _fake_analyzer_registry,
            )
        ],
        # `path`/`brief_path` are workspace-scoped strings and `report` is a
        # live object; the analyzer list is the id-free signal.
        observe=lambda _ws, result: tuple(result.data["analyzers"]),
    )


def _seed_generate_plan(tmp_path: Path) -> ToolFixture:
    """Two hypotheses whose techniques route to different plan templates.

    Verified empirically (not assumed): technique="mixup" compiles to the
    augmentation template (12 tasks, includes modify_config + a second
    run_training/update_belief pair); technique="target_encoding" compiles to
    the feature_engineering template (9 tasks) — genuinely different task
    graphs, not just a different `plan.id`.
    """
    ws = _base_workspace(tmp_path, "plan")
    store = HypothesisStore(ws.knowledge_dir, ws.competition)
    h1 = store.create(
        observation="mixup observation",
        reason="mixup reason",
        prediction="mixup prediction",
        confidence=0.7,
        technique="mixup",
    )
    h2 = store.create(
        observation="target encoding observation",
        reason="target encoding reason",
        prediction="target encoding prediction",
        confidence=0.7,
        technique="target_encoding",
    )
    return ToolFixture(
        workspace=ws,
        inputs_a={"hypothesis_id": h1.id},
        inputs_b={"hypothesis_id": h2.id},
        # NOT the plan id — that increments regardless of input (§6.2.1).
        observe=lambda _ws, result: tuple(str(task.type) for task in result.data["plan"].tasks),
    )


def _seed_reflect(tmp_path: Path) -> ToolFixture:
    """Two prior executions with different metrics/comparison outcomes.

    Same shape as `test_reflection_capstone.py::test_capstone_reflect_to_journal`
    — `metrics.json` + `baseline_choice.json` + `artifacts/comparison.json`
    under an execution-scoped directory, passed as `workspace_path`. Verified
    empirically: cv_accuracy 0.79/delta +0.012 yields evidence_strength
    "strong"; 0.55/delta -0.05 yields "rejected" — a real classification
    difference, not just a different `execution_id` echoed back.
    """
    ws = _base_workspace(tmp_path, "reflect")

    def _seed_execution_dir(path: Path, *, accuracy: float, delta: float, verdict: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_text(
            json.dumps({"cv_accuracy": accuracy, "runtime_seconds": 8.0}),
            encoding="utf-8",
        )
        (path / "baseline_choice.json").write_text(
            json.dumps(
                {
                    "template_name": "tabular_classification",
                    "problem_type": "tabular_classification",
                    "metric_name": "accuracy",
                }
            ),
            encoding="utf-8",
        )
        (path / "artifacts").mkdir(exist_ok=True)
        (path / "artifacts" / "comparison.json").write_text(
            json.dumps(
                {
                    "compare_to": "P-001",
                    "delta": delta,
                    "verdict": verdict,
                    "maximize": True,
                    "outcome": "improved" if delta > 0 else "regressed",
                }
            ),
            encoding="utf-8",
        )

    dir_a = tmp_path / "exec-a"
    dir_b = tmp_path / "exec-b"
    _seed_execution_dir(dir_a, accuracy=0.79, delta=0.012, verdict="worth_keeping")
    _seed_execution_dir(dir_b, accuracy=0.55, delta=-0.05, verdict="not_worth_keeping")

    return ToolFixture(
        workspace=ws,
        inputs_a={
            "execution_id": "E-fixture-a",
            "workspace_path": str(dir_a),
            "plan_id": "P-fixture-a",
            "persist": False,
        },
        inputs_b={
            "execution_id": "E-fixture-b",
            "workspace_path": str(dir_b),
            "plan_id": "P-fixture-b",
            "persist": False,
        },
        # Evidence strength is the classification; the ids are noise.
        observe=lambda _ws, result: result.data["evidence_strength"],
    )


def _seed_search_papers(tmp_path: Path) -> ToolFixture:
    """`partial` tool — degraded path only; §6.2's `_assert_degraded` branch."""
    ws = _base_workspace(tmp_path, "papers")
    return ToolFixture(
        workspace=ws,
        degraded_inputs={"offline": True, "query": "anything"},
        assert_degraded=assert_search_papers_degraded,
    )


def _seed_submit(tmp_path: Path) -> ToolFixture:
    """`fixed` tool — the harness's fixed branch never calls the handler."""
    ws = _base_workspace(tmp_path, "submit")
    return ToolFixture(workspace=ws)


#: Filler techniques deliberately outnumber the default retrieval limit (12)
#: so that whichever target technique gets the query's +0.1 term-match boost
#: (see fetchers.py's `_select_techniques`) crosses the cutoff and the other
#: doesn't. Confidence gap (fillers 0.5, targets 0.45 base) makes this a
#: guaranteed score difference, not a tiebreak.
_QUERY_MEMORY_FILLER_COUNT = 20
_QUERY_MEMORY_FILLER_CONFIDENCE = 0.5
_QUERY_MEMORY_TARGET_BASE_CONFIDENCE = 0.45


def _seed_query_memory(tmp_path: Path) -> ToolFixture:
    """Two named techniques, deterministically ranked in/out by query text.

    Verified this needs `store.merge_technique(name, confidence=X)` directly
    — not `ResearchArtifact` + `KnowledgeHub.ingest()`, which computes its own
    concept-cluster confidence (`_unit_confidence` in `knowledge/hub.py`)
    independent of whatever confidence the artifact declares. A first attempt
    seeding through the hub produced an *identical* technique set for two
    different queries; going through `merge_technique` directly gives exact
    control over the score `_select_techniques` ranks on.

    Verified empirically through the real `query_memory` handler (not just
    `build_research_context`): querying "Mixup" returns `Mixup` in
    `data["context"]["techniques"]` and excludes `SpecAugment`; querying
    "SpecAugment" is the reverse.
    """
    from labpilot.research_engine.intelligence.knowledge import KnowledgeStore

    ws = _base_workspace(tmp_path, "memory")
    with KnowledgeStore(ws.knowledge_dir, ws.competition) as store:
        for i in range(_QUERY_MEMORY_FILLER_COUNT):
            store.merge_technique(f"Filler{i}", confidence=_QUERY_MEMORY_FILLER_CONFIDENCE)
        store.merge_technique("Mixup", confidence=_QUERY_MEMORY_TARGET_BASE_CONFIDENCE)
        store.merge_technique("SpecAugment", confidence=_QUERY_MEMORY_TARGET_BASE_CONFIDENCE)

    return ToolFixture(
        workspace=ws,
        inputs_a={"query": "Mixup"},
        inputs_b={"query": "SpecAugment"},
        observe=lambda _ws, result: tuple(
            sorted(t["name"] for t in result.data["context"]["techniques"])
        ),
    )


def _seed_submit_learn(tmp_path: Path) -> ToolFixture:
    """Two prior executions with different stored outcomes, `dry_run=True`.

    Verified `dry_run=True` doesn't fake this — `load_execution_outcome`
    picks up a pre-written per-execution `execution_outcome.json` (under
    `<knowledge_dir>/<competition>/executions/<execution_id>/artifacts/`)
    before falling back to reconstructing one, so two executions with
    different `learning_gain` genuinely return different summaries.

    Needs three stores wired together — a real `PlanStore` row (execution
    creation validates `plan_id` exists), a real `ExecutionArtifacts`-created
    execution per id, and a packaged `submission_<id>.csv` (`submit_learn`'s
    `resolve_submission_csv` raises immediately if that's missing — the
    auto-package fallback in the handler only covers the *default* filename,
    not a pre-existing scoped one, so this fixture packages it explicitly
    rather than relying on that fallback).
    """
    from datetime import UTC, datetime

    from labpilot.research_engine.artifacts.execution import ExecutionArtifacts
    from labpilot.research_engine.execution.outcome import (
        ExecutionOutcomeSummary,
        package_execution_submission,
    )
    from labpilot.research_engine.intelligence.paths import ResearchPaths
    from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
    from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
    from labpilot.research_engine.planner.store import PlanStore

    ws = _base_workspace(tmp_path, "submit-learn")
    (ws.root / "submission.csv").write_text("id,prediction\n1,0.5\n", encoding="utf-8")

    now = datetime.now(UTC)
    plan = ResearchPlan(
        id="P-fixture",
        competition=ws.competition,
        hypothesis_id="",
        goal="mini",
        status=PlanStatus.READY,
        tasks=[
            ResearchTask(
                id="P-fixture-T01",
                plan_id="P-fixture",
                type=TaskType.RUN_TRAINING,
                description="t",
                order=0,
            )
        ],
        created_at=now,
        updated_at=now,
    )
    plan_store = PlanStore(ws.knowledge_dir, ws.competition)
    plan_store.upsert_plan(plan)
    plan_store.close()

    exec_arts = ExecutionArtifacts(ws.knowledge_dir, ws.competition)
    exec_a, _ = exec_arts.create("P-fixture", workspace_path=str(ws.root))
    exec_b, _ = exec_arts.create("P-fixture", workspace_path=str(ws.root))
    exec_arts.close()

    package_execution_submission(ws.root, exec_a.id)
    package_execution_submission(ws.root, exec_b.id)

    paths = ResearchPaths(ws.knowledge_dir, ws.competition)

    def _write_outcome(execution_id: str, gain: float) -> None:
        summary = ExecutionOutcomeSummary(
            competition=ws.competition,
            execution_id=execution_id,
            plan_id="P-fixture",
            learning_gain=gain,
            metrics={"cv_accuracy": 0.5 + gain},
        )
        out_dir = paths.executions_dir / execution_id / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "execution_outcome.json").write_text(summary.model_dump_json(), encoding="utf-8")

    _write_outcome(exec_a.id, 0.03)
    _write_outcome(exec_b.id, -0.02)

    return ToolFixture(
        workspace=ws,
        inputs_a={"execution_id": exec_a.id, "dry_run": True},
        inputs_b={"execution_id": exec_b.id, "dry_run": True},
        observe=lambda _ws, result: result.data["summary"].learning_gain,
    )


def _seed_implement(tmp_path: Path) -> ToolFixture:
    """Two descriptions through the extended `FakeCodegenLLM`.

    Two things this fixture encodes that the design doc's §6.2.2 table got
    wrong, both found by building it:

    * **`force_rewrite=True` is mandatory.** Without it,
      `ImplementationSpecialist`'s `prefer_patch` shortcut returns the
      untouched `train.py` whenever one exists, and the contract test would
      pass vacuously against a no-op. The first invocation on a fresh
      workspace would work either way; the second would not.
    * **It varies by `description`, not `technique`.** The `technique` kwarg
      never reaches the codegen prompt on this path (see `catalog.py`'s note
      — task metadata is written to the `ResearchTask` and read from the
      `ResearchPlan`), so the prompt always renders `Technique: —`. Passing
      two techniques and asserting a digest difference would fail, correctly,
      and the fix is an honest `varies_by`, not a louder fixture.

    `profile.json` is required: `_write` refuses a non-dry run without a
    dataset profile.
    """
    ws = _base_workspace(tmp_path, "implement")
    (ws.root / "profile.json").write_text('{"n_rows": 100, "n_cols": 5}', encoding="utf-8")
    return ToolFixture(
        workspace=ws,
        inputs_a={"description": "apply mixup augmentation", "force_rewrite": True},
        inputs_b={"description": "apply SWA weight averaging", "force_rewrite": True},
        common_kwargs={"llm_client": FakeCodegenLLM()},
        # The written file, not the ToolResult — `paths` are workspace-scoped
        # and identical across both calls (same train.py path, new content).
        observe=lambda ws, _result: (ws.root / "pipeline" / "train.py").read_text(encoding="utf-8"),
    )


def _seed_two_plans(ws: Workspace) -> tuple[str, str]:
    """Two plans with genuinely different task *graphs*, not just ids.

    `READ_CODE` / `MODIFY_CONFIG` only — deliberately no `RUN_TRAINING`, so
    the fixture needs no dataset and spawns no subprocess. That is honest for
    what `varies_by=["plan_id"]` claims: a different plan must produce a
    different execution. It does **not** claim training itself varies, which
    is a separate question these tools' contract does not assert.
    """
    from datetime import UTC, datetime

    from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
    from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
    from labpilot.research_engine.planner.store import PlanStore

    def _plan(plan_id: str, types: list[Any]) -> ResearchPlan:
        now = datetime.now(UTC)
        tasks = [
            ResearchTask(
                id=f"{plan_id}-T{index:02d}",
                plan_id=plan_id,
                type=task_type,
                description=str(task_type),
                order=index,
                dependencies=[f"{plan_id}-T{index - 1:02d}"] if index else [],
            )
            for index, task_type in enumerate(types)
        ]
        return ResearchPlan(
            id=plan_id,
            competition=ws.competition,
            hypothesis_id="",
            goal="contract fixture",
            status=PlanStatus.READY,
            tasks=tasks,
            created_at=now,
            updated_at=now,
        )

    store = PlanStore(ws.knowledge_dir, ws.competition)
    try:
        store.upsert_plan(_plan("P-two-step", [TaskType.READ_CODE, TaskType.MODIFY_CONFIG]))
        store.upsert_plan(_plan("P-one-step", [TaskType.READ_CODE]))
    finally:
        store.close()
    return "P-two-step", "P-one-step"


def _seed_run_plan(tmp_path: Path) -> ToolFixture:
    """Two plans whose executions record different work.

    **Corrects the design doc's §6.2.2 assumption.** That table predicted
    `dry_run=True` would be "very likely not enough" to prove variance and
    that this needed a real `dry_run=False` training run against a synthetic
    dataset. Measured instead: with two genuinely different task *graphs*, a
    dry run already writes different per-task evidence
    (`<executions_dir>/<execution_id>/evidence/<task_id>.json`, one file per
    task, each recording its `capability` and `checks`). The doc's concern —
    a stub short-circuiting to the same artifact — applies to plans that
    differ only in *technique*, which is not what `varies_by=["plan_id"]`
    claims.

    **What must be compared matters more than the inputs here.** `run_plan`'s
    `ToolResult.data` is `{execution_id, plan_id, status, error,
    workspace_path}` and `ResearchExecution.metadata` is empty — so a raw
    digest differs for two calls *purely because the ids incremented*, and
    would pass for a tool that ignored its input entirely. This is §6.2.1's
    false-real-verdict trap in its purest form in this catalog; the harness
    must compare the evidence set (see `execution_capability_checks`), not
    the payload.
    """
    ws = _base_workspace(tmp_path, "run-plan")
    plan_a, plan_b = _seed_two_plans(ws)
    return ToolFixture(
        workspace=ws,
        inputs_a={"plan_id": plan_a, "dry_run": True},
        inputs_b={"plan_id": plan_b, "dry_run": True},
        observe=lambda ws, result: tuple(
            execution_capability_checks(ws, result.data["execution_id"])
        ),
    )


def _seed_run_experiment(tmp_path: Path) -> ToolFixture:
    """Same two-plan shape as `run_plan`, through the experiment specialist.

    Kept as its own fixture rather than aliased: the 2026-08-02 audit scored
    these two jointly, and the whole point of the re-audit splitting them is
    that they are independent handlers which could diverge.
    """
    ws = _base_workspace(tmp_path, "run-experiment")
    plan_a, plan_b = _seed_two_plans(ws)
    return ToolFixture(
        workspace=ws,
        inputs_a={"plan_id": plan_a, "dry_run": True},
        inputs_b={"plan_id": plan_b, "dry_run": True},
        # Same evidence-set comparison as `run_plan`. An earlier version
        # observed `(data["plan_id"], bool(data["experiment_path"]))`, which
        # was **vacuous**: the handler echoes `plan_id` straight back from
        # its argument, and `experiment_path` is the same file both times
        # (`.../ws/experiment/record.json`, overwritten per run), so the only
        # thing that differed was the input compared against itself. That is
        # precisely the false-real-verdict §6.2.1 exists to prevent, and it
        # would have passed a `run_experiment` gutted to ignore its plan.
        #
        # The execution id is reachable after all — via `result.refs`, not
        # `data` — so the honest signal is available here too.
        observe=lambda ws, result: tuple(
            execution_capability_checks(ws, _execution_id_from(result))
        ),
    )


_BUILDERS = {
    "implement": _seed_implement,
    "run_plan": _seed_run_plan,
    "run_experiment": _seed_run_experiment,
    "analyze_competition": _seed_analyze_competition,
    "generate_plan": _seed_generate_plan,
    "reflect": _seed_reflect,
    "search_papers": _seed_search_papers,
    "submit": _seed_submit,
    "query_memory": _seed_query_memory,
    "submit_learn": _seed_submit_learn,
}


def build_fixture(name: str, tmp_path: Path) -> ToolFixture:
    """Return the fixture for `name`, or raise for the not-yet-built five."""
    builder = _BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"no contract fixture registered for tool {name!r}")
    return builder(tmp_path)


def assert_search_papers_degraded(data: dict[str, Any]) -> None:
    """§6.2's per-tool `_assert_degraded` for `search_papers` specifically —
    not generic (a generic `error`-key check was tried and found vacuous;
    see §6.2's discussion in the design doc)."""
    assert data.get("source") == "offline", (
        f"search_papers: degraded path should report source='offline', got {data.get('source')!r}"
    )
    assert data.get("papers") == [], (
        f"search_papers: degraded path should return no papers, got {data.get('papers')!r}"
    )


def _execution_id_from(result: Any) -> str:
    """The execution id off a ToolResult's refs.

    `run_experiment` does not put it in `data` the way `run_plan` does, but
    it does emit an `execution`-kind ref — which is what makes the same
    id-free evidence comparison available to both.
    """
    for ref in result.refs:
        if ref.kind == "execution":
            return str(ref.id)
    raise AssertionError(
        "no execution ref on this ToolResult, so its work cannot be observed "
        "id-free — do not fall back to comparing echoed inputs (§6.2.1)"
    )


def execution_capability_checks(workspace: Workspace, execution_id: str) -> list[tuple[str, str]]:
    """What an execution actually *did*, id-free — for `run_plan`-shaped tools.

    Reads `<executions_dir>/<execution_id>/evidence/<task_id>.json` (one file
    per executed task) and returns sorted `(capability, check)` pairs. The
    task ids embed the plan id and the execution dir embeds the execution id,
    so both are deliberately dropped: what survives is only the work done,
    which is the thing that must differ when the plan differs (§6.2.1).
    """
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    paths = ResearchPaths(workspace.knowledge_dir, workspace.competition)
    evidence_dir = paths.executions_dir / execution_id / "evidence"
    if not evidence_dir.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    for path in sorted(evidence_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        capability = str(record.get("capability") or "")
        for check in record.get("checks") or []:
            pairs.append((capability, str(check)))
    return sorted(pairs)


def normalized_digest(payload: dict[str, Any], *, drop: tuple[str, ...]) -> str:
    """Digest a JSON-ish payload after stripping auto-generated/volatile keys.

    §6.2.1: several tools' artifacts carry an id or timestamp that changes on
    every call regardless of input — digesting the raw payload would make the
    contract test pass for a tool that ignores its input.

    `drop` is a flat set of key *names*, removed wherever they appear at any
    nesting depth. Dotted paths are **not** supported — an earlier docstring
    claimed they were, which would have let a caller write
    ``drop=("summary.created_at",)``, strip nothing, and reintroduce the very
    false positive this helper exists to remove. Scope a drop by renaming the
    field or pre-trimming the payload instead.
    """

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k not in drop}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    normalized = _strip(payload)
    blob = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
