"""Per-tool fixtures for M15's contract-test harness.

Design: docs/research-os/design/12-capability-audit.md §6.2, §6.2.1, §6.2.2.
Not the harness itself (that's test_tool_contracts.py, M15 task #7) — this
module answers the question the design's §6.2 discussion says costs real
time: for each catalog tool, what genuine, non-vacuous fixture proves (or
disproves) its declared `varies_by`.

Coverage is partial, deliberately. Eight tools are built and verified here:
`implement`, `analyze_competition`, `generate_plan`, `reflect`,
`search_papers`, `submit`, `query_memory`, `submit_learn`. The remaining two
(`run_plan`, `run_experiment`) need a live-enough training run to prove
variance under `dry_run=False` — a stub run tends to short-circuit to the
same wiring-only artifact regardless of input, the same failure shape M19
removed for `implement` — and are left for a follow-up pass rather than
shipped with an unverified or vacuous fixture. `build_fixture` raises
`NotImplementedError` for those two, with the reason, so the harness fails
loudly on them rather than silently skipping.

Building these fixtures found two real defects that reading the code had
missed, both recorded in the 2026-08-11 re-audit
(docs/research-os/autonomy-roadmap/10-capability-audit.md): `implement`'s
`prefer_patch` silent no-op, and its declared `varies_by` being wrong
(`technique` never reaches codegen; `description` does).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.workspace_facade import Workspace

_DEFERRED_TOOLS = frozenset({"run_plan", "run_experiment"})


@dataclass
class ToolFixture:
    """Everything one contract-test invocation needs for one tool.

    Not every field applies to every `capability_status` branch — `real`
    tools use `inputs_a`/`inputs_b`/`digest`; `partial` tools use
    `degraded_inputs`/`assert_degraded`; `fixed` tools use none of it (the
    harness's fixed branch never calls the handler — see §6.2).
    """

    workspace: Workspace
    inputs_a: dict[str, Any] = field(default_factory=dict)
    inputs_b: dict[str, Any] = field(default_factory=dict)
    degraded_inputs: dict[str, Any] = field(default_factory=dict)


def _base_workspace(tmp_path: Path, name: str) -> Workspace:
    ws = Workspace.from_competition(
        tmp_path / "knowledge", f"fixture-{name}", code_root=tmp_path / "ws"
    )
    ws.ensure_roots()
    return ws


def _seed_analyze_competition(tmp_path: Path) -> ToolFixture:
    """Two single-analyzer selections against a FakeAnalyzer-stubbed registry.

    Reuses the pattern `test_research_intelligence.py::FakeAnalyzer` already
    exercises for the CLI path — same double, same monkeypatch target the
    handler itself imports (`tools.handlers.analyze.build_default_registry`).
    """
    ws = _base_workspace(tmp_path, "analyze")
    # inputs_a/inputs_b just carry `only`; the caller is responsible for
    # monkeypatching build_default_registry with a FakeAnalyzer-backed
    # registry exposing at least "competition" and "dataset" before invoking
    # — a monkeypatch fixture doesn't survive being returned from here.
    return ToolFixture(
        workspace=ws,
        inputs_a={"only": "competition"},
        inputs_b={"only": "dataset"},
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
    )


def _seed_search_papers(tmp_path: Path) -> ToolFixture:
    """`partial` tool — degraded path only; §6.2's `_assert_degraded` branch."""
    ws = _base_workspace(tmp_path, "papers")
    return ToolFixture(workspace=ws, degraded_inputs={"offline": True, "query": "anything"})


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
    )


_BUILDERS = {
    "implement": _seed_implement,
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
    if name in _DEFERRED_TOOLS:
        raise NotImplementedError(
            f"contract fixture for {name!r} not yet built — see "
            "docs/research-os/design/12-capability-audit.md §6.2.2 and "
            "tool_contract_fixtures.py's module docstring for why"
        )
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


def normalized_digest(payload: dict[str, Any], *, drop: tuple[str, ...]) -> str:
    """Digest a JSON-ish payload after stripping auto-generated/volatile keys.

    §6.2.1: several tools' artifacts carry an id or timestamp that changes on
    every call regardless of input — digesting the raw payload would make the
    contract test pass for a tool that ignores its input. `drop` names the
    keys to strip before hashing (nested keys as dotted paths, one level).
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
