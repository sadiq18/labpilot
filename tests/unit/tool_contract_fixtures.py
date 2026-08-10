"""Per-tool fixtures for M15's contract-test harness.

Design: docs/research-os/design/12-capability-audit.md §6.2, §6.2.1, §6.2.2.
Not the harness itself (that's test_tool_contracts.py, M15 task #7) — this
module answers the question the design's §6.2 discussion says costs real
time: for each catalog tool, what genuine, non-vacuous fixture proves (or
disproves) its declared `varies_by`.

Coverage is partial, deliberately. Five tools are built and verified here:
`analyze_competition`, `generate_plan`, `reflect`, `search_papers`, `submit`.
The other five (`implement`, `run_plan`, `run_experiment`, `submit_learn`,
`query_memory`) turned out harder than the design doc's table assumed —
`query_memory`'s retrieval only reorders/scores techniques rather than
filtering them out at the default limit, so a naive two-technique seed
produces an *identical* technique set for two different queries (verified
empirically, not assumed) — and are left for a follow-up pass rather than
shipped with an unverified or vacuous fixture. `_fixture_workspace` raises
`NotImplementedError` for those five, with the reason, so the harness fails
loudly on them rather than silently skipping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.workspace_facade import Workspace

_DEFERRED_TOOLS = frozenset(
    {"implement", "run_plan", "run_experiment", "submit_learn", "query_memory"}
)


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


_BUILDERS = {
    "analyze_competition": _seed_analyze_competition,
    "generate_plan": _seed_generate_plan,
    "reflect": _seed_reflect,
    "search_papers": _seed_search_papers,
    "submit": _seed_submit,
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
