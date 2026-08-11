"""Verify M15's per-tool contract fixtures against the real catalog tools.

Not the parametrized harness (M15 task #7, test_tool_contracts.py) — this
proves `helpers/tool_contract_fixtures.py`'s fixtures actually distinguish real
variance from a vacuous one, per AGENTS.md's "prove your test fails without
your fix." Each test calls the real handler through the real
`build_default_tool_registry()`, not a stand-in.

Two tests here pin *defects* rather than correct behaviour, so they fail
loudly if it changes in either direction:
`test_implement_without_force_rewrite_is_a_silent_noop` and
`test_run_plan_payload_digest_would_falsely_pass`.
"""

from __future__ import annotations

from pathlib import Path

from helpers.tool_contract_fixtures import (
    _execution_id_from,
    assert_search_papers_degraded,
    build_fixture,
    execution_capability_checks,
    normalized_digest,
)

from labpilot.research_engine.tools.catalog import build_default_tool_registry


class _FakeAnalyzer:
    """Same shape as `test_research_intelligence.py::FakeAnalyzer` — kept
    local rather than imported across test modules, to avoid coupling this
    file to another test file's internals."""

    def __init__(self, name: str, *, items=None) -> None:
        self.name = name
        self.default_enabled = True
        self._items = items or []

    def analyze(self, context):
        from labpilot.research_engine.intelligence.models import ResearchArtifacts

        return ResearchArtifacts(analyzer=self.name, items=self._items, notes=[])


def test_analyze_competition_varies_by_only(tmp_path: Path, monkeypatch) -> None:
    from labpilot.research_engine.intelligence.models import (
        ResearchArtifact,
        ResearchArtifactType,
    )
    from labpilot.research_engine.intelligence.registry import AnalyzerRegistry

    def _fake_registry() -> AnalyzerRegistry:
        reg = AnalyzerRegistry()
        reg.register(
            _FakeAnalyzer(
                "competition",
                items=[
                    ResearchArtifact(
                        id="c:1",
                        type=ResearchArtifactType.PAPER,
                        source="fixture",
                        title="competition artifact",
                        confidence=0.9,
                    )
                ],
            )
        )
        reg.register(
            _FakeAnalyzer(
                "dataset",
                items=[
                    ResearchArtifact(
                        id="d:1",
                        type=ResearchArtifactType.PAPER,
                        source="fixture",
                        title="dataset artifact",
                        confidence=0.9,
                    )
                ],
            )
        )
        return reg

    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.analyze.build_default_registry",
        _fake_registry,
    )
    fixture = build_fixture("analyze_competition", tmp_path)
    registry = build_default_tool_registry()

    result_a = registry.invoke(
        "analyze_competition",
        fixture.workspace,
        verify_auto=True,
        **fixture.inputs_a,
    )
    result_b = registry.invoke(
        "analyze_competition",
        fixture.workspace,
        verify_auto=True,
        **fixture.inputs_b,
    )

    assert result_a.data["analyzers"] == ["competition"]
    assert result_b.data["analyzers"] == ["dataset"]
    digest_a = normalized_digest(result_a.data, drop=("path", "brief_path", "report"))
    digest_b = normalized_digest(result_b.data, drop=("path", "brief_path", "report"))
    assert digest_a != digest_b


def test_generate_plan_varies_by_hypothesis(tmp_path: Path) -> None:
    fixture = build_fixture("generate_plan", tmp_path)
    registry = build_default_tool_registry()

    result_a = registry.invoke("generate_plan", fixture.workspace, **fixture.inputs_a)
    result_b = registry.invoke("generate_plan", fixture.workspace, **fixture.inputs_b)

    types_a = [str(t.type) for t in result_a.data["plan"].tasks]
    types_b = [str(t.type) for t in result_b.data["plan"].tasks]
    assert types_a != types_b, "different hypotheses produced identical task graphs"


def test_reflect_varies_by_execution(tmp_path: Path) -> None:
    fixture = build_fixture("reflect", tmp_path)
    registry = build_default_tool_registry()

    result_a = registry.invoke("reflect", fixture.workspace, **fixture.inputs_a)
    result_b = registry.invoke("reflect", fixture.workspace, **fixture.inputs_b)

    assert result_a.data["evidence_strength"] != result_b.data["evidence_strength"], (
        "different executions produced identical evidence strength"
    )


def test_search_papers_degrades_honestly(tmp_path: Path) -> None:
    fixture = build_fixture("search_papers", tmp_path)
    registry = build_default_tool_registry()

    result = registry.invoke("search_papers", fixture.workspace, **fixture.degraded_inputs)
    assert_search_papers_degraded(result.data)


def test_query_memory_varies_by_query(tmp_path: Path) -> None:
    fixture = build_fixture("query_memory", tmp_path)
    registry = build_default_tool_registry()

    result_a = registry.invoke("query_memory", fixture.workspace, **fixture.inputs_a)
    result_b = registry.invoke("query_memory", fixture.workspace, **fixture.inputs_b)

    names_a = {t["name"] for t in result_a.data["context"]["techniques"]}
    names_b = {t["name"] for t in result_b.data["context"]["techniques"]}
    assert names_a != names_b, "different queries returned identical technique sets"
    assert "Mixup" in names_a and "Mixup" not in names_b
    assert "SpecAugment" in names_b and "SpecAugment" not in names_a


def test_implement_varies_by_description(tmp_path: Path) -> None:
    from helpers.fake_codegen import FakeCodegenLLM

    fixture = build_fixture("implement", tmp_path)
    registry = build_default_tool_registry()
    llm = FakeCodegenLLM()
    train_py = fixture.workspace.root / "pipeline" / "train.py"

    registry.invoke("implement", fixture.workspace, llm_client=llm, **fixture.inputs_a)
    first = train_py.read_text(encoding="utf-8")
    registry.invoke("implement", fixture.workspace, llm_client=llm, **fixture.inputs_b)
    second = train_py.read_text(encoding="utf-8")

    assert first != second, "different descriptions produced identical train.py"


def test_implement_without_force_rewrite_is_a_silent_noop(tmp_path: Path) -> None:
    """The prefer_patch finding, pinned as a test.

    Documents current behaviour rather than asserting it is correct: the
    second call reports success while leaving `train.py` byte-identical, even
    though it asked for something different. Closing this is M7's job (see the
    2026-08-11 re-audit); until then this test fails loudly if the behaviour
    changes, in either direction.
    """
    from helpers.fake_codegen import FakeCodegenLLM

    fixture = build_fixture("implement", tmp_path)
    registry = build_default_tool_registry()
    llm = FakeCodegenLLM()
    train_py = fixture.workspace.root / "pipeline" / "train.py"

    registry.invoke(
        "implement",
        fixture.workspace,
        llm_client=llm,
        description="apply mixup augmentation",
        force_rewrite=True,
    )
    before = train_py.read_text(encoding="utf-8")

    result = registry.invoke(
        "implement",
        fixture.workspace,
        llm_client=llm,
        description="apply SWA weight averaging",
    )

    assert train_py.read_text(encoding="utf-8") == before, (
        "prefer_patch no longer short-circuits — update the re-audit finding"
    )
    assert result.refs, "the no-op still reports refs, which is what makes it silent"


def test_submit_learn_varies_by_execution_under_dry_run(tmp_path: Path) -> None:
    fixture = build_fixture("submit_learn", tmp_path)
    registry = build_default_tool_registry()

    result_a = registry.invoke("submit_learn", fixture.workspace, **fixture.inputs_a)
    result_b = registry.invoke("submit_learn", fixture.workspace, **fixture.inputs_b)

    summary_a = result_a.data["summary"]
    summary_b = result_b.data["summary"]
    assert summary_a.learning_gain != summary_b.learning_gain, (
        "dry_run=True returned identical outcomes for different executions"
    )
    assert result_a.data["dry_run"] is True and result_b.data["dry_run"] is True


def test_run_plan_varies_by_plan(tmp_path: Path) -> None:
    fixture = build_fixture("run_plan", tmp_path)
    registry = build_default_tool_registry()

    result_a = registry.invoke("run_plan", fixture.workspace, **fixture.inputs_a)
    result_b = registry.invoke("run_plan", fixture.workspace, **fixture.inputs_b)

    work_a = execution_capability_checks(fixture.workspace, result_a.data["execution_id"])
    work_b = execution_capability_checks(fixture.workspace, result_b.data["execution_id"])
    assert work_a, "the two-step plan recorded no evidence at all"
    assert work_a != work_b, "different plans executed identical work"


def test_run_plan_payload_digest_would_falsely_pass(tmp_path: Path) -> None:
    """Why `run_plan`'s contract compares evidence, not the payload (§6.2.1).

    Pins the trap rather than the fix: `ToolResult.data` differs between two
    calls *only* because `execution_id` and `plan_id` incremented. Strip
    those two id fields and the payload is byte-identical — so a naive digest
    over the payload would report this tool as proven-real even if it had
    ignored its input entirely.
    """
    fixture = build_fixture("run_plan", tmp_path)
    registry = build_default_tool_registry()

    result_a = registry.invoke("run_plan", fixture.workspace, **fixture.inputs_a)
    result_b = registry.invoke("run_plan", fixture.workspace, **fixture.inputs_b)

    naive_a = normalized_digest(result_a.data, drop=())
    naive_b = normalized_digest(result_b.data, drop=())
    assert naive_a != naive_b, "sanity: the raw payloads do differ"

    id_free = ("execution_id", "plan_id", "workspace_path")
    assert normalized_digest(result_a.data, drop=id_free) == normalized_digest(
        result_b.data, drop=id_free
    ), "payload now carries real per-plan content — revisit the evidence-set comparison"


def test_run_experiment_varies_by_plan(tmp_path: Path) -> None:
    fixture = build_fixture("run_experiment", tmp_path)
    registry = build_default_tool_registry()

    # Observe each call before the next, and compare the *work recorded*, not
    # the payload. An earlier version asserted
    # `result_a.data["plan_id"] != result_b.data["plan_id"]` — which compares
    # the two fixture inputs to each other and would hold for a
    # `run_experiment` gutted to return a constant stub.
    def _work(inputs: dict[str, object]) -> tuple:
        result = registry.invoke("run_experiment", fixture.workspace, **inputs)
        assert result.data["submit"] is False, "the specialist path must never upload"
        return tuple(execution_capability_checks(fixture.workspace, _execution_id_from(result)))

    work_a = _work(fixture.inputs_a)
    work_b = _work(fixture.inputs_b)
    assert work_a, "the two-step plan recorded no evidence at all"
    assert work_a != work_b, "different plans executed identical work"


def test_every_catalog_tool_has_a_fixture(tmp_path: Path) -> None:
    """No tool may silently lack a contract fixture (M15 exit criterion 1)."""
    from labpilot.research_engine.tools.catalog import default_tool_descriptors

    missing = []
    for descriptor in default_tool_descriptors():
        try:
            build_fixture(descriptor.name, tmp_path / descriptor.name)
        except (NotImplementedError, KeyError) as exc:
            missing.append(f"{descriptor.name}: {type(exc).__name__}")
    assert not missing, "catalog tools without a contract fixture:\n" + "\n".join(missing)
