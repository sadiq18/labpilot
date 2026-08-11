"""M15's contract-test harness — one parametrized test over the whole catalog.

Design: docs/research-os/design/12-capability-audit.md §6.2.
Exit criterion 1: *every catalog tool has a contract test proving different
inputs yield different artifacts, or is renamed to admit it is fixed.*

The assertion is uniform; only the fixtures differ, and those live in
`tool_contract_fixtures.py`. Three branches, chosen by what the descriptor
declares about itself:

* **declares `varies_by`** — two inputs differing only in a declared key must
  produce different *work*. What counts as "work" is per-tool
  (`ToolFixture.observe`) and must be id-free: see §6.2.1, and
  `test_tool_contract_fixtures.py::test_run_plan_payload_digest_would_falsely_pass`
  for why a payload digest is not good enough for at least one tool here.
* **`partial` with no `varies_by`** — the degraded path must fail honestly
  rather than raise or fake a real result.
* **`fixed`** — nothing to prove about output, but the *name* must not read
  as an action (exit criterion 3).

This file is the standing check. `test_tool_contract_fixtures.py` is its
companion: that one proves each individual fixture is non-vacuous, this one
proves the whole catalog is covered by them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tool_contract_fixtures import build_fixture

from labpilot.research_engine.tools.catalog import (
    build_default_tool_registry,
    default_tool_descriptors,
)

#: Verbs that promise the caller a capability. A step that cannot vary its
#: output must not be named with one — that is the whole "rename what stays
#: fixed" half of M15's exit criterion 3. Deliberately a small explicit list
#: rather than a heuristic (design doc, Open Questions #1): the catalog has
#: ten entries, and a wrong guess here would be a failing build for a
#: correctly-named tool.
_ACTION_VERBS = frozenset(
    {
        "implement",
        "optimise",
        "optimize",
        "tune",
        "train",
        "improve",
        "fix",
        "solve",
        "write",
        "build",
    }
)

_TOOL_NAMES = sorted(descriptor.name for descriptor in default_tool_descriptors())


def _reads_as_action_verb(name: str) -> bool:
    """Whether a tool name promises work it may not do.

    Matches on the leading word only: `submit` and `reflect` are honest
    names for what they do, while `implement_model` would not be. Compound
    names split on `_`, so `run_plan` is checked as `run` — which is not in
    the list, because "run this specific plan" describes a step rather than
    promising an outcome.
    """
    return name.split("_", 1)[0].lower() in _ACTION_VERBS


@pytest.mark.parametrize("name", _TOOL_NAMES)
def test_tool_contract(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = build_default_tool_registry()
    tool = registry.require(name)
    fixture = build_fixture(name, tmp_path)

    for target, replacement in fixture.patches:
        monkeypatch.setattr(target, replacement)

    if not tool.varies_by:
        # Exit criterion 3, applied to *any* tool that cannot prove variance
        # — not only `fixed` ones. A `partial` tool with no `varies_by` makes
        # the same promise a `fixed` one does and keeps it no better, so
        # gating this on status alone would leave a hole exactly the width of
        # a hypothetical `optimise` that merely degrades politely.
        assert not _reads_as_action_verb(name), (
            f"{name} declares varies_by=[] but is named like a capability — "
            "either make it vary, or rename it to admit it is a fixed step "
            "(M15 exit criterion 3)"
        )

        if tool.capability_status == "partial":
            result = registry.invoke(
                name,
                fixture.workspace,
                **fixture.common_kwargs,
                **fixture.degraded_inputs,
            )
            assert fixture.assert_degraded is not None, (
                f"{name} is partial with no varies_by but declares no degraded check"
            )
            fixture.assert_degraded(result.data)
        return

    assert fixture.observe is not None, f"{name} declares varies_by but no observe()"

    # Observe each call *before* making the next one. Not cosmetic: several
    # tools' real artifact is a file at a fixed path (`implement` rewrites
    # `pipeline/train.py`), so invoking both and observing afterwards reads
    # the second call's output twice and reports "identical work" for a tool
    # that varied perfectly well. That is a false *negative*, the mirror of
    # the false positive §6.2.1 warns about, and it failed exactly this way
    # on first run.
    def _run(inputs: dict[str, object]) -> object:
        result = registry.invoke(name, fixture.workspace, **fixture.common_kwargs, **inputs)
        return fixture.observe(fixture.workspace, result)

    observed_a = _run(fixture.inputs_a)
    observed_b = _run(fixture.inputs_b)

    # Guard the guard: an observe() returning an empty container for both
    # calls would make the inequality below trivially unfalsifiable (AGENTS.md
    # rule 4: "if a test could pass on an empty list, assert non-empty").
    #
    # Emptiness, not falsiness: `0`, `0.0` and `False` are all legitimate
    # observations — `submit_learn` observes a `learning_gain` float, and a
    # gain of exactly 0.0 is a real result, not a missing one. A bare
    # `assert observed_a` would reject it as "proves nothing".
    assert observed_a is not None, f"{name}: observe() returned None"
    if isinstance(observed_a, (str, bytes, list, tuple, set, dict)):
        assert len(observed_a) > 0, (
            f"{name}: observe() returned an empty {type(observed_a).__name__} — "
            "the fixture proves nothing"
        )

    assert observed_a != observed_b, (
        f"{name} declares varies_by={tool.varies_by} but varying it produced "
        f"identical work: {observed_a!r}"
    )


@pytest.mark.parametrize("name", ["generate_plan", "query_memory", "run_plan", "run_experiment"])
def test_the_contract_would_catch_a_hollow_tool(name: str, tmp_path: Path) -> None:
    """Prove the harness can fail — feed one input twice and watch it collapse.

    AGENTS.md rule 4: a check that cannot fail is not a check. The assertion
    in `test_tool_contract` is `observed_a != observed_b`, so its whole value
    rests on `observe()` returning the *same* thing when the tool does the
    same work. If a tool went hollow tomorrow — same output whatever the
    input — this is the property that makes the contract notice.

    Four tools rather than all ten: one payload-observing
    (`generate_plan`), one store-observing (`query_memory`), and both
    evidence-observing ones (`run_plan`, `run_experiment`).

    `run_experiment` is here because leaving it out is how its observe()
    shipped broken: it read back `data["plan_id"]`, the caller's own
    argument, so the contract passed on the inputs differing rather than on
    the tool doing anything. A list that samples strategies is only as good
    as its claim to cover them, so this enumerates rather than samples.
    """
    registry = build_default_tool_registry()

    # A fresh fixture per invocation, not one reused three times: `run_plan`
    # marks its plan `done` and refuses a second run, and reusing a workspace
    # would also let the first call's side effects colour the second. Each
    # run therefore starts from identical-but-separate state, which is what
    # makes "same input, same observation" a meaningful claim.
    def _run(which: str, slot: str) -> object:
        fixture = build_fixture(name, tmp_path / slot)
        inputs = fixture.inputs_a if which == "a" else fixture.inputs_b
        result = registry.invoke(name, fixture.workspace, **fixture.common_kwargs, **inputs)
        return fixture.observe(fixture.workspace, result)

    repeated = (_run("a", "first"), _run("a", "second"))
    assert repeated[0] == repeated[1], (
        f"{name}: observe() differs across two IDENTICAL calls, so the contract "
        "test passes on noise rather than on real variance — this is the "
        "false-real-verdict failure of §6.2.1"
    )
    assert _run("b", "third") != repeated[0], (
        f"{name}: sanity — the declared varies_by input must still change the result"
    )


def test_no_action_named_tool_is_a_fixed_step() -> None:
    """M15 exit criterion 3, stated once over the whole catalog.

    The parametrized test enforces this per tool, but only reaches the check
    for tools with an empty `varies_by`. This states the criterion in the
    form the milestone words it, so the guarantee survives a refactor of the
    branch above.

    Current verdict (2026-08-11 re-audit): **no tool needs renaming.**
    `submit` is the only `fixed` entry and reads as a noun-ish step, not a
    promise. `implement` *is* an action verb, and stays — it genuinely varies
    (by `description`), so it is not a fixed step dressed as a capability.
    """
    offenders = [
        descriptor.name
        for descriptor in default_tool_descriptors()
        if not descriptor.varies_by and _reads_as_action_verb(descriptor.name)
    ]
    assert not offenders, (
        "these tools promise an action but cannot vary their output — "
        f"rename them or make them vary: {offenders}"
    )


def test_every_catalog_tool_is_parametrized() -> None:
    """The parametrization is derived, not hand-listed — pin that it stayed so.

    Without this, adding an eleventh tool to the catalog and forgetting to
    touch this file would silently leave it uncovered, which is exactly the
    drift M15 exists to stop.
    """
    assert set(_TOOL_NAMES) == {d.name for d in default_tool_descriptors()}
    assert len(_TOOL_NAMES) == 10, (
        f"catalog size changed to {len(_TOOL_NAMES)} — confirm the new tool has a "
        "fixture in tool_contract_fixtures.py, then update this count"
    )


def test_a_descriptor_cannot_contradict_itself() -> None:
    """`capability_status` and `varies_by` must agree at construction.

    Both directions are incoherent and both used to be accepted:
    `real` with nothing declared is an unproven capability claim, and `fixed`
    with a declared input says "nothing changes my output" and "these inputs
    change my output" in the same breath.
    """
    import pytest as _pytest

    from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult

    def _handler(_workspace: object, **_kwargs: object) -> ToolResult:
        return ToolResult()

    with _pytest.raises(ValueError, match="varies_by=\\[\\]"):
        ToolDescriptor(name="t", handler=_handler, capability_status="real")

    with _pytest.raises(ValueError, match="varies by nothing"):
        ToolDescriptor(name="t", handler=_handler, capability_status="fixed", varies_by=["q"])

    # The coherent combinations still build.
    ToolDescriptor(name="t", handler=_handler, capability_status="fixed")
    ToolDescriptor(name="t", handler=_handler, capability_status="partial")
    ToolDescriptor(name="t", handler=_handler, capability_status="real", varies_by=["q"])


#: The re-audited verdict per tool (2026-08-11). Pinned because the contract
#: harness cannot check a status directly: it branches on `varies_by`, so a
#: tool that varies satisfies the same assertions whether it is labelled
#: `real` or `partial`. Without this, flipping `implement` to `real` would
#: pass the whole suite and start advertising a capability the `prefer_patch`
#: no-op does not deliver.
_EXPECTED_STATUS = {
    "analyze_competition": "real",
    "generate_plan": "real",
    "implement": "partial",
    "query_memory": "real",
    "reflect": "real",
    "run_experiment": "real",
    "run_plan": "real",
    "search_papers": "partial",
    "submit": "fixed",
    "submit_learn": "real",
}


def test_declared_statuses_match_the_re_audit() -> None:
    """A status is a claim; changing one must be a deliberate act.

    `implement` is the reason this exists. It is `partial` because
    `ImplementationSpecialist`'s `prefer_patch` shortcut makes it a silent
    no-op on any workspace that already has code — pinned separately by
    `test_tool_contract_fixtures.py::test_implement_without_force_rewrite_is_a_silent_noop`.
    That test proves the defect; this one keeps the *label* honest about it.
    Promote `implement` only when that test is deleted because the no-op is
    gone.
    """
    actual = {d.name: d.capability_status for d in default_tool_descriptors()}
    assert actual == _EXPECTED_STATUS, (
        "a tool's capability_status changed — confirm the code earns the new "
        "verdict, update docs/research-os/autonomy-roadmap/10-capability-audit.md, "
        "then update _EXPECTED_STATUS"
    )
