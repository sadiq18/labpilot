"""M20 exit criterion 1: a guard ships with the failure it rejects.

Fifteen defects on 2026-08-08, **eight of them one shape** — a gate that tests
something easier than it promises, and passes. Every one read as correct. Four of
the nine defects in the 2026-08-07 log were guards that could never fire, and
each had been read and approved.

`AGENTS.md` has recorded the countermeasure since then — *feed a guard a real bad
record before trusting it* — and three of the eight gates above were written
**after** it. Advice does not hold. So this file enumerates the capabilities that
report pass/fail and requires each to have a test proving it can say **no**.

The link is a marker, `@pytest.mark.rejects("<capability>")`, rather than a
naming convention or a grep for `passed is False`. A grep finds tests that
mention failure; a marker is a claim someone had to make on purpose, and it names
which gate the claim is about.

**What changed, and why.** The marker used to be taken at its word. Deciding
which gates existed meant parsing the capability sources for `passed=` and
resolving each to a name, and seven review rounds landed inside that parser —
literal strings behind `+` and `if/else`, `no_verification` stamped from a
nested block, one file holding two capabilities, a dict keyed by name dropping a
duplicate. Every fix was right and every one left the next shape unhandled,
because the parser was reconstructing from source text something the runtime
already knows exactly.

Now `helpers/verdict_observer.py` records each verdict as it is made, and a
marked test has to actually have caused the rejection it claims. Switching over
found two markers that had never been earned: one on a test that greps a module,
one on a test that calls a helper directly. Both had been read and approved, and
neither was visible to the parser — it could see that the marker existed, never
that the test rejected nothing.

This file is what remains: which capabilities exist, and which are claimed. The
proving is the observer's job now.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

_TESTS = Path("tests")


@lru_cache(maxsize=1)
def _registry():
    from labpilot.research_engine.execution.engineer import default_capability_registry

    return default_capability_registry(install_packages=False)


def _names(capabilities, verifying_only: bool) -> set[str]:
    """Capability names, optionally only those that claim to verify.

    Split from the registry lookup so the `verifies` rule can be tested on its
    own. The default registry currently holds no `verifies = False` capability
    — `StubCapability` is registered by `default_stub_registry`, a different
    registry — so nothing here would exercise the rule, and a mutation sweep
    found it surviving deletion.

    `verifies = False` is excluded from the requirement — M20's other option,
    taken in the open: the verdict says the step ran, the card says nothing was
    checked, and a reviewer can see the claim being declined.
    """
    return {
        capability.name
        for capability in capabilities
        if getattr(capability, "verifies", True) or not verifying_only
    }


def _capability_names(verifying_only: bool) -> set[str]:
    """The capabilities the system runs, asked of the registry.

    `capabilities`, the public list, not `_by_type`. That dict is
    `TaskType -> capability` and the last `register()` for a type wins, so a
    capability whose types are all claimed by a later registration vanishes from
    it — two capabilities both declaring `{RUN_TRAINING}` give
    `capabilities == ['alpha', 'beta']` and `_by_type.values() == ['beta']`.

    Names only. An earlier version carried a file path per capability and had to
    answer two questions it got wrong — which capability a file belongs to when
    it holds two, and what to do when the class has no file. Neither question
    arises once nothing downstream reads source.
    """
    return _names(_registry().capabilities, verifying_only)


def _test_module(path: Path):
    """The module object, preferring the one pytest already imported.

    Re-importing a module pytest has loaded would give two copies of every class
    and fixture in it; under the default import mode pytest's copy is keyed by
    the bare stem, so that is where to look first.
    """
    existing = sys.modules.get(path.stem)
    if existing is not None and getattr(existing, "__file__", None) == str(path.resolve()):
        return existing
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _claims() -> dict[str, tuple[str, ...]]:
    """Every `@pytest.mark.rejects(...)` in the suite, claim -> test ids.

    Read as marker objects off the imported functions, which is what pytest
    itself does, rather than parsed out of the decorator syntax or taken from
    `session.items`. Parsing was the mechanism this file just left. `items`
    would be exact but would make the answer depend on which subset of the suite
    was invoked, so running one file could report the other 168 as uncovered.
    """
    found: dict[str, list[str]] = {}
    for path in sorted(_TESTS.rglob("test_*.py")):
        module = _test_module(path)
        for name, obj in vars(module).items():
            if not name.startswith("test_"):
                continue
            for mark in getattr(obj, "pytestmark", []):
                if mark.name != "rejects":
                    continue
                for claim in mark.args:
                    found.setdefault(claim, []).append(f"{path}::{name}")
    return {claim: tuple(tests) for claim, tests in found.items()}


def _uncovered(capabilities, claims) -> list[str]:
    """Capabilities no claim names. `reporting:reflect` covers `reporting`."""
    claimed = {claim.split(":", 1)[0] for claim in claims}
    return sorted(set(capabilities) - claimed)


def _unknown(capabilities, claims) -> list[str]:
    """Claims naming a capability that is not registered."""
    return sorted(claim for claim in claims if claim.split(":", 1)[0] not in set(capabilities))


def test_the_coverage_comparison_reports_a_capability_nobody_claims():
    """The comparisons, on doubles, because the assertions below cannot test
    them: both are `assert not <list>`, and today both lists are empty, so
    anything that always returns empty passes. A sweep confirmed it — replacing
    the difference with `capabilities - capabilities` left the suite green. What
    those assertions check is the current *state*; what this checks is that the
    check works.
    """
    assert _uncovered({"training", "reporting"}, ["reporting:reflect"]) == ["training"]
    assert _uncovered({"training"}, ["training"]) == []
    assert _uncovered({"training"}, []) == ["training"]

    assert _unknown({"training"}, ["traning:x", "training:y"]) == ["traning:x"]
    assert _unknown({"training"}, ["training"]) == []


def test_every_capability_that_reports_a_verdict_can_be_shown_to_fail():
    """The criterion itself. A capability with no rejection test is a gate
    nobody has proven can say no — and on 2026-08-08 that described eight."""
    capabilities = _capability_names(verifying_only=True)

    assert capabilities, "no capabilities found — has the registry moved?"
    missing = _uncovered(capabilities, _claims())

    assert not missing, (
        "these report pass/fail with no test proving they can reject a real bad "
        f'artifact: {missing}. Write one, marked `@pytest.mark.rejects("<name>")`, '
        "and confirm it is red before the fix and green after."
    )


def test_every_rejection_claim_names_a_capability_that_exists():
    """A marker for a capability that is not registered proves nothing and
    reports as coverage. The likeliest cause is a rename the test missed, and
    the failure it hides is total: the real capability has no rejection test and
    the count says otherwise."""
    claims = _claims()

    assert claims, "no rejection markers found at all — has the marker been renamed?"
    unknown = _unknown(_capability_names(verifying_only=False), claims)

    assert not unknown, (
        f"markers naming no registered capability: { {claim: claims[claim] for claim in unknown} }"
    )


def test_a_capability_that_declines_to_verify_is_not_required_to_reject():
    """The exemption, on doubles, because the default registry has nobody using
    it. Both directions: a capability that declines is left out of the
    requirement, and one that says nothing is held to it — a default of "exempt"
    would silently drop every capability that never thought about the question.
    """

    class _Capability:
        def __init__(self, name, **kwargs):
            self.name = name
            for key, value in kwargs.items():
                setattr(self, key, value)

    capabilities = [
        _Capability("declines", verifies=False),
        _Capability("verifies", verifies=True),
        _Capability("silent"),
    ]

    assert _names(capabilities, verifying_only=True) == {"verifies", "silent"}
    assert _names(capabilities, verifying_only=False) == {"declines", "verifies", "silent"}


def test_a_capability_shadowed_on_every_task_type_is_still_enumerated():
    """Reported reviewing PR #121, sixth round.

    `CapabilityRegistry.register` keeps two collections: `_capabilities`, which
    is complete, and `_by_type`, where the last registration for a task type
    wins. Iterating the dict drops any capability whose types are all claimed by
    a later one — no coverage requirement, every guard green.

    **Driven with a registry that actually shadows.** The first attempt at this
    guard compared two sets both built from `_by_type`, so it reproduced the bug
    rather than catching it; the second used the *real* registry, where no
    shadowing happens, so reverting the fix left it green. A guard needs the
    adversarial input, not the one that happens to be lying around — which is
    the whole subject of this file, arriving in its own tests for the third
    time.
    """
    from labpilot.research_engine.execution.registry import CapabilityRegistry
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    class _Shadowed:
        name = "shadowed"
        verifies = True
        supported_task_types = frozenset({TaskType.RUN_TRAINING})

    class _Winner:
        name = "winner"
        verifies = True
        supported_task_types = frozenset({TaskType.RUN_TRAINING})

    registry = CapabilityRegistry()
    registry.register(_Shadowed())
    registry.register(_Winner())
    assert [c.name for c in registry._by_type.values()] == ["winner"], "the dict does lose one"

    assert _names(registry.capabilities, verifying_only=True) == {"shadowed", "winner"}


def test_discovery_does_not_depend_on_how_a_capability_spells_its_name():
    """Reported reviewing PR #121, fifth round, and the end of a pattern.

    `_capability_names` matched `name = "..."` and not `name: str = "..."` — the
    form `BaseCapability` itself declares — so a capability written in the style
    it inherits from would never have been enumerated, silently.

    Five rounds of this file were spent moving one silence outward: from the
    verdict, to the resolver, to the filter deciding which files the resolver
    sees. Syntax was the wrong tool each time. *Which capabilities run* is
    answered by the registry that runs them, so `name` and `verifies` are read
    from the objects and no source style can hide one.
    """
    verifying = {
        capability.name
        for capability in _registry().capabilities
        if getattr(capability, "verifies", True)
    }

    assert verifying, "the registry must actually register something"
    assert _capability_names(verifying_only=True) == verifying


def test_declining_to_verify_is_declared_not_implied():
    """A capability that opts out has to say so on the class, where a reviewer
    reads it — and its evidence has to say so too, or the card is
    indistinguishable from one that checked something."""
    from labpilot.research_engine.execution.capabilities.stub import StubCapability

    assert StubCapability.verifies is False
    source = Path("src/labpilot/research_engine/execution/capabilities/stub.py").read_text()
    assert "stub_no_verification" in source
