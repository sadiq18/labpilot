"""A test must not assert equality against a value only this machine has.

The gap this closes. Mutation testing proves an assertion is wired to the code
it covers; it cannot prove the assertion means the same thing anywhere else. On
the M11 fan-out branch a test read `available_cpus()` and asserted `== cpus`,
which passed on a ten-core laptop and failed on a two-core CI runner — the
assertion was about the box, not the behaviour. The fix was to inject the count
(`resolve_k(..., cpus=10)`) so the assertion says one thing everywhere.

Scoped to **equality** deliberately. An inequality against an ambient value is
usually the honest form of the same test — `assert resolve_k(...) <= max(2,
machine)` holds on any machine, and flagging it would train people to silence
the check. Timing assertions (`perf_counter`) are inequalities for the same
reason and are left alone.

Companion to `tests/plugins/hostile_env.py`, which runs the suite under a faked
CPU count and a narrow terminal — this catches the defect structurally, that
one catches it empirically.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]

#: Calls whose answer is a property of the machine, the clock, or the terminal.
_AMBIENT = frozenset(
    {
        "available_cpus",
        "cpu_count",
        "process_cpu_count",
        "sched_getaffinity",
        "gethostname",
        "getpid",
        "getcwd",
        "get_terminal_size",
        "now",
        "today",
        "monotonic",
        "perf_counter",
    }
)

#: `(test file stem, test name)` allowed to compare against an ambient value,
#: because the ambient value *is* the subject under test. Kept as exact pairs
#: rather than whole-file exemptions so a new test in these files is still
#: checked.
_SUBJECT_IS_THE_MACHINE = frozenset(
    {
        # These test `available_cpus()` and `cpu_share()` themselves — asserting
        # what the machine reports is the entire point.
        ("test_compute_budget", "test_discovery_prefers_the_affinity_aware_source"),
        (
            "test_compute_budget",
            "test_discovery_falls_through_when_the_preferred_source_answers_none",
        ),
        ("test_compute_budget", "test_a_share_larger_than_the_machine_is_clamped"),
        # Asserts the campaign stamped *this* process, which is the behaviour.
        ("test_fanout_in_the_loop", "test_a_campaign_stamps_the_process_running_it"),
    }
)


def _ambient_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = (
        func.id
        if isinstance(func, ast.Name)
        else func.attr
        if isinstance(func, ast.Attribute)
        else None
    )
    return name if name in _AMBIENT else None


def _offenders_in(path: Path) -> list[tuple[str, str, int]]:
    """`(stem, test name, line)` for each equality against an ambient value."""
    found: list[tuple[str, str, int]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test_"):
            continue
        # Names bound to an ambient call, so `cpus = available_cpus()` followed
        # by `== cpus` is caught — which is the shape the real defect took.
        tainted = {
            target.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Assign)
            and any(_ambient_name(c) for c in ast.walk(node.value))
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                continue
            for side in (node.left, *node.comparators):
                names = {n.id for n in ast.walk(side) if isinstance(n, ast.Name)}
                if (names & tainted) or any(_ambient_name(c) for c in ast.walk(side)):
                    found.append((path.stem, fn.name, node.lineno))
                    break
    return found


def test_no_test_asserts_equality_against_an_ambient_value() -> None:
    offenders = [
        (stem, name, line)
        for path in sorted(TESTS.rglob("test_*.py"))
        for stem, name, line in _offenders_in(path)
        if (stem, name) not in _SUBJECT_IS_THE_MACHINE
    ]
    assert offenders == [], (
        "these assert equality against the machine, the clock, or the terminal, "
        "so they mean something different on another box — inject the value, or "
        "assert an inequality that holds anywhere: "
        + ", ".join(f"{s}.{n}:{ln}" for s, n, ln in offenders)
    )


def test_every_exemption_still_names_a_real_test() -> None:
    """An allowlist nobody prunes becomes a blanket exemption.

    A renamed or deleted test would otherwise leave an entry that silently
    excuses whatever takes its name next.
    """
    live = {
        (path.stem, fn.name)
        for path in sorted(TESTS.rglob("test_*.py"))
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_")
    }
    assert _SUBJECT_IS_THE_MACHINE <= live, (
        f"stale exemptions: {sorted(_SUBJECT_IS_THE_MACHINE - live)}"
    )


def test_the_rule_catches_the_defect_it_was_written_for() -> None:
    """The M11 shape, verbatim. A checker that flags nothing is indistinguishable
    from a clean tree, and this file's whole value is that it would have failed."""
    source = '''
def test_k_is_capped_by_the_cores_the_machine_actually_has():
    cpus = available_cpus()
    over = cpus * 4
    assert resolve_k(over, available=over) == cpus
'''
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "test_probe.py"
        probe.write_text(source, encoding="utf-8")
        assert _offenders_in(probe), "the rule no longer catches the original defect"


def test_an_inequality_against_the_machine_is_not_flagged() -> None:
    """The other half — flagging these would train people to silence the check,
    and `<= max(2, machine)` is the honest portable form."""
    source = '''
def test_k_stays_within_the_machine():
    machine = available_cpus()
    assert resolve_k(64, available=64) <= max(2, machine)
'''
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "test_probe.py"
        probe.write_text(source, encoding="utf-8")
        assert _offenders_in(probe) == []
