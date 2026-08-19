"""Run the suite as if this were a different machine.

A test can be wired to the code, mutation-proof, and still assert nothing but
the box it ran on. Mutation testing cannot catch that — it answers "is this
assertion wired to that line?", never "would this assertion hold elsewhere?".
A second machine answers the second question; this plugin is the cheap local
stand-in for one.

Measured worth: on the M11 fan-out branch, four tests passed on a ten-core
laptop and failed on a two-core CI runner — a hardcoded core count, a
terminal-width-dependent `--help` assertion, and two frozen race outcomes. All
four reproduce under `FAKE_CPUS=2` or `COLUMNS=40` in about eight seconds.

Usage::

    FAKE_CPUS=1 uv run pytest -p plugins.hostile_env -q tests/unit
    COLUMNS=40  uv run pytest -q tests/unit          # no plugin needed

`tests/` is already on `pythonpath` (pyproject), so `-p plugins.hostile_env`
resolves without installation.
"""

from __future__ import annotations

import os

#: The three sources `available_cpus()` consults, in its order. Patched here —
#: at the machine — rather than by replacing `available_cpus` itself in each
#: module that imports it.
#:
#: Replacing the function stubs out the code under test. `test_compute_budget`
#: does `from ...compute_budget import available_cpus` and drives the real
#: discovery by monkeypatching these same `os` calls; with the function itself
#: swapped out, three of its assertions failed on an unmodified tree — and
#: `scripts/hostile-test.sh` reported them as "the suite is asserting the
#: machine, not the code". They were doing the opposite. Two of the four are
#: even named in `_SUBJECT_IS_THE_MACHINE` in
#: `tests/unit/test_tests_do_not_assert_the_machine.py`, which exempts them
#: because the machine *is* their subject; patching one layer down means this
#: plugin needs no exemption list of its own to agree with that one.
_CPU_SOURCES = ("process_cpu_count", "cpu_count", "sched_getaffinity")


def pytest_report_header(config) -> str | None:
    del config
    cpus = os.environ.get("FAKE_CPUS")
    columns = os.environ.get("COLUMNS")
    if not cpus and not columns:
        return None
    parts = [f"FAKE_CPUS={cpus}" if cpus else "", f"COLUMNS={columns}" if columns else ""]
    return "hostile env: " + " ".join(p for p in parts if p)


def pytest_configure(config) -> None:
    """Shrink the machine before collection, so module-level reads see it."""
    del config
    raw = os.environ.get("FAKE_CPUS")
    if not raw:
        return
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(f"FAKE_CPUS must be an integer, got {raw!r}") from exc
    if count < 1:
        raise ValueError(f"FAKE_CPUS must be >= 1, got {count}")

    # Which sources this interpreter actually has. `available_cpus` asks the
    # same `hasattr` question before calling each one, so the set it can consult
    # here is the set worth patching — and it is not the same everywhere:
    # `os.process_cpu_count` arrives in 3.13 while `pyproject.toml` supports
    # >=3.11, and `os.sched_getaffinity` is Linux-only. Requiring a fixed count
    # made the plugin raise on macOS + 3.11/3.12, where only `cpu_count` exists,
    # and both FAKE_CPUS legs died before collecting a test.
    present = [name for name in _CPU_SOURCES if hasattr(os, name)]

    # Loudly, because a plugin that silently patches nothing reports a hostile
    # run that never happened — the same failure as a mutation sweep that runs
    # no tests. The loop below cannot skip an entry, so the only thing left to
    # assert is that there was one: a `patched != len(present)` counter beside
    # it could never fire and read as a check it was not.
    if not present:
        raise RuntimeError(
            "FAKE_CPUS found none of the CPU sources `available_cpus` consults "
            f"({', '.join(_CPU_SOURCES)}); the list in hostile_env.py is stale"
        )

    for name in present:
        if name == "sched_getaffinity":
            os.sched_getaffinity = lambda _pid=0, _n=count: set(range(_n))
        else:
            setattr(os, name, lambda _n=count: _n)
