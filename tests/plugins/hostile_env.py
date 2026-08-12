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

#: Modules that read the CPU count. Patched together, because a test that
#: resolves K through one and asserts through the other would otherwise see two
#: different machines in the same run.
_CPU_READERS = (
    "labpilot.research_engine.execution.training.compute_budget",
    "labpilot.research_engine.conductor.fanout",
)


def pytest_report_header(config) -> str | None:
    del config
    cpus = os.environ.get("FAKE_CPUS")
    columns = os.environ.get("COLUMNS")
    if not cpus and not columns:
        return None
    parts = [f"FAKE_CPUS={cpus}" if cpus else "", f"COLUMNS={columns}" if columns else ""]
    return "hostile env: " + " ".join(p for p in parts if p)


def pytest_configure(config) -> None:
    """Install the fake CPU count before collection, so module-level reads see it."""
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

    import importlib

    patched = 0
    for name in _CPU_READERS:
        module = importlib.import_module(name)
        if hasattr(module, "available_cpus"):
            module.available_cpus = lambda _count=count: _count
            patched += 1
    # Loudly, because a plugin that silently patches nothing reports a hostile
    # run that never happened — the same failure as a mutation sweep that runs
    # no tests.
    if patched != len(_CPU_READERS):
        raise RuntimeError(
            f"FAKE_CPUS patched {patched} of {len(_CPU_READERS)} readers; "
            "the module list in hostile_env.py is stale"
        )
