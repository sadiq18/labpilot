"""M11 task 4: per-branch CPU caps for generated training code.

K branches under library defaults do not get K times the throughput — they
oversubscribe the same cores, and the wall-clock can end up worse than running
the experiments sequentially, which would defeat M11's only exit criterion.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading

import pytest

from labpilot.research_engine.execution.training.compute_budget import (
    THREAD_LIMIT_VARS,
    available_cpus,
    cpu_share,
    reset_branch_cpu_share,
    set_branch_cpu_share,
    thread_limit_env,
)
from labpilot.research_engine.execution.training.environment import child_environment

# --- discovering how many CPUs we may use ---------------------------------


def test_available_cpus_reports_something_usable() -> None:
    cpus = available_cpus()
    assert cpus is None or cpus >= 1


def test_discovery_prefers_the_affinity_aware_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A container pinned to 2 cores must not be told it has the host's 64."""

    monkeypatch.setattr(os, "process_cpu_count", lambda: 2, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)

    assert available_cpus() == 2


def test_discovery_falls_through_when_the_preferred_source_answers_none(
    monkeypatch,
) -> None:
    """`os.process_cpu_count()` returns `int | None`, so present != useful.

    Stopping at the first source that *exists* rather than the first that
    *answers* left the caller uncapped on a machine whose count `os.cpu_count()`
    could still supply — and an uncapped fan-out is the failure this module
    exists to prevent.
    """

    monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 12)
    if hasattr(os, "sched_getaffinity"):
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set())

    assert available_cpus() == 12


def test_discovery_returns_none_only_when_every_source_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    if hasattr(os, "sched_getaffinity"):
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set())

    assert available_cpus() is None


# --- dividing them between branches ---------------------------------------


def test_share_divides_the_machine_between_branches() -> None:
    assert cpu_share(4, total=16) == 4
    assert cpu_share(2, total=16) == 8


def test_share_never_falls_to_zero() -> None:
    """`2 // 3` is 0, and 0 means 'unset, use every core' to these variables.

    That would hand each branch the whole machine at exactly the moment it is
    most contended — the opposite of the cap's purpose.
    """
    assert cpu_share(3, total=2) == 1
    assert cpu_share(64, total=1) == 1


def test_share_rejects_a_nonsensical_branch_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        cpu_share(0, total=8)


def test_a_single_branch_is_not_capped_at_all() -> None:
    """K=1 is the sequential path: no contention, so nothing to prevent.

    Task 7's natural implementation computes the share once per step and
    installs it without special-casing K=1, so this is what keeps a
    non-fanned-out run's environment identical to what it is today.
    """
    assert cpu_share(1) is None
    assert cpu_share(1, total=16) is None

    token = set_branch_cpu_share(cpu_share(1))
    try:
        assert thread_limit_env() == {}
        assert child_environment({"PATH": "/bin"}) == {"PATH": "/bin"}
    finally:
        reset_branch_cpu_share(token)


def test_share_is_none_and_warns_when_cpus_are_undiscoverable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Capping to 1 would serialise a possibly-large machine; say so instead.

    `None` rather than 0, so "do not cap" is never carried by a number that
    these environment variables would read as "use every core".
    """

    with caplog.at_level(logging.WARNING):
        assert cpu_share(4, total=0) is None
    assert any("could not determine available CPUs" in r.getMessage() for r in caplog.records)


def test_a_failed_discovery_leaves_the_run_uncapped_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The real discovery-failure path, not the `total=0` shortcut."""

    monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    if hasattr(os, "sched_getaffinity"):
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set())

    with caplog.at_level(logging.WARNING):
        assert cpu_share(4) is None
    assert any("could not determine available CPUs" in r.getMessage() for r in caplog.records)


# --- installing a share on a context --------------------------------------


def test_no_cap_is_installed_by_default() -> None:
    """The sequential path must be byte-for-byte what it was before this module."""
    assert thread_limit_env() == {}


def test_an_uncapped_share_installs_nothing() -> None:
    """The None from `cpu_share` has to flow through the setter unchanged."""
    token = set_branch_cpu_share(cpu_share(4, total=0))
    try:
        assert thread_limit_env() == {}
    finally:
        reset_branch_cpu_share(token)


def test_installing_a_share_sets_every_thread_variable() -> None:
    token = set_branch_cpu_share(3)
    try:
        env = thread_limit_env()
        assert env == {name: "3" for name in THREAD_LIMIT_VARS}
    finally:
        reset_branch_cpu_share(token)
    assert thread_limit_env() == {}


def test_a_negative_share_is_refused() -> None:
    """`-1` written into these variables is not an error everywhere — an
    implementation that ignores it leaves the run uncapped, which is the
    failure this module exists to prevent, arrived at silently."""
    with pytest.raises(ValueError, match="must be positive"):
        set_branch_cpu_share(-1)


def test_a_share_larger_than_the_machine_is_clamped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The mirror of the negative case, and honoured rather than ignored.

    A five-digit thread count is not rejected by OpenMP or the BLAS families —
    it is obeyed, producing pool thrashing worse than the uncapped default
    this module exists to improve on.
    """

    ceiling = available_cpus()
    assert ceiling, "this test needs a discoverable CPU count"

    with caplog.at_level(logging.WARNING):
        token = set_branch_cpu_share(ceiling * 1000)
    try:
        assert thread_limit_env()["OMP_NUM_THREADS"] == str(ceiling)
    finally:
        reset_branch_cpu_share(token)
    assert any("clamping to" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_a_share_within_the_machine_is_left_alone(offset: int) -> None:
    """Clamping must not quietly rewrite a legitimate share.

    Parametrised across the values *at* and just below the boundary, not just
    1: a clamp written as `min(cpus, ceiling // 2)`, or firing at `>=` rather
    than `>`, rewrites every realistic mid-range share while leaving 1 alone.
    """
    ceiling = available_cpus()
    assert ceiling, "this test needs a discoverable CPU count"
    share = max(1, ceiling - offset)

    token = set_branch_cpu_share(share)
    try:
        assert thread_limit_env()["OMP_NUM_THREADS"] == str(share)
    finally:
        reset_branch_cpu_share(token)


def test_each_branch_thread_sees_its_own_share() -> None:
    """A ContextVar, not os.environ: branches are threads in one process.

    os.environ is shared, so a per-branch cap could not be expressed with it —
    the last branch to write would set the value for all of them.
    """
    seen: dict[int, str] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def branch(i: int) -> None:
        token = set_branch_cpu_share(i + 1)
        try:
            barrier.wait()  # every branch has installed its own share by now
            with lock:
                seen[i] = thread_limit_env()["OMP_NUM_THREADS"]
        finally:
            reset_branch_cpu_share(token)

    threads = [threading.Thread(target=branch, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {0: "1", 1: "2", 2: "3", 3: "4"}


# --- which variables, and why each one earns its place --------------------


def test_loky_is_capped_because_n_jobs_minus_one_reads_it() -> None:
    """`OMP_NUM_THREADS` alone would miss the realistic failure.

    Generated code writing `n_jobs=-1` routes through joblib/loky, which
    consults `LOKY_MAX_CPU_COUNT` and ignores the OpenMP variable.
    """
    token = set_branch_cpu_share(2)
    try:
        env = thread_limit_env()
        assert env["LOKY_MAX_CPU_COUNT"] == "2"
        assert env["OMP_NUM_THREADS"] == "2"
        # Apple Accelerate backs numpy on the common dev platform.
        assert env["VECLIB_MAXIMUM_THREADS"] == "2"
    finally:
        reset_branch_cpu_share(token)


def test_polars_thread_pool_is_capped_too() -> None:
    """Generated code picks its own dependencies; polars is a plausible one.

    The list cannot be complete against a PEP 723 open world, but a library
    this likely for tabular work running a full-machine pool per branch would
    oversubscribe exactly as if there were no cap.
    """
    token = set_branch_cpu_share(2)
    try:
        env = thread_limit_env()
        assert env["POLARS_MAX_THREADS"] == "2"
        assert env["RAYON_NUM_THREADS"] == "2"
    finally:
        reset_branch_cpu_share(token)


# --- reaching the child process -------------------------------------------


def test_child_environment_carries_the_cap_and_still_drops_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap rides the one function every generated-code launcher uses."""
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("PATH", "/bin")

    token = set_branch_cpu_share(2)
    try:
        env = child_environment()
    finally:
        reset_branch_cpu_share(token)

    assert env["OMP_NUM_THREADS"] == "2"
    assert env["LOKY_MAX_CPU_COUNT"] == "2"
    assert "GROQ_API_KEY" not in env
    assert env["PATH"] == "/bin"


def test_the_cap_overrides_an_inherited_machine_wide_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under fan-out the operator's own value describes the whole machine."""
    monkeypatch.setenv("OMP_NUM_THREADS", "16")

    token = set_branch_cpu_share(2)
    try:
        assert child_environment()["OMP_NUM_THREADS"] == "2"
    finally:
        reset_branch_cpu_share(token)

    # Uncapped, the operator's setting is left exactly as they set it.
    assert child_environment()["OMP_NUM_THREADS"] == "16"


def test_cap_can_be_opted_out_for_a_deterministic_environment() -> None:
    """`base` exists so a caller can build an environment predictably."""
    token = set_branch_cpu_share(2)
    try:
        assert child_environment({"PATH": "/bin"}, apply_cpu_cap=False) == {"PATH": "/bin"}
        assert "OMP_NUM_THREADS" in child_environment({"PATH": "/bin"})
    finally:
        reset_branch_cpu_share(token)


def test_a_real_subprocess_receives_the_cap() -> None:
    """The dict being right does not prove the child process sees it.

    Everything else here asserts the contents of a dict; this launches an
    actual process, which is the thing the module exists to constrain. A
    dropped `env=` at a call site, or a launcher that builds its own
    environment, would leave every other test in this file passing.
    """

    token = set_branch_cpu_share(3)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os;print(os.environ.get('OMP_NUM_THREADS'),"
                "os.environ.get('LOKY_MAX_CPU_COUNT'))",
            ],
            env=child_environment(),
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        reset_branch_cpu_share(token)

    assert proc.stdout.split() == ["3", "3"]


def test_a_real_subprocess_is_uncapped_on_the_sequential_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fan-out, no caps — the child sees whatever the operator set.

    The variable is cleared first because `child_environment()` inherits
    `os.environ` and `LOKY_MAX_CPU_COUNT` is not a secret, so a developer who
    exports it — the documented way to silence loky's core-count warning —
    would otherwise see this fail and read it as a bug in the module rather
    than an assumption in the test.
    """

    monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
    proc = subprocess.run(
        [sys.executable, "-c", "import os;print(os.environ.get('LOKY_MAX_CPU_COUNT'))"],
        env=child_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "None"
