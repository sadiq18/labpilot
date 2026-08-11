"""M11 task 4: per-branch CPU caps for generated training code.

K branches under library defaults do not get K times the throughput — they
oversubscribe the same cores, and the wall-clock can end up worse than running
the experiments sequentially, which would defeat M11's only exit criterion.
"""

from __future__ import annotations

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


def test_share_divides_the_machine_between_branches() -> None:
    assert cpu_share(4, total=16) == 4
    assert cpu_share(1, total=16) == 16


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


def test_share_is_none_and_warns_when_cpus_are_undiscoverable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Capping to 1 would serialise a possibly-large machine; say so instead.

    `None` rather than 0, so "do not cap" is never carried by a number that
    these environment variables would read as "use every core".
    """
    import logging

    with caplog.at_level(logging.WARNING):
        assert cpu_share(4, total=0) is None
    assert any("could not determine available CPUs" in r.getMessage() for r in caplog.records)


def test_an_uncapped_share_installs_nothing() -> None:
    """The None from `cpu_share` has to flow through the setter unchanged."""
    token = set_branch_cpu_share(cpu_share(4, total=0))
    try:
        assert thread_limit_env() == {}
    finally:
        reset_branch_cpu_share(token)


def test_no_cap_is_installed_by_default() -> None:
    """The sequential path must be byte-for-byte what it was before this module."""
    assert thread_limit_env() == {}


def test_installing_a_share_sets_every_thread_variable() -> None:
    token = set_branch_cpu_share(3)
    try:
        env = thread_limit_env()
        assert env == {name: "3" for name in THREAD_LIMIT_VARS}
    finally:
        reset_branch_cpu_share(token)
    assert thread_limit_env() == {}


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


def test_child_environment_carries_the_cap_and_still_drops_secrets(monkeypatch) -> None:
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


def test_the_cap_overrides_an_inherited_machine_wide_setting(monkeypatch) -> None:
    """Under fan-out the operator's own value describes the whole machine."""
    monkeypatch.setenv("OMP_NUM_THREADS", "16")

    token = set_branch_cpu_share(2)
    try:
        assert child_environment()["OMP_NUM_THREADS"] == "2"
    finally:
        reset_branch_cpu_share(token)

    # Uncapped, the operator's setting is left exactly as they set it.
    assert child_environment()["OMP_NUM_THREADS"] == "16"


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


def test_available_cpus_reports_something_usable() -> None:
    cpus = available_cpus()
    assert cpus is None or cpus >= 1


def test_a_real_subprocess_receives_the_cap() -> None:
    """The dict being right does not prove the child process sees it.

    Everything else here asserts the contents of a dict; this launches an
    actual process, which is the thing the module exists to constrain. A
    dropped `env=` at a call site, or a launcher that builds its own
    environment, would leave every other test in this file passing.
    """
    import subprocess
    import sys

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


def test_a_real_subprocess_is_uncapped_on_the_sequential_path() -> None:
    """No fan-out, no caps — the child sees whatever the operator set."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", "import os;print(os.environ.get('LOKY_MAX_CPU_COUNT'))"],
        env=child_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "None"


def test_a_negative_share_is_refused() -> None:
    """`-1` written into these variables is not an error everywhere — an
    implementation that ignores it leaves the run uncapped, which is the
    failure this module exists to prevent, arrived at silently."""
    with pytest.raises(ValueError, match="must be positive"):
        set_branch_cpu_share(-1)


def test_cap_can_be_opted_out_for_a_deterministic_environment() -> None:
    """`base` exists so a caller can build an environment predictably."""
    token = set_branch_cpu_share(2)
    try:
        assert child_environment({"PATH": "/bin"}, apply_cpu_cap=False) == {"PATH": "/bin"}
        assert "OMP_NUM_THREADS" in child_environment({"PATH": "/bin"})
    finally:
        reset_branch_cpu_share(token)
