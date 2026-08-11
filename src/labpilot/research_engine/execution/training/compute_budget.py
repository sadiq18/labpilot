"""Per-branch CPU budget for generated training code (M11 task 4).

Nothing in this system caps threads today — a repo-wide search for
``n_jobs``/``num_threads``/``nthread`` finds no producer. Generated training
code therefore gets library defaults, and LightGBM and XGBoost default to
using every core. K branches under that arrangement do not get K times the
throughput: they oversubscribe the same cores, and cache thrashing plus
context-switch overhead can make the wall-clock *worse* than running the
experiments one after another — which would defeat the only exit criterion
M11 has.

The cap is a set of environment variables injected into the subprocess that
runs `train.py`. ``OMP_NUM_THREADS`` alone would not do: the realistic failure
is generated code writing ``n_jobs=-1``, which routes through joblib/loky and
reads ``LOKY_MAX_CPU_COUNT``. A **hard-coded** count (``n_jobs=8``) is covered
by neither, and that gap is accepted rather than solved.

Why environment variables rather than a cgroup, why a `ContextVar` rather than
`os.environ`, and the rest of the reasoning behind these choices:
`docs/research-os/autonomy-roadmap/design/05-parallel-branches.md` §8.

## Not wired yet, deliberately

`set_branch_cpu_share` has no production caller. M11 task 7 owns that: the
conductor computes the share once for a fan-out step and installs it around
`run_parallel_sync`, so the K worker threads inherit it through the context.
Until then `thread_limit_env()` returns nothing and the sequential (K=1) path
runs exactly as it does today.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar, Token

logger = logging.getLogger(__name__)

#: Variables that bound a thread pool the generated code might reach. Set
#: together because they govern different layers and a script can use any of
#: them: OpenMP backs LightGBM and XGBoost, the three BLAS families back
#: numpy/scipy depending on the wheel, numexpr backs `pandas.eval`, loky is
#: what scikit-learn's `n_jobs=-1` actually consults, and polars runs its own
#: rayon pool.
#:
#: **This list is curated against an open world, and that is a real limit.**
#: Generated code declares its own dependencies via PEP 723, so the set of
#: libraries is unbounded by design — `environment.py` argues at length why a
#: curated allowlist is the wrong answer for package *names*, and the same
#: objection applies here: a library whose pool is governed by a variable not
#: listed runs uncapped. The difference is the cost of being wrong. An unknown
#: package fails loudly at import; an unknown thread variable just means one
#: library ignores the budget, so the list is worth keeping accurate without
#: pretending it can ever be complete.
THREAD_LIMIT_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
)

#: The share for the branch running in this context. A `ContextVar` rather
#: than a module global because M11's branches are concurrent threads in one
#: process — `os.environ` is shared between them and could not differ per
#: branch, while a context value propagates into `anyio.to_thread.run_sync`
#: and stays per-task. Same idiom as `accessor/common/provenance.py`.
_branch_cpu_share: ContextVar[int | None] = ContextVar(
    "_branch_cpu_share", default=None
)


def available_cpus() -> int | None:
    """CPUs this process may actually use, or None if undiscoverable.

    Prefers `os.process_cpu_count()` (3.13+) and `sched_getaffinity`, both of
    which respect CPU affinity and cgroup limits. `os.cpu_count()` does not:
    in a CI container pinned to 2 cores it reports the host's count, so a
    share computed from it would license every branch to oversubscribe the
    very limit the container imposed. The project floor is 3.11, so the newer
    API is probed rather than assumed.

    Each source is tried until one gives an answer, rather than stopping at
    the first that *exists*: `os.process_cpu_count()` is documented as
    returning `int | None`, so a present-but-undetermined answer must fall
    through to the next source. Returning that `None` would leave the caller
    with no cap on a machine whose count the older call could still supply,
    and an uncapped fan-out is the failure this module exists to prevent.
    """
    for source in (
        getattr(os, "process_cpu_count", None),
        (lambda: len(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else None,
        os.cpu_count,
    ):
        if source is None:
            continue
        cpus = source()
        if cpus:
            return cpus
    return None


def cpu_share(branches: int, *, total: int | None = None) -> int | None:
    """CPUs each of `branches` concurrent branches may use, or None to not cap.

    `None` is returned in two cases, both meaning "install no cap":

    * **`branches == 1`** — the sequential path, where nothing contends, so
      there is nothing to prevent. `total` is ignored. This exists so a caller
      can compute the share unconditionally rather than special-casing K=1
      itself, and it is what keeps a non-fanned-out run's environment
      identical to what it was before this module existed.
    * **the CPU count is undiscoverable** — capping to 1 would serialise a
      machine that may be large, so the library default is preferred and the
      reason is logged.

    Never returns 0: integer division floors, so `2 // 3` is 0, and 0 does not
    mean "one thread" to these variables — it typically means "unset, use the
    default", which would hand every branch the whole machine at exactly the
    moment it is most contended. `None` is the explicit "do not cap" answer so
    that meaning is never carried by a number.
    """
    if branches < 1:
        raise ValueError(f"branches must be at least 1, got {branches}")
    if branches == 1:
        # Nothing to divide the machine with, so nothing to prevent. Returning
        # the full count instead would be honest arithmetic and the wrong
        # answer: installing it pins six variables that were previously unset,
        # and a pinned number is not the same as absent — a library that would
        # apply its own heuristic gets a hard value instead. Keeping this None
        # is what lets a caller compute the share unconditionally without
        # having to remember that K=1 is the sequential path.
        return None
    cpus = available_cpus() if total is None else total
    if cpus is None or cpus < 1:
        # Undiscoverable: capping to 1 would serialise the fan-out on a
        # machine that may be large, so prefer the library default and say so.
        logger.warning(
            "could not determine available CPUs; leaving thread limits unset, "
            "so %d concurrent branches may oversubscribe this machine",
            branches,
        )
        return None
    return max(1, cpus // branches)


def set_branch_cpu_share(cpus: int | None) -> Token:
    """Install the share for this context; returns a restoring token.

    `None` or `0` clears the cap, which is how the sequential path stays
    byte-for-byte identical to its behaviour before this module existed.

    Anything else must be a plausible count, and both directions are checked.
    A negative would be written verbatim into every variable, and these do not
    treat a negative as an error uniformly — an implementation that ignores it
    leaves the run *uncapped*, silently the oversubscription this module
    exists to prevent. A value far above the machine's own capacity is the
    same mistake mirrored: honoured rather than ignored, it produces
    thread-pool thrashing worse than no cap at all.

    `cpu_share()` cannot produce either, but this is public API and task 7
    calls it with a computed value, so the check belongs here rather than in
    the one caller that happens to be careful.
    """
    if cpus is not None and cpus < 0:
        raise ValueError(f"cpu share must be positive or None/0 to clear, got {cpus}")
    # Looked up only when there is a share to compare against — clearing a cap
    # has no use for the machine's size, and discovery is a syscall.
    ceiling = available_cpus() if cpus else None
    if cpus and ceiling and cpus > ceiling:
        # Clamped rather than refused: the share is an upper bound on threads,
        # so asking for more than the machine has is a caller's arithmetic
        # error, not a reason to abort a campaign mid-fan-out.
        logger.warning(
            "cpu share %d exceeds the %d CPUs available; clamping to %d",
            cpus,
            ceiling,
            ceiling,
        )
        cpus = ceiling
    return _branch_cpu_share.set(cpus or None)


def reset_branch_cpu_share(token: Token) -> None:
    """Restore whatever share was installed before the matching set."""
    _branch_cpu_share.reset(token)


def thread_limit_env() -> dict[str, str]:
    """Thread-cap variables for the current context, or `{}` when uncapped.

    Empty by default and empty on the sequential path, so the environment a
    training run receives is unchanged until a fan-out actually installs a
    share.
    """
    share = _branch_cpu_share.get()
    if not share:
        return {}
    return {name: str(share) for name in THREAD_LIMIT_VARS}
