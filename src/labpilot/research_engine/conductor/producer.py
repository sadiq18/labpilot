"""The evidence routine as a unit that can run on its own (M16).

Gathering evidence and testing hypotheses have opposite cost profiles and
opposite cadences, and today they share one sequential loop: when
`should_gather_evidence` says yes, the campaign stops testing and sweeps for
minutes. The gate that decides is already written and already tested; what did
not exist was any way to *call* it and the pipeline behind it without going
through the policy step this milestone exists to bypass.

That is all this module is. Deciding when to run it, and on which thread, is
the runner's job; see `design/11-background-routine.md` §5.
"""

from __future__ import annotations

import contextvars
import copy
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from labpilot.research_engine.conductor.budgets import BudgetConfig, BudgetState
from labpilot.research_engine.conductor.policy import should_gather_evidence
from labpilot.research_engine.conductor.scheduler import with_llm_client
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatherPlan:
    """Which gathering tool to invoke, and what to ask it for.

    Data, not a constant, and that is the whole point. The campaign's existing
    gathering args are shaped by Kaggle twice over — `fetch_kaggle=True` with a
    kernel-scored fetch plan, and `exclude=["papers"]`, justified by "on a
    Kaggle competition the kernels are the better-grounded source anyway".
    Off Kaggle that rationale inverts: papers and repositories may be the only
    sources there are.

    Baked into this module, those defaults would give a non-Kaggle workspace a
    fetch that soft-fails to a log warning, papers excluded, and therefore
    near-zero new evidence — while the gate, seeing a pool that never fills,
    keeps answering *gather*. A sweep every `_MIN_RESWEEP_HOURS`, forever,
    producing nothing. Run sequentially that is one bad step; run as a
    background producer it is a permanent one.

    So the producer decides *whether* and *when*, never *what*.
    `default_gather_plan` is the single place allowed to name a source.
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GatherOutcome:
    """What one tick did, and the gate's own words for why."""

    gathered: bool
    reason: str
    hypotheses_created: int = 0
    duration_s: float = 0.0


def default_gather_plan(_workspace: Workspace) -> GatherPlan:
    """Today's gathering plan: the campaign's own budget for `analyze_competition`.

    **The one domain-coupled site in this milestone.** Beyond Kaggle
    ([M12](../../../../docs/research-os/autonomy-roadmap/06-beyond-kaggle.md))
    this function changes and nothing else here does. If a second domain
    arrives and any other part of the producer needs editing, the boundary was
    drawn in the wrong place.

    Takes the workspace so that resolving per-workspace later is a change of
    body, not of every call site.
    """
    from labpilot.research_engine.conductor.actions import ANALYZE_ARGS

    # Deep, not `dict(...)`: `ANALYZE_ARGS["exclude"]` is a list, and a shallow
    # copy shares it with the module constant. A handler appending one analyzer
    # to skip would edit what every later campaign step and producer tick
    # gathers, for the life of the process.
    return GatherPlan(tool="analyze_competition", args=copy.deepcopy(ANALYZE_ARGS))


def _created_count(result: Any) -> int:
    """How many hypotheses the sweep added, best-effort.

    Read from the tool's own summary rather than by diffing the pool: the
    consumer claims hypotheses while this runs, and a claim moves a row out of
    `proposed`, so a before/after count would report the producer's work minus
    the consumer's progress. Zero when the tool reports nothing — this is an
    observability number, and guessing one is worse than admitting to none.
    """
    data = getattr(result, "data", None)
    if not isinstance(data, dict):
        return 0
    report = data.get("report")
    summary = getattr(report, "summary", None)
    if not isinstance(summary, dict):
        return 0
    try:
        return int(summary.get("hypothesis_count") or 0)
    except (TypeError, ValueError):
        return 0


#: Role the sweep's expensive work runs under — hypothesis generation and
#: extraction are both `reasoning` calls.
_PRODUCER_ROLE = "reasoning"


#: Clients whose `preview` has already raised. A signature mismatch does not
#: heal, and a traceback every tick for twelve hours buries the campaign's own
#: log without telling the operator anything the first one did not.
_PREVIEW_REFUSED: set[int] = set()


def _budget_headroom(llm_client: Any | None, reserve: float) -> str | None:
    """Reason to skip when the producer would be spending the consumer's quota.

    The producer is the *lower* claim on the LLM budget: a sweep is minutes of
    reasoning calls, and finishing one at the cost of the campaign's next plan
    is a bad trade in both directions — the campaign stalls and the evidence
    lands with nothing to test it.

    Asked once, before the sweep, rather than enforced per call inside it. The
    decision that matters is whether to *start*: a sweep abandoned halfway has
    already spent the quota it was meant to protect.

    Returns None (go ahead) when there is no reserve, no client, or no router
    behind the client — a legacy provider path has no ledger to ask, and
    refusing to gather on that basis would be inventing a limit rather than
    respecting one.
    """
    if reserve <= 0.0 or llm_client is None:
        return None
    preview = getattr(llm_client, "preview", None)
    if not callable(preview):
        return None
    if id(llm_client) in _PREVIEW_REFUSED:
        return None
    try:
        route = preview(_PRODUCER_ROLE, reserve=reserve)
    except Exception:
        _PREVIEW_REFUSED.add(id(llm_client))
        logger.exception(
            "Evidence producer could not preview its route; gathering without a "
            "budget reserve for the rest of this run"
        )
        return None
    if getattr(route, "provider", None) is not None:
        return None
    detail = str(getattr(route, "reason", "") or "no provider available")
    return f"holding {reserve:.0%} of the LLM budget for the campaign ({detail})"


def gather_once(
    workspace: Workspace,
    registry: ToolRegistry,
    plan: GatherPlan,
    *,
    llm_client: Any | None = None,
    budgets: tuple[BudgetConfig, BudgetState] | None = None,
    reserve: float = 0.0,
) -> GatherOutcome:
    """Gather evidence if the gate allows, and report what happened either way.

    The gate is `should_gather_evidence` unchanged — one predicate, one place.
    It moves from deciding the consumer's tool allowlist to deciding this
    tick, and keeps its blind spots and its tests along with it.

    Invoked through the registry rather than by reaching into the analyze
    orchestrator, so the tool's verification gate and its report write stay in
    the path. `llm_client` reaches the handler by the same signature-driven
    rule the Conductor's scheduler uses: a gathering pipeline running without
    one degrades silently, which is a failure this repository has already paid
    for once.

    ``reserve`` is the fraction of the LLM budget this call refuses to touch,
    so the producer runs out before the campaign does. Default 0.0 keeps a
    direct caller (and every test) on the real limits.

    Raises whatever the tool raises. Isolating a bad tick is the runner's job,
    and a unit that swallows its own failures cannot be tested for them.
    """
    ok, reason = should_gather_evidence(workspace, budgets)
    if not ok:
        logger.info("Evidence producer: skipping — %s", reason)
        return GatherOutcome(gathered=False, reason=reason)

    yielded = _budget_headroom(llm_client, reserve)
    if yielded is not None:
        logger.info("Evidence producer: skipping — %s", yielded)
        return GatherOutcome(gathered=False, reason=yielded)

    logger.info("Evidence producer: gathering — %s", reason)
    started = time.monotonic()
    args = with_llm_client(registry, plan.tool, dict(plan.args), llm_client)
    result = registry.invoke(plan.tool, workspace, **args)
    duration = time.monotonic() - started
    created = _created_count(result)
    logger.info(
        "Evidence producer: %s finished in %.1fs, %d hypothesis(es) added",
        plan.tool,
        duration,
        created,
    )
    return GatherOutcome(
        gathered=True,
        reason=reason,
        hypotheses_created=created,
        duration_s=duration,
    )


#: Seconds between ticks. A **rate limit, not a cadence assumption**: a domain
#: whose evidence moves weekly rather than hourly needs no different runner, it
#: just gets more no-ops, and a no-op is one freshness read plus one pool scan.
_TICK_SECONDS = float(os.environ.get("LABPILOT_GATHER_TICK_S", "300"))

#: Floor under the tick. `Event.wait(0)` returns immediately, so a zero or
#: negative interval turns the loop into a spin that globs the hypothesis
#: directory and opens two SQLite connections as fast as it can — burning a
#: core next to a training job, with the re-sweep floor hiding the symptom by
#: making every tick a no-op.
_MIN_TICK_SECONDS = 1.0

#: Fraction of the LLM budget the producer leaves alone. Hypothesis generation
#: is a `reasoning` call and a sweep makes many of them; a background worker
#: running hot exhausts the free tier the campaign needs to plan.
_RESERVE = float(os.environ.get("LABPILOT_GATHER_RESERVE", "0.2"))

#: How long campaign shutdown waits for a tick already in flight. A sweep runs
#: for minutes and an operator must not wait it out; past this the thread is
#: left daemon and the process exits.
_STOP_GRACE_S = float(os.environ.get("LABPILOT_GATHER_STOP_GRACE_S", "5"))


class EvidenceProducer:
    """Runs `gather_once` on its own thread, so the campaign never waits on it.

    A thread rather than a bus subscriber or a second process:

    * `EventBus.publish` is a synchronous `signal.send`, so an
      `ExperimentCompleted` handler would run on the *publisher's* thread —
      blocking the consumer at exactly the moment this exists to unblock it.
    * A separate process outlives the campaign, which means orphan detection,
      its own config and LLM resolution, and sweeps against a workspace nobody
      is using. Correct eventually; not this milestone.

    Nothing live is shared across the thread boundary. `SqliteClient` opens
    `check_same_thread=True` and `ConductorStore` does not opt out, so each tick
    opens its own handles (~1.7ms warm) and closes them. The budget pair is
    re-read for the same reason it is not passed in: `BudgetState` is mutated by
    the campaign on every recorded experiment, and sharing the instance would
    have the gate judge a verdict assembled from two different campaign states.
    One tick of staleness is fine — "has this campaign stopped improving" does
    not change meaningfully inside five minutes.
    """

    def __init__(
        self,
        workspace: Workspace,
        registry: ToolRegistry,
        *,
        session_id: str | None = None,
        llm_client: Any | None = None,
        plan: GatherPlan | None = None,
        tick_seconds: float | None = None,
        reserve: float | None = None,
    ) -> None:
        self.workspace = workspace
        self.registry = registry
        self.session_id = session_id
        self.llm_client = llm_client
        self.plan = plan or default_gather_plan(workspace)
        requested = _TICK_SECONDS if tick_seconds is None else tick_seconds
        self.tick_seconds = max(_MIN_TICK_SECONDS, requested)
        if requested < _MIN_TICK_SECONDS:
            logger.warning(
                "Evidence producer tick %.3fs is below the %.0fs floor; using the floor",
                requested,
                _MIN_TICK_SECONDS,
            )
        self.reserve = _RESERVE if reserve is None else reserve
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ticks = 0
        self._last: GatherOutcome | None = None
        self._last_error: str | None = None

    # -- state the policy can see (design §9) --------------------------------

    def status(self) -> dict[str, Any]:
        """What the producer last did, for the observe bundle.

        Description, not control. The gate decides; this is here so the policy
        is not watching the pool change between steps with no account of why.
        """
        with self._lock:
            last, ticks, error = self._last, self._ticks, self._last_error
        return {
            "running": self.is_running(),
            "ticks": ticks,
            "last_decision": None if last is None else ("gathered" if last.gathered else "skipped"),
            "last_reason": None if last is None else last.reason,
            "last_hypotheses_created": None if last is None else last.hypotheses_created,
            "last_error": error,
        }

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- one tick, callable without a thread ---------------------------------

    def tick_once(self) -> GatherOutcome | None:
        """Evaluate the gate and gather if it allows. `None` when the tick failed.

        Never raises. A producer that can take the campaign down with it is
        worse than one that gathers nothing: the consumer's work is the thing
        with a deadline.
        """
        try:
            outcome = gather_once(
                self.workspace,
                self.registry,
                self.plan,
                llm_client=self.llm_client,
                budgets=self._budgets(),
                reserve=self.reserve,
            )
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.exception("Evidence producer tick failed")
            with self._lock:
                self._ticks += 1
                # Cleared, not kept: leaving the previous outcome in place made
                # `status()` report `last_decision="gathered"` beside a fresh
                # `last_error`, describing a sweep that did not happen on the
                # tick being reported — into the observe bundle the policy reads.
                self._last = None
                self._last_error = str(exc)
            return None
        with self._lock:
            self._ticks += 1
            self._last = outcome
            self._last_error = None
        return outcome

    def _budgets(self) -> tuple[BudgetConfig, BudgetState] | None:
        """This campaign's budget pair, read fresh, or None when unavailable.

        Without it the gate simply loses its stagnant clause — the pool and
        freshness clauses still decide — so a failure here degrades the
        producer's judgement rather than stopping it.
        """
        if not self.session_id:
            return None
        from labpilot.research_engine.conductor.budgets import load_budget_pair
        from labpilot.research_engine.conductor.store import ConductorStore

        store = None
        try:
            store = ConductorStore(self.workspace.knowledge_dir, self.workspace.competition)
            session = store.get_session(self.session_id)
            if session is None:
                return None
            return load_budget_pair(session)
        except Exception:
            logger.exception("Evidence producer could not read budgets; gating without them")
            return None
        finally:
            if store is not None:
                store.close()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start ticking. A no-op while a previous thread is still alive.

        Refused rather than stacked: `self._stop.clear()` below would un-stop a
        loop that was told to finish, so a second `start()` would leave two
        threads sweeping the same workspace — the double-gathering this design
        exists to prevent.
        """
        if self.is_running():
            logger.debug("Evidence producer already running; not starting a second")
            return
        self._stop.clear()
        # The provenance sink lives in a `ContextVar`, and a fresh thread gets a
        # fresh empty context — so without copying, every micro-agent call the
        # sweep makes records nothing in `agent_invocations`. That table is what
        # M14 2b/3 are blocked on, and this is the fourth time in this codebase
        # that thread-crossing has silently emptied an instrument (see
        # `SqliteInvocationSink`, which names the first three).
        context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=lambda: context.run(self._loop),
            name="labpilot-evidence-producer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Evidence producer started (tick %.0fs, tool %s)", self.tick_seconds, self.plan.tool
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.tick_once()
            # `wait`, not `sleep`: an idle producer must not hold shutdown for
            # the rest of its interval.
            if self._stop.wait(self.tick_seconds):
                return

    def stop(self, timeout: float | None = None) -> bool:
        """Signal the thread and wait, bounded. True when it actually finished.

        A thread that outlived the timeout is **kept**, not dropped. Clearing
        the handle would report `is_running() is False` while a sweep was still
        writing to the workspace, and would let a later `start()` clear the stop
        event out from under it and run two producers at once.
        """
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(_STOP_GRACE_S if timeout is None else timeout)
        if thread.is_alive():
            logger.info(
                "Evidence producer still sweeping at shutdown; leaving it to the process exit"
            )
            return False
        self._thread = None
        return True

    def __enter__(self) -> EvidenceProducer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
