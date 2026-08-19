"""Campaign budgets and automatic stop evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from labpilot.research_engine.conductor.models import _now
from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
    MEASUREMENT_PREFIXES,
)

StopReason = Literal[
    "none",
    "submission_budget",
    "wall_time",
    "cost_budget",
    "metric_target",
    "plateau",
    "operator_pause",
    "policy_stop",
    "max_steps",
    "failing",
]

#: Consecutive failed executions before a campaign is stopped.
#:
#: Every stop condition above answers "am I finished?" — none answered "is any
#: of this working?". Measured on rogii 2026-08-08: **108 consecutive failures**,
#: each ~33 ms, each the identical `ModuleNotFoundError`, across four sessions
#: and 80 conductor steps. Nothing stopped, and `tasks_failed` read 0 throughout,
#: so the campaign looked healthy the entire time.
#:
#: Three is deliberately low. A transient failure is worth retrying; the third
#: identical one is a pattern, and every step after it is spent. Raising this
#: costs steps linearly and buys nothing — the 109th failure taught us no more
#: than the 3rd.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3

#: Steps allowed with no successful experiment at all before stopping.
#:
#: The slower sibling of the above, for a campaign that never reaches an
#: execution: rogii's S-021 spent 30 steps choosing `run_experiment` while
#: `plateau` could not fire (it needs metrics) and `metric_target` could not
#: fire (it needs a metric). A campaign that has produced nothing after this
#: many steps is not mid-flight.
DEFAULT_MAX_BARREN_STEPS = 8


class BudgetConfig(BaseModel):
    """Resource and objective limits for a campaign session."""

    max_submissions: int | None = None
    max_wall_s: float | None = None
    max_cost_usd: float | None = None
    target_metric: str | None = None
    target_value: float | None = None
    maximize: bool = True
    plateau_window: int = 3
    plateau_epsilon: float = 1e-6
    #: `None` disables the breaker. Opt-out exists because a campaign
    #: deliberately probing a broken workspace is a legitimate thing to run.
    max_consecutive_failures: int | None = DEFAULT_MAX_CONSECUTIVE_FAILURES
    max_barren_steps: int | None = DEFAULT_MAX_BARREN_STEPS


class ScoreEvent(BaseModel):
    """One comparable score, appended once per successful experiment.

    See docs/research-os/autonomy-roadmap/design/02-objective-loop.md §3 for
    why `metric_name` is the resolved key rather than the raw one, and why
    `maximize` travels with the value.
    """

    experiment_id: str
    hypothesis_id: str | None = None
    technique: str | None = None
    #: Techniques applied together, not cumulative lineage — the `Hypothesis`
    #: distinction is `combo_techniques` vs `technique_stack`.
    combo_techniques: list[str] = Field(default_factory=list)
    #: Resolved metric key (e.g. ``cv_rmse``), not the raw `metrics.json` key.
    metric_name: str
    #: Rejects NaN/inf. A diverged run's NaN is not a comparable score: it
    #: serializes as a bare `NaN` token (invalid JSON) into the session blob,
    #: and every NaN comparison is False, so it silently disables the
    #: `plateau` and `metric_target` stops that read this series. A writer
    #: that finds a non-finite metric must skip the event, as it already
    #: skips a placeholder run.
    value: float = Field(allow_inf_nan=False)
    maximize: bool
    #: Defaults to the same UTC-aware format every other timestamp in this
    #: package uses, so callers cannot supply a naive one that later fails to
    #: compare against the rest of the series.
    timestamp: str = Field(default_factory=_now)


class BudgetState(BaseModel):
    """Live counters persisted in session metadata / metrics table."""

    submissions: int = 0
    llm_cost_usd: float = 0.0
    wall_started_at: str | None = None
    metric_history: list[float] = Field(default_factory=list)
    last_metric: float | None = None
    #: The comparable score series (M8). No writer yet — the conductor loop
    #: appends one event per successful experiment in M8-2, which also derives
    #: `metric_history`/`last_metric` from it. Until then those two stay as
    #: they are: read by `evaluate_stops`, written by nothing.
    score_events: list[ScoreEvent] = Field(default_factory=list)
    #: Set once the metric-name mismatch has been reported for this campaign.
    #: On the state rather than in a module-level set, for the same reason
    #: `stagnation_mint_fired` below is: a process-wide latch reports the first
    #: campaign and silently skips every one after it, and the operator who has
    #: not yet met the failure mode is exactly the one running the second.
    metric_mismatch_reported: bool = False
    #: Set when the M8-6 stagnation mint fires for the current plateau, so a
    #: long plateau doesn't mint a near-duplicate hypothesis every step.
    #: Cleared on the next improvement. Read by `_maybe_mint_on_stagnation`
    #: in conductor/loop.py.
    stagnation_mint_fired: bool = False
    #: Reset by any execution that succeeds, so a campaign that recovers is not
    #: punished for the failures it climbed out of.
    consecutive_failures: int = 0
    #: Steps taken since the last successful experiment. Distinct from
    #: `consecutive_failures`: a campaign can burn steps without ever reaching
    #: an execution, which is how 30 steps passed with nothing to count.
    steps_since_success: int = 0
    #: What the failures said, most recent last. Bounded — this is a stop
    #: *reason*, not a log, and it is written into session metadata.
    recent_failures: list[str] = Field(default_factory=list)

    def record_execution(self, *, succeeded: bool, error: str = "") -> None:
        """Fold one execution outcome into the breaker's counters."""
        if succeeded:
            self.consecutive_failures = 0
            self.steps_since_success = 0
            self.recent_failures = []
            return
        self.consecutive_failures += 1
        excerpt = " ".join(str(error).split())[:200]
        if excerpt:
            self.recent_failures = [*self.recent_failures[-2:], excerpt]

    def ensure_wall_start(self) -> None:
        if not self.wall_started_at:
            self.wall_started_at = datetime.now(UTC).isoformat()

    def elapsed_s(self, *, now: datetime | None = None) -> float:
        if not self.wall_started_at:
            return 0.0
        start = datetime.fromisoformat(self.wall_started_at)
        current = now or datetime.now(UTC)
        return max(0.0, (current - start).total_seconds())


#: Imported rather than restated. Two copies of "how a metric was measured"
#: drift, and a prefix present in one and missing from the other means
#: `--target-metric rmse` silently stops matching a recorded `val_rmse`.


def metric_names_match(recorded: str | None, requested: str | None) -> bool:
    """Whether a recorded metric key answers a request for `requested`.

    Lives beside `ScoreEvent` because it interprets that model's
    `metric_name`, and is public because the conductor needs it too.

    The two names come from different places and are spelled differently on
    purpose: a `ScoreEvent` carries the resolver's key (`cv_rmse`), while
    `--target-metric` takes the competition's own metric (`rmse`, from
    `MetricSpec.key`) — the only spelling a user has. Requiring equality means
    the target never matches and the stop never fires.

    An unqualified request matches any measurement of that metric, because the
    user naming `rmse` cannot be asking for a particular one. A *qualified*
    request is taken literally: `lb_auc` is not answered by a `cv_auc`
    reading, since local and leaderboard scores are the distinction the
    milestone keeps separate rather than a spelling difference.
    """
    left = str(recorded or "").strip().lower()
    right = str(requested or "").strip().lower()
    if not left or not right:
        return False
    if left == right:
        return True
    # Only an unqualified request can widen: a request that already names a
    # measurement can never equal `prefix + itself` unless the recorded key is
    # doubly prefixed, so no separate guard for it is reachable.
    return any(left == f"{prefix}{right}" for prefix in MEASUREMENT_PREFIXES)


def comparable_tail(events: list[ScoreEvent]) -> list[ScoreEvent]:
    """The trailing run of events measuring the same metric as the newest one.

    The series can legitimately change metric mid-campaign — `analyze` can
    correct which key is primary. Readings either side of that change are on
    different scales, so `plateau` must not take a max-minus-min across them.

    Narrowing the *derived view* rather than deleting the events is what keeps
    both guarantees: the comparison stays honest, and every experiment id
    remains citable, which exit criterion 1 and the stagnation mint both
    depend on. Truncating the series instead would break exactly the
    citation the design doc refused to sacrifice.
    """
    if not events:
        return []
    newest = events[-1].metric_name
    tail: list[ScoreEvent] = []
    for event in reversed(events):
        if not metric_names_match(event.metric_name, newest):
            break
        tail.append(event)
    return list(reversed(tail))


class ScoreSummary(BaseModel):
    """What the score series says about progress, in the terms a decision needs.

    Derived, never stored: computed from `score_events` on demand so it cannot
    drift from the series the way `metric_history` did before it had a writer.
    """

    #: The best value seen, read in the series' own direction. `None` until a
    #: comparable score exists.
    best_so_far: float | None = None
    #: Most recent last, so the tail reads in the order it happened.
    last_3_scores: list[float] = Field(default_factory=list)
    #: How far the latest reading sits from the best, in the metric's own
    #: direction: `0.0` at a record, negative behind one, never positive —
    #: `best_so_far` includes the latest, so it can at most tie. Whether the
    #: latest run *improved* is `steps_since_improvement == 0`, which is why
    #: this measures distance instead of repeating that.
    delta_vs_best: float | None = None
    #: Completed experiments since one last improved on everything before it —
    #: not conductor steps. A campaign that spends ten steps reflecting between
    #: two experiments has taken one, not eleven.
    steps_since_improvement: int = 0
    #: The metric these numbers are readings of, so a consumer cannot compare
    #: them against a threshold for something else.
    metric_name: str | None = None


def score_summary(state: BudgetState, config: BudgetConfig) -> ScoreSummary:
    """Summarise the comparable score series.

    Takes the same two arguments as `goal_progress(config, state)` — the
    shape M17's plan records as validated by a prototype — so that milestone
    renders its progress line by calling this rather than deriving the same
    four numbers a second way, which is how the primary-metric key ended up
    with four disagreeing resolvers. (The order differs; only the pair
    matters, since `goal_progress` will call this from inside itself.)

    Only the comparable tail counts. Readings either side of a metric change
    are on different scales, so a "best" across them would compare an RMSE to
    an accuracy — the defect this milestone exists to prevent.
    """
    events = comparable_tail(state.score_events)
    if not events:
        return ScoreSummary()

    values = [event.value for event in events]
    # Direction is a property of the metric, not of an individual reading, so
    # the newest event decides for the whole window. Flags within one metric
    # can genuinely disagree: M8-2 falls back to the campaign's configured
    # direction when the competition profile cannot answer, so an early
    # experiment may carry a guess and a later one — after `analyze` writes
    # the spec — the resolved answer. The later reading is the better-informed
    # one, and re-splitting the window by flag would fragment a series that
    # measures a single thing.
    maximize = events[-1].maximize
    best = max(values) if maximize else min(values)
    latest = values[-1]

    # Distance behind the record, expressed the same way in both directions so
    # a consumer never re-derives the sign from the metric — that
    # re-derivation is exactly what `ScoreEvent.maximize` exists to prevent.
    delta = latest - best if maximize else best - latest

    return ScoreSummary(
        best_so_far=best,
        last_3_scores=values[-3:],
        delta_vs_best=delta,
        steps_since_improvement=_steps_since_improvement(values, maximize, config.plateau_epsilon),
        metric_name=events[-1].metric_name,
    )


def _steps_since_improvement(values: list[float], maximize: bool, epsilon: float) -> int:
    """Experiments since one beat everything before it by more than `epsilon`.

    Measured against the best of the *preceding* readings, not the running
    best including itself — otherwise every event trivially ties its own best
    and nothing ever counts as an improvement. `epsilon` is the same
    noise floor `plateau` uses, so the two agree about what "no change" means.

    That epsilon is **absolute**, and must be set to the metric's scale. It
    was harmless while only `plateau` read it — that stop needs near-exact
    ties and has fired on essentially nothing — but this drives the gathering
    gate and the policy's view of progress, so a default of 1e-6 against a
    metric whose values live near or below it swallows every real gain: the
    campaign reads as permanently stagnant while improving on every run.

    One pass, carrying the best rather than re-scanning the prefix: this runs
    in the observe bundle and again in the gathering gate, so it is paid at
    least twice per conductor step against a series the campaign is designed
    to grow.
    """
    if not values:
        return 0
    improved_at = 0
    best_before = values[0]
    for index in range(1, len(values)):
        value = values[index]
        gain = value - best_before if maximize else best_before - value
        if gain > epsilon:
            improved_at = index
        best_before = max(best_before, value) if maximize else min(best_before, value)
    return len(values) - 1 - improved_at


def _last_metric_matches_target(config: BudgetConfig, state: BudgetState) -> bool:
    """Whether `last_metric` is a reading of the metric the target names.

    `last_metric` is a bare number; `target_value` is a threshold for a
    specific metric. Comparing them without checking which metric produced the
    number lets a `cv_rmse` of 190.97 satisfy an `lb_auc` target of 0.90 and
    end the campaign on a metric it was never measuring.

    Only enforced when the series says what the metric was. A session whose
    `last_metric` predates `score_events` has no key to check, and refusing to
    compare there would silently disarm a target that used to fire — so an
    unknown metric keeps the older, looser behaviour rather than a new
    stricter one.
    """
    if not state.score_events:
        return True
    return metric_names_match(state.score_events[-1].metric_name, config.target_metric)


def evaluate_stops(
    config: BudgetConfig,
    state: BudgetState,
    *,
    now: datetime | None = None,
) -> StopReason:
    """Return the first matching automatic stop reason, or ``none``.

    ``max_submissions=0`` means **do not submit**, not **do not run**. The cap
    is a limit on an action the campaign may take; reaching it is only a reason
    to stop once the campaign has actually taken it. Evaluated before the first
    step, ``0 >= 0`` ended a campaign having done nothing — one `stop` decision
    and no research — so the natural way to ask for "run, but never upload"
    silently asked for nothing at all.

    `submit_tools_allowed` is what enforces the zero; this only stops on a cap
    the campaign has spent.

    The breaker is evaluated **first**, because every other reason here answers
    "am I finished?" and a campaign whose every execution is failing is not
    finished — it is broken, and each further step spends budget to learn
    nothing. It reports `failing` rather than any of the completion reasons so
    the transcript cannot read as a normal end.
    """
    if (
        config.max_consecutive_failures is not None
        and state.consecutive_failures >= config.max_consecutive_failures
    ):
        return "failing"
    if config.max_barren_steps is not None and state.steps_since_success >= config.max_barren_steps:
        return "failing"
    if (
        config.max_submissions is not None
        and config.max_submissions > 0
        and state.submissions >= config.max_submissions
    ):
        return "submission_budget"
    if config.max_wall_s is not None and state.elapsed_s(now=now) >= config.max_wall_s:
        return "wall_time"
    if config.max_cost_usd is not None and state.llm_cost_usd >= config.max_cost_usd:
        return "cost_budget"
    if (
        config.target_metric
        and config.target_value is not None
        and state.last_metric is not None
        and _last_metric_matches_target(config, state)
    ):
        if config.maximize and state.last_metric >= config.target_value:
            return "metric_target"
        if not config.maximize and state.last_metric <= config.target_value:
            return "metric_target"
    hist = state.metric_history
    n = max(1, config.plateau_window)
    if len(hist) >= n:
        window = hist[-n:]
        gain = max(window) - min(window)
        if gain <= config.plateau_epsilon:
            return "plateau"
    return "none"


def submit_tools_allowed(config: BudgetConfig) -> bool:
    """False when the campaign is configured never to submit.

    Enforced by removing the submit tools from the allowlist rather than by
    approving and then refusing them. `--yes` maps every gated tool to
    `auto_approve`, so approval is not a brake in a non-interactive run: a
    campaign told not to submit could still upload to Kaggle because nobody was
    at the terminal to say no.
    """
    return not (config.max_submissions is not None and config.max_submissions <= 0)


def budgets_from_metadata(meta: dict[str, Any]) -> tuple[BudgetConfig, BudgetState]:
    cfg = BudgetConfig.model_validate(meta.get("budgets") or {})
    state = BudgetState.model_validate(meta.get("budget_state") or {})
    return cfg, state


def budgets_to_metadata(
    meta: dict[str, Any],
    config: BudgetConfig,
    state: BudgetState,
) -> dict[str, Any]:
    out = dict(meta)
    out["budgets"] = config.model_dump()
    out["budget_state"] = state.model_dump()
    return out
