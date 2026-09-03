"""Campaign budgets and automatic stop evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from labpilot.accessor.common.provenance import failure_signature
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
    #: A schema question is open and there is no channel to ask it on. Distinct
    #: from `policy_stop` deliberately: M20's finding is that collapsing states
    #: into one boolean is how eight gates reported `pass` on things that could
    #: not run, and "waiting for a person" is not "decided to stop".
    "schema_question",
    "policy_stop",
    "max_steps",
    "failing",
    "needs_guidance",
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
#:
#: **Identical** is load-bearing, and for a long time only the prose said so.
#: The counter asked how many failures there had been, never whether they were
#: the same one — see `BudgetState.failures_are_repeating`, which is what the
#: breaker now consults alongside it.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3

#: Steps allowed with no successful experiment at all before stopping.
#:
#: The slower sibling of the above, for a campaign that never reaches an
#: execution: rogii's S-021 spent 30 steps choosing `run_experiment` while
#: `plateau` could not fire (it needs metrics) and `metric_target` could not
#: fire (it needs a metric). A campaign that has produced nothing after this
#: many steps is not mid-flight.
DEFAULT_MAX_BARREN_STEPS = 8

#: Conductor steps allowed with no *new comparable score* before pausing.
#:
#: The gap `steps_since_success` cannot see. That counter resets on any
#: successful execution, and the score writer skips a placeholder run or a
#: non-finite metric — so a campaign can succeed on every step while the series
#: both objective stops read stays frozen.
#:
#: **Derived from the barren threshold, and strictly greater than it.** A score
#: append always also resets `steps_since_success` (the writer records the
#: execution before the score), so `steps_since_new_score >= steps_since_success`
#: for every reachable state. Set below `DEFAULT_MAX_BARREN_STEPS` this counter
#: therefore fires *first in time* on every campaign, and M20's `failing` — the
#: stop that keeps a broken campaign from reading as a normal end — becomes
#: unreachable. The plan asked for 6; 6 would have silently retired a stop that
#: took nine campaign runs to earn.
#:
#: The margin only decides how long a campaign that *is* executing is allowed
#: to keep producing nothing comparable, which is the case barren cannot see
#: and this counter exists for.
DEFAULT_MAX_STEPS_WITHOUT_SCORE = DEFAULT_MAX_BARREN_STEPS + 2

#: Consecutive steps whose plan mapped to no tool before pausing.
#:
#: `plan.unmapped` files a suggestion and continues. Bounded by `max_steps`
#: that cost a few steps; unbounded it is a forever-spin burning one policy
#: call per step and producing nothing.
DEFAULT_MAX_CONSECUTIVE_UNMAPPED = 3

#: Noise floor as a fraction of the readings' own magnitude.
#:
#: An absolute epsilon only works for metrics that happen to be measured in a
#: particular range. Three readings within 1e-6 of each other is a plateau on
#: an accuracy near 0.9 and an impossibility on rogii's RMSE near 1380, so
#: whether `plateau` could fire at all depended on the metric's units — a
#: domain assumption sitting in a control-plane stop, which is exactly what
#: docs/research-os/autonomy-roadmap/06-beyond-kaggle.md (M12) says must not
#: happen if the loop is to generalise past Kaggle.
#:
#: Survivable while `max_steps` ended every campaign. Not survivable once the
#: step bound is gone: `plateau` becomes the terminator of record, and an
#: under-firing one just substitutes a wall clock for a step counter.
#:
#: 1e-3 is a 0.1% spread. `0` disables the relative test and restores the
#: absolute comparison exactly, which is this change's rollback.
DEFAULT_PLATEAU_REL_EPSILON = 1e-3

#: The same idea for "this reading beat the ones before it", and deliberately
#: two orders of magnitude tighter.
#:
#: These read as one question and are two. `plateau` asks whether a whole
#: *window* failed to move; `_steps_since_improvement` asks whether a single
#: *step* cleared measurement noise. A window of three readings 0.05% apart
#: spans 0.1% — a plateau by the wide band while every step was an improvement
#: by the tight one, and both statements are true.
#:
#: Sharing one band made that contradiction resolve the wrong way: an accuracy
#: series gaining 0.05% a run reported three experiments with no improvement,
#: which is what `available_tools`' stagnant clause and the stagnation mint
#: read. A campaign improving on every run was told it was stuck.
#:
#: Erring permissive is the safe direction here — a floor set too low calls a
#: little noise an improvement and resets a counter; set too high it invents
#: stagnation and mints hypotheses against a campaign that is working.
DEFAULT_IMPROVEMENT_REL_EPSILON = 1e-5


class BudgetConfig(BaseModel):
    """Resource and objective limits for a campaign session."""

    max_submissions: int | None = None
    max_wall_s: float | None = None
    max_cost_usd: float | None = None
    target_metric: str | None = None
    target_value: float | None = None
    maximize: bool = True
    plateau_window: int = 3
    #: Absolute noise floor, and now the *lower* bound of one: `_noise_floor`
    #: takes the larger of this and the relative test, so no configuration
    #: that fires today stops firing. It still governs alone for a series
    #: whose values sit at or near zero, where a relative test degenerates.
    plateau_epsilon: float = 1e-6
    plateau_rel_epsilon: float = DEFAULT_PLATEAU_REL_EPSILON
    improvement_rel_epsilon: float = DEFAULT_IMPROVEMENT_REL_EPSILON
    #: `None` disables the breaker. Opt-out exists because a campaign
    #: deliberately probing a broken workspace is a legitimate thing to run.
    max_consecutive_failures: int | None = DEFAULT_MAX_CONSECUTIVE_FAILURES
    max_barren_steps: int | None = DEFAULT_MAX_BARREN_STEPS
    #: Both `None`-able for the same reason as the breaker above.
    max_steps_without_score: int | None = DEFAULT_MAX_STEPS_WITHOUT_SCORE
    max_consecutive_unmapped: int | None = DEFAULT_MAX_CONSECUTIVE_UNMAPPED


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


#: How many recent failures to keep, and therefore how long a cycle
#: `failures_are_repeating` can see. Three held only the failures that would
#: trip the shipped threshold, which is enough for a stall repeating one defect
#: and enough for an A/B oscillation, but not for an A/B/C one — that needs a
#: fourth slot to see the repeat of A. Five, at 200 characters each, is a
#: kilobyte of session metadata for a cycle length no repair loop should reach.
_FAILURE_WINDOW = 5


class BudgetState(BaseModel):
    """Live counters persisted in session metadata / metrics table."""

    submissions: int = 0
    llm_cost_usd: float = 0.0
    wall_started_at: str | None = None
    metric_history: list[float] = Field(default_factory=list)
    last_metric: float | None = None
    #: The comparable score series (M8). Written by
    #: `_record_experiment_outcome`, one event per successful experiment,
    #: which also derives `metric_history`/`last_metric` from it.
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
    #: Steps since `score_events` last grew. Distinct again from the above,
    #: which resets on any successful execution — a run that succeeds and
    #: writes a placeholder metric resets that counter while the series it is
    #: meant to guard stays frozen. Reset in `_record_experiment_outcome`, at
    #: the append, so it is keyed on the series rather than on a hardcoded
    #: tool list: a validator that produces scores another way is counted
    #: with no edit here.
    steps_since_new_score: int = 0
    #: Consecutive steps whose plan mapped to no tool at all.
    consecutive_unmapped: int = 0
    #: What the failures said, most recent last. Bounded — this is a stop
    #: *reason*, not a log, and it is written into session metadata.
    #: `_FAILURE_WINDOW` sets the bound; it is also the span
    #: `failures_are_repeating` can see, so the two moved together.
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
            self.recent_failures = [*self.recent_failures[-(_FAILURE_WINDOW - 1) :], excerpt]

    def failures_are_repeating(self) -> bool:
        """True when the campaign is stuck rather than working through defects.

        Three attempts that each fail *differently* is the repair loop doing its
        job — measured on playground-series-s6e8 (2026-08-30): an undeclared
        import, then a LightGBM 4 kwarg, then a pandas comparison, each one
        surfaced by fixing the last. Three that fail *identically* is a stall,
        and only the second is a reason to end a campaign. The breaker counted
        both the same way and stopped the converging one at three.

        Asked against **every** failure still in the window, not just the one
        before. Comparing adjacent pairs answers a narrower question — "is this
        the same as last time?" — and a loop where fixing A reintroduces B and
        fixing B reintroduces A answers no to it forever: measured on the branch,
        A/B/A/B/A/B/A/B reported novel on all eight and the breaker never fired.
        That cycle is a stall by any reading; it is only *adjacent* failures that
        differ. Convergence means each failure is one not seen before.

        The window (`_FAILURE_WINDOW`) is what bounds the cycle length this can
        see. A longer one still evades it and is left to `max_barren_steps`,
        which is the backstop for every case this predicate declines to stop.

        **Fewer than two recorded failures answers True**, keeping the old
        behaviour wherever this cannot see. A failure with no error text
        appends nothing, so a campaign whose failures arrive blank still stops
        at the threshold rather than running on a signal that does not exist.
        """
        if len(self.recent_failures) < 2:
            return True
        newest = failure_signature(self.recent_failures[-1])
        return any(failure_signature(prior) == newest for prior in self.recent_failures[:-1])

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
    #: The oldest reading in the comparable window — where this campaign
    #: started measuring the metric it is measuring now. `goal_progress`
    #: needs it to say how much of the distance to the target is covered.
    first_score: float | None = None
    #: How many comparable readings the window holds.
    result_count: int = 0
    #: The window's direction, carried so a consumer never re-derives it from
    #: the metric's name. Meaningless with no readings, hence the default
    #: rather than an opinion.
    maximize: bool = True


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
        first_score=values[0],
        result_count=len(values),
        maximize=maximize,
        last_3_scores=values[-3:],
        delta_vs_best=delta,
        steps_since_improvement=_steps_since_improvement(
            values,
            maximize,
            _noise_floor(values, config.plateau_epsilon, config.improvement_rel_epsilon),
        ),
        metric_name=events[-1].metric_name,
    )


def _noise_floor(values: list[float], absolute: float, relative: float) -> float:
    """How large a change has to be before it counts as one.

    The absolute floor, or a fraction of the readings' own magnitude,
    whichever is larger. A constant alone cannot serve both an RMSE near 1380
    and an accuracy near 0.9: it is a quantity in the metric's units, and
    every metric has different ones. Scaling by the window's magnitude asks a
    question with the same answer in every domain — *did these readings move
    by more than some fraction of what they measure?*

    `max` of the two rather than either alone. The relative test degenerates
    as the readings approach zero, and the absolute floor is what catches
    that; the absolute floor is meaningless at large magnitudes, and the
    relative one is what catches that. Taking the larger also means no
    configuration that fires today stops firing.

    Magnitude is `max(abs(v))` over the window rather than the best value, so
    it is defined without reference to direction and stays stable for a series
    that straddles zero.

    One definition of *how* to be scale-free, two bands. `relative` is the
    caller's, because `plateau` and `_steps_since_improvement` are asking
    different questions — see `DEFAULT_IMPROVEMENT_REL_EPSILON`. An earlier
    version took one band from the config for both readers, and a campaign
    gaining 0.05% a run read as stagnant.
    """
    scale = max((abs(v) for v in values), default=0.0)
    return max(absolute, relative * scale)


def _steps_since_improvement(values: list[float], maximize: bool, floor: float) -> int:
    """Experiments since one beat everything before it by more than `floor`.

    Measured against the best of the *preceding* readings, not the running
    best including itself — otherwise every event trivially ties its own best
    and nothing ever counts as an improvement. `floor` comes from
    `_noise_floor`, the same one `plateau` uses, so the two agree about what
    "no change" means.

    That floor used to be a bare absolute epsilon, and the hazard was real:
    this drives the gathering gate and the policy's view of progress, so
    1e-6 against a metric whose values live near or below it swallowed every
    real gain and the campaign read as permanently stagnant while improving
    on every run. `_noise_floor` scales with the readings, which removes the
    dependence on what units the metric happens to use.

    It is built from `improvement_rel_epsilon`, **not** the plateau band. A
    single step clearing measurement noise and a whole window failing to move
    are different questions, and answering both with the 0.1% plateau band
    recreated the same hazard from the other side.

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
        if gain > floor:
            improved_at = index
        best_before = max(best_before, value) if maximize else min(best_before, value)
    return len(values) - 1 - improved_at


def _fmt(value: float) -> str:
    """Short enough to read at a glance, exact enough to compare two lines."""
    return f"{value:g}"


def goal_progress(config: BudgetConfig, state: BudgetState) -> str | None:
    """One line saying how far this campaign has come, in the metric's own
    direction.

        goal mse: best 120 → target 5 · 41% closed · 3 result(s) · 0 since improvement

    Derives nothing itself. Every number comes from `score_summary`, because
    the primary metric already ended up with four disagreeing resolvers in
    this tree and a renderer quietly becoming the fifth is how that happens
    again. Direction is `ScoreEvent.maximize`, carried by the series — never
    read back from competition config and never inferred from the metric's
    name, which is what lets this line read the same for a campaign scored by
    a benchmark harness or a simulator as for one scored by a competition.

    **Percent closed** is the fraction of the distance from the *first*
    comparable reading to the target that has been covered. Measured from
    `best_so_far`, which includes that first reading, so it never goes
    negative — progress banked is progress kept, because the best model is
    the one retained. The plan expected a negative case; there is none to
    render, and the signal it wanted (the latest run went backwards) is
    `steps_since_improvement`, already on the line.

    The target is only shown when the series is measuring the metric it names.
    Rendering an `lb_auc` threshold beside a `cv_rmse` reading is the same
    mistake `_last_metric_matches_target` exists to keep the `metric_target`
    stop from making, and it is worse here, because a person reads this one.

    Returns `None` only when there is neither a target nor a result — a
    campaign with nothing to report, where the caller prints nothing rather
    than a line of empty fields.
    """
    # One scan. `score_summary` has already walked the comparable tail, and
    # this runs every conductor step against a series the campaign is designed
    # to grow — `_steps_since_improvement`'s docstring already counts the times
    # that walk is paid, and a renderer is not a good place to add another.
    summary = score_summary(state, config)

    metric = summary.metric_name or config.target_metric
    if metric is None:
        return None

    target: float | None = None
    if config.target_metric is not None and config.target_value is not None:
        # An empty series names no metric, so there is nothing to contradict
        # the target yet and it is shown.
        if summary.metric_name is None or metric_names_match(
            summary.metric_name, config.target_metric
        ):
            target = config.target_value

    best = summary.best_so_far
    first = summary.first_score
    if best is None or first is None:
        suffix = f" · target {_fmt(target)}" if target is not None else ""
        return f"goal {metric}: no result yet{suffix}"

    head = f"goal {metric}: best {_fmt(best)}"
    parts: list[str] = []
    if target is not None:
        head += f" → target {_fmt(target)}"
        span = (target - first) if summary.maximize else (first - target)
        gain = (best - first) if summary.maximize else (first - best)
        # A campaign whose first reading already met the target has no
        # distance to have covered; a percentage of zero distance is either
        # a division error or a meaningless 100%.
        parts.append("target met at first result" if span <= 0 else f"{gain / span:.0%} closed")
    parts.append(f"{summary.result_count} result(s)")
    parts.append(f"{summary.steps_since_improvement} since improvement")
    return " · ".join([head, *parts])


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

    `needs_guidance` is evaluated **before `plateau`** for a related reason.
    Plateau is a claim about results — *improvement has flattened*. A campaign
    that has produced no new score for six steps has not flattened; it is
    stuck, and the last `plateau_window` readings are unchanged for the
    trivial reason that nothing wrote to them. Firing `plateau` there is a
    stop asserting something it never measured, which is the shape M20 exists
    to remove. The ordering is the whole guard — `plateau` itself needs no
    freshness check.

    It is evaluated **after `metric_target`** because a campaign that reached
    its goal is finished, not stuck. That case is unreachable in practice, the
    target being checked every step, and the ordering should not depend on it
    being so.
    """
    if (
        config.max_consecutive_failures is not None
        and state.consecutive_failures >= config.max_consecutive_failures
        # Repeating, not merely numerous. A repair loop that fixes one defect
        # and surfaces the next is progress, and stopping it at three reported
        # "this model cannot write code" for a model that needed five attempts.
        # The distinct case is not unbounded: it accrues barren steps and stops
        # on `max_barren_steps` below.
        and state.failures_are_repeating()
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
    if (
        config.max_steps_without_score is not None
        and state.steps_since_new_score >= config.max_steps_without_score
    ):
        return "needs_guidance"
    if (
        config.max_consecutive_unmapped is not None
        and state.consecutive_unmapped >= config.max_consecutive_unmapped
    ):
        return "needs_guidance"
    hist = state.metric_history
    n = max(1, config.plateau_window)
    if len(hist) >= n:
        window = hist[-n:]
        gain = max(window) - min(window)
        if gain <= _noise_floor(window, config.plateau_epsilon, config.plateau_rel_epsilon):
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
