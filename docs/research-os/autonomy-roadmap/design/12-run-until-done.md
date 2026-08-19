# Design — M17: run until plateau or goal

**Plan:** [../12-run-until-done.md](../12-run-until-done.md) · **Status:** design ·
**Owner:** unassigned · **Shares wiring with:** [M8](02-objective-loop.md)
(`metric_history` / `last_metric`, now written) ·
**Audited against:** [M12](../06-beyond-kaggle.md) (§10)

---

## 1. Background

M17 was written when its own first step was also its blocker: the two
metric-driven stops read fields nothing wrote. M8's score writer (PR #125)
closed that — `_record_experiment_outcome` appends a `ScoreEvent` per successful
experiment and derives both fields from the series
([loop.py:723](../../../../src/labpilot/research_engine/conductor/loop.py#L723)).
`metric_target` and `plateau` can now fire. What remains is the loop that would
let them.

| Plan step | State |
|---|---|
| 1. Harvest the metric after every experiment | **Done** (M8). Verify, do not rebuild |
| 2. `--max-steps` becomes a cap | Not started |
| 3. Stops for "cannot progress unaided" | **Partial** — M20/#139 shipped `failing`; two conditions uncovered |
| 4. Goal-progress line every step | Not started |
| 5. Same line in `conduct status` | Not started |

---

## 2. Problem statement

**A campaign ends on a step counter.**
[loop.py:1116](../../../../src/labpilot/research_engine/conductor/loop.py#L1116)
is `for step in range(max_steps)`; its `else:` clause
([loop.py:1536](../../../../src/labpilot/research_engine/conductor/loop.py#L1536))
pauses with `stop_reason: max_steps`. Three CLI options default it to 8
([conduct.py:243](../../../../src/labpilot/cli/conduct.py#L243),
[415](../../../../src/labpilot/cli/conduct.py#L415),
[452](../../../../src/labpilot/cli/conduct.py#L452)). Every campaign in this
roadmap's evidence logs ended there or on an advisory policy stop — none on the
objective.

**Two of the three "cannot progress" conditions have no counter.** M20 shipped
`consecutive_failures` (threshold 3) and `steps_since_success` (threshold 8),
both reported as `failing`. Uncovered:

- *A step that succeeds and writes no comparable score.* `record_execution`
  resets `steps_since_success` on any successful execution, while the score
  writer skips placeholder runs and non-finite metrics. `score_events` can stay
  frozen while the counter watching for a stall keeps resetting. Both objective
  stops read only the series; nothing counts steps against it.
- *No eligible tool.* `plan.unmapped` files a suggestion and continues
  ([loop.py:1284](../../../../src/labpilot/research_engine/conductor/loop.py#L1284)–[1305](../../../../src/labpilot/research_engine/conductor/loop.py#L1305)),
  with no counter and no bound. At 8 steps that is 8 wasted steps; unbounded it
  is a forever-spin burning one policy call per step.

**`plateau` cannot fire on most metrics.** `plateau_epsilon` defaults to `1e-6`
and is compared against a raw spread
([budgets.py:390](../../../../src/labpilot/research_engine/conductor/budgets.py#L390)),
so whether the stop works at all depends on the magnitude the metric happens to
be measured in. On rogii's RMSE ≈ 1380 a plateau is three readings within
0.0000001% of each other — never. This was survivable while `max_steps` ended
every campaign. It is not survivable once `max_steps` is gone, because plateau
then becomes the terminator of record (exit criterion 1) and an under-firing
plateau means the campaign runs until a clock instead — the same failure in a
new costume.

**Progress is invisible.** `goal_progress` exists only as a promise —
[budgets.py:245](../../../../src/labpilot/research_engine/conductor/budgets.py#L245)
reserves its `(config, state)` signature and requires it to call `score_summary`
rather than deriving the same four numbers a second way. `conduct status` prints
`last_metric=190.97`
([conduct.py:592](../../../../src/labpilot/cli/conduct.py#L592)) — no target, no
best, no direction, no distance.

That raw field **stays** where it is; the goal line joins it rather than
replacing it. An earlier version of this design removed it, on the argument that
progress deserves one rendering — which conflated a field dump with an
interpretation. `metric_target` fires on `last_metric` while the goal line shows
`best`, and the two differ the moment a run regresses, so removing it displayed
the number the stop is evaluated against nowhere at all. It also blanked the
metric entirely for a session predating `score_events`, which has readings here
and an empty series.

**One correction to the plan's evidence.** Dispatch failures *do* reach the
breaker on the campaign path: the `except` at
[loop.py:1436](../../../../src/labpilot/research_engine/conductor/loop.py#L1436)
calls `_record_experiment_outcome(succeeded=False)` for experiment tools. The
uncounted handler at
[loop.py:1528](../../../../src/labpilot/research_engine/conductor/loop.py#L1528)
is the legacy M2 single-tool path, reachable only with `campaign_mode=False`. No
new failure counting is needed.

---

## 3. Requirements

**Functional**

| # | Requirement |
|---|---|
| F1 | `conduct run` / `continue` / `resume` take no step bound by default; `--max-steps N` still bounds a debugging run |
| F2 | A stalled campaign stops on `needs_guidance` after **`max_barren_steps + 2`** conductor steps with no new comparable score, or **3** consecutive steps whose plan maps to no tool. The first threshold is derived, not chosen — see §8.2 |
| F3 | `needs_guidance` sets session status `paused` and records a suggestion naming the condition |
| F4 | A `needs_guidance` pause resumes under `conduct continue` with **no** `--session` argument, and the resumed run *dispatches* — being findable is not being resumable (§8.5) |
| F5 | One goal-progress line per conductor step, and the identical string from `conduct status` |
| F6 | `plateau` fires on a flat series regardless of the metric's magnitude — the same three readings must plateau whether they are ≈1380 or ≈0.91 |

**Non-functional**

| # | Requirement |
|---|---|
| N1 | Every stop threshold is a **count of decisions** (steps, failures, unmapped plans), never a quantity in domain units. An operator may still set `--max-wall-s`; nothing defaults it |
| N2 | Metric direction is read from `ScoreEvent.maximize` carried by the series — never from competition config, and never inferred from a metric's name |
| N3 | No existing in-process caller changes behaviour — `run_until_stop`'s own default stays `8`, covering **6** test modules and `conductor/__init__` |
| N4 | `goal_progress` derives nothing itself: every number comes from `score_summary`. Cost is O(comparable tail), evaluated at most twice per step |
| N5 | Persistence is additive — 3 `BudgetState`/`BudgetConfig` additions, all defaulted, so sessions written before this milestone load unchanged |
| N6 | Nothing added here names a submission, a leaderboard, a competition or a notebook. A campaign whose validator is a benchmark harness or a simulator reads the same |

---

## 4. Goals & success metrics

The plan's five exit criteria, made checkable:

| # | Criterion | Check |
|---|---|---|
| 1 | Unreachable target stops on `plateau` | Campaign with a target far past reach: `stop_reason == "plateau"` in session metadata, step count > 8 |
| 2 | Reachable target stops on `metric_target` | Target set just past the current best: `stop_reason == "metric_target"`, status line reports `REACHED` |
| 3 | Broken trainer gives a `needs_guidance` pause | Session `paused`, `stop_reason == "needs_guidance"`, a suggestion of that kind names the cause. Three *failed executions* give `failing` instead, by §9 — this criterion is met via the no-new-score and no-eligible-tool paths |
| 4 | Every step prints a goal line; `status` shows the same | One `goal ` line per step in the transcript; byte-identical string from `conduct status` |
| 5 | `metric_history` non-empty after a campaign that ran an experiment | **Already true** (M8) — asserted as a regression check, not new work |

Criteria 1–3 cannot be met by unit tests. They need a campaign on a workspace
that passes PR #142's objective preflight — the same evidence M8 and M11 still
owe.

---

## 5. Scope

**In:** the step bound; two stall counters and the `needs_guidance` stop; a
scale-free plateau comparison (§8.3); `goal_progress` and its two call sites;
the `no_capability` miscount in `record_suggestion` (§8.5).

**Out, and why:**

- **`llm_cost_usd`**, so `cost_budget` stays dead. `fitroute` records *tokens*
  (`BudgetLedger.record(provider, tokens=...)`), not dollars; converting them
  needs a per-model price table, which belongs to the router's roadmap. This
  design does not pretend `cost_budget` works, and does not lean on it as the
  backstop for an unbounded run — §8.2's counters are that.
- **The comparator's absolute `noise_epsilon`.** `compare()` applies the same
  raw-delta-vs-constant test at
  [comparator.py:58](../../../../src/labpilot/research_engine/shared/experiments/comparator.py#L58),
  so the scale assumption §8.3 removes from `plateau` also lives in the evidence
  path. Fixing it there changes recorded verdicts and needs its own measurement;
  named here so it is not mistaken for already handled.
- **Multi-objective progress.** `ScoreEvent` is one scalar per experiment, by
  M8's design. `goal_progress` inherits that and must not become the fifth place
  that decides which metric is primary — when [M12](../06-beyond-kaggle.md)'s
  `ValidationResult` carries several, the series changes first and the renderer
  follows.

---

## 6. Design

```
  conduct run  ──►  max_steps: None (unbounded)
                         │
                         ▼
     ┌──────────  evaluate_stops(cfg, state)  ────────────┐
     │  failing ─ budgets ─ metric_target ─ needs_guidance ─ plateau
     │                            ▲                          ▲
     │           steps_since_new_score  (6)      scale-free spread
     │           consecutive_unmapped   (3)
     └───────────────────────────────────────────────────────┘
                         │
                         ▼
        goal_progress(cfg, state) ──► step line
                                 └──► conduct status
```

The one ordering decision that carries weight: **`needs_guidance` precedes
`plateau`.** Plateau is a claim about results — "improvement has flattened". A
campaign that produced no new score for six steps has not flattened; it is
stuck, and its last `plateau_window` readings are unchanged for the trivial
reason that nothing wrote to them. Reporting `plateau` there is a stop asserting
something it did not measure, the exact shape [M20](../15-gates-must-fail.md)
exists to remove. Ordering is the entire fix — `plateau` needs no freshness
guard of its own.

It sits *after* `metric_target` because a campaign that reached its goal is
finished, not stuck. That case is unreachable in practice (the target is
evaluated every step), and the ordering should not depend on it being so.

---

## 7. Components & responsibility

| Component | Responsibility | Interface |
|---|---|---|
| `loop.py` — step loop | Unbounded iteration with an optional cap; increments both new counters | `run_until_stop(max_steps: int \| None = 8)` |
| `loop.py` — `_record_experiment_outcome` | Resets `steps_since_new_score` **only** where `score_events` grows | unchanged signature |
| `budgets.py` — `evaluate_stops` | Adds `needs_guidance`, ordered per §6; plateau compares against a scale-free floor | `(config, state) -> StopReason` |
| `budgets.py` — `_noise_floor` | The one place *how* to be scale-free is defined; the band is the caller's | `(values, absolute, relative) -> float` |
| `budgets.py` — `goal_progress` | Renders one line from `score_summary`; derives no metric and no direction itself | `(config, state) -> str \| None` |
| `cli/conduct.py` | Passes `max_steps=None`; exposes `--plateau-epsilon` / `--plateau-rel-epsilon`; prints the goal line in `status` | 3 option sites + status |
| `metrics.py` — `record_suggestion` | Stops charging `no_capability` for non-capability kinds | adds `kind` guard |

---

## 8. Implementation details

### 8.1 The bound becomes optional

```python
step = -1
while True:
    step += 1
    if max_steps is not None and step >= max_steps:
        store.update_session_status(session_id, "paused")
        _progress(f"Reached max_steps={max_steps}")
        save_checkpoint(store, session_id, extra={"stop_reason": "max_steps"})
        break
    ...
```

The `for/else` cannot express "bound only when asked", which is why the shape
changes rather than passing `range(max_steps or sys.maxsize)` — that would
report `max_steps` on a run that never had one.

### 8.2 Two counters

| Field (`BudgetState`) | Incremented | Reset |
|---|---|---|
| `steps_since_new_score` | once per step, beside `steps_since_success` ([loop.py:1174](../../../../src/labpilot/research_engine/conductor/loop.py#L1174)) | in `_record_experiment_outcome`, only when a `ScoreEvent` is appended |
| `consecutive_unmapped` | on the `plan.unmapped` branch | on any step whose plan maps ≥1 tool |

`BudgetConfig` gains `max_steps_without_score: int | None = DEFAULT_MAX_BARREN_STEPS + 2` and
`max_consecutive_unmapped: int | None = 3`; `None` disables either, as
`max_consecutive_failures` already allows.

The reset site is the whole point of the first counter. Resetting on a
*successful execution* is what `steps_since_success` already does, and that is
precisely the reset that hides a run writing placeholder metrics. Defining it on
the series rather than on a tool list also means it needs no maintenance when a
new validator arrives: `_EXPERIMENT_TOOLS` is a hardcoded pair, and this counter
never consults it.

**The threshold is derived from `max_barren_steps`, not chosen.** A score append
always also resets `steps_since_success` — the writer records the execution
before the score — so `steps_since_new_score >= steps_since_success` in every
reachable state. Set below M20's barren threshold this counter fires *first in
time* on every campaign, and `failing` becomes unreachable: a campaign that
executed nothing lands in `paused`, reading like a normal end, which is the one
distinction M20's breaker exists to draw. The plan asked for 6, and 6 would have
silently retired a stop that took nine campaign runs to earn. The margin above 8
only decides how long a campaign that *is* executing may keep producing nothing
comparable — the case barren cannot see, and the only one this counter is for.

### 8.3 A plateau that does not depend on the metric's units

One helper defines "too small to be a change", and both readers use it:

```python
def _noise_floor(values: list[float], absolute: float, relative: float) -> float:
    """Absolute floor, or a fraction of the readings' own magnitude."""
    scale = max((abs(v) for v in values), default=0.0)
    return max(absolute, relative * scale)
```

```python
window = hist[-n:]
if max(window) - min(window) <= _noise_floor(window, config.plateau_epsilon, config.plateau_rel_epsilon):
    return "plateau"
```

**Two bands, not one.** `plateau` asks whether a whole *window* failed to move;
`_steps_since_improvement` asks whether a single *step* cleared measurement
noise. They read as one question and are two — a window of three readings 0.05%
apart spans 0.1%, which is a plateau by the wide band while every step was an
improvement by the tight one, and both statements are true. Answering both with
`plateau_rel_epsilon` resolved that contradiction the wrong way: an accuracy
series gaining 0.05% a run reported three experiments with no improvement, which
is exactly what `available_tools`' stagnant clause and the stagnation mint read,
so a campaign improving on every run was told it was stuck and had hypotheses
minted at it. `improvement_rel_epsilon` defaults two orders of magnitude tighter
at `1e-5`; erring permissive is the safe direction, since a floor set too low
resets a counter and one set too high invents stagnation.

`plateau_rel_epsilon: float = 1e-3` — a 0.1% spread. `plateau_epsilon` keeps its
`1e-6` default and its meaning, and now acts as the floor for a series whose
values sit at or near zero, where a relative test degenerates. Taking the larger
of the two makes the combination backward compatible: no configuration that
fires today stops firing.

`_steps_since_improvement` takes the same floor, because its docstring already
requires the two to agree about what "no change" means — and it drives the
gathering gate and the policy's view of progress, so an absolute `1e-6` there
reports a campaign improving on every run as permanently stagnant.

This is F6, and it is also the reason the unbounded default cannot ship first
(§13).

### 8.4 `goal_progress`

Pure function of the pair, in `budgets.py` beside `score_summary` — no store, so
it is testable without a session.

```
goal mse: best 120 → target 5 · 41% closed · 3 result(s) · 0 since improvement
```

```python
gain = (best - first) if maximize else (first - best)
span = (target - first) if maximize else (first - target)
pct  = gain / span
```

`best`, `metric_name` and `steps_since_improvement` come from
`score_summary(state, config)`; `first` and the count come from the same
comparable tail. `maximize` is `ScoreEvent.maximize`, carried by the series —
never re-derived from the metric's name and never read back from competition
config (N2).

| Case | Rendered |
|---|---|
| No target set | `goal cv_rmse: best 190.97 · 3 result(s) · 0 since improvement` |
| No comparable score yet | `goal rmse: no result yet · target 2.236` |
| Neither target nor score | `None` — caller prints nothing |
| `span <= 0` (first reading already met the target) | `target met at first result`, no percentage |
| Latest reading worse than the best | percentage unchanged; never negative, because `best_so_far` includes the first reading — see below |
| One reading | `first == best` → `0% closed` |

**One divergence from the plan.** It describes percent-closed as able to go
negative, "which is itself informative". Measured from `best_so_far` it cannot:
the best includes the first reading, so the gain is never below zero. Measuring
from the *latest* reading instead would make it negative-capable, at the cost of
a percentage that contradicts the `best …` it sits beside and that bounces with
every run. Progress banked is progress kept — the best model is the one
retained — so the percentage stays best-based, and the regression signal the
plan wanted lives in `since improvement`, which is already on the line.

### 8.5 The `needs_guidance` stop

```python
store.update_session_status(session_id, "paused")
record_suggestion(store, session_id, why, kind="needs_guidance", context={...})
save_checkpoint(store, session_id, extra={"stop_reason": "needs_guidance", ...})
```

`paused` is what `conduct continue` resumes
([conduct.py:383](../../../../src/labpilot/cli/conduct.py#L383)) and what
`latest_active_session` counts as live. `failed` — where `failing` puts a
session, deliberately — is neither.

**Parking the session is not enough, and this is where the first version was
wrong.** The counters that tripped the stop are persisted at their thresholds,
so the resumed run's first `evaluate_stops` re-fired it before dispatching
anything: the campaign took zero steps, every time, however thoroughly the
operator fixed what it asked about. `_run_until_stop_inner` therefore clears
both guidance counters at start-up — invoking the loop again *is* the operator
saying "try again", and a campaign still unable to progress simply spends them
afresh and pauses again. Cleared in the loop rather than in `conduct continue`
so every resumer gets it. The M20 breaker's counters are deliberately not
cleared: `failing` parks a session in `failed`, which resuming must name with
`--session` to reach at all, and that friction is the distinction.

`record_suggestion` increments the `no_capability` metric unconditionally
([metrics.py:52](../../../../src/labpilot/research_engine/conductor/metrics.py#L52)),
so a `needs_guidance` suggestion would inflate a counter about missing tools.
The increment moves under that kind.

---

## 9. Design choices & tradeoffs

| Choice | Rejected | Taken | Tradeoff |
|---|---|---|---|
| Unbounded default | Library and CLI both `None` | CLI only; library stays `8` | An odd asymmetry, against a missed test call site becoming a hang in a suite with no timeout |
| Backstop when unbounded | A defaulted wall clock (6h was drafted) | Decision counters only; `--max-wall-s` stays operator-supplied and unset | See §10.1 — no seconds-based default can be right for every domain, and the counters already bound a stuck campaign to ~8 steps |
| Plateau noise floor | Absolute only; or relative only | `max(absolute, relative × magnitude)` | Two knobs instead of one, against a stop that works on RMSE 1380 and accuracy 0.91 alike |
| `needs_guidance` shape | Three new `StopReason` literals | One literal; detail in the rationale and `observe` | Follows `failing`, which already lumps two conditions; coarser to grep for |
| Stall detection | Reuse `steps_since_success` | New `steps_since_new_score` | One more field, against a counter that resets in exactly the case being detected |
| Plateau vs stall | Guard `plateau` on series freshness | Order `needs_guidance` first | Same outcome, no new condition on a stop that already works |
| `failing` on 3 dispatch failures | Change it to pause, per the plan | Leave it `failed` | Diverges from the plan's table. M20 shipped `failed` with a recorded rationale — a broken campaign must not land where a successful one does — and the plan's row is already served by that breaker. The adopted distinction: **`failing` = something is broken; `needs_guidance` = nothing is broken and it still cannot proceed** |

---

## 10. Domain neutrality — audited against M12

[M12](../06-beyond-kaggle.md) is the product thesis: the loop generalises to
benchmarking, paper reproduction, simulation science and software engineering
only if Kaggle assumptions are not baked into the control plane. M17 sits
squarely in that control plane, so every decision here was checked against it.

### 10.1 The rule this design follows

> **A threshold may be a count of decisions. It may never be a quantity in
> domain units.**

Steps, failures and unmapped plans are properties of the *loop* — six steps
without a score means the same thing whether a step takes four seconds or four
hours. Seconds, metric magnitudes and dollars are properties of the *domain*,
and any constant chosen for one is wrong for the next.

This is what removed the 6h wall-clock default an earlier draft carried. It was
calibrated, unavoidably, to a tabular campaign whose experiments take minutes: a
simulation or paper-reproduction validator whose single experiment runs eight
hours would have been killed before its first result, reported as `wall_time`,
and looked like a working stop. `--max-wall-s` remains available to an operator
who knows their domain; nothing defaults it. The same rule is why §8.3 exists —
`1e-6` is a magnitude, and a magnitude is a domain unit.

### 10.2 Decision-by-decision

| Decision | Domain-neutral? | Note |
|---|---|---|
| Unbounded `max_steps` | Yes | Removes a bound; adds no assumption |
| `max_steps_without_score` | Yes | Counts decisions (N1), and derived from `max_barren_steps` rather than chosen |
| `max_consecutive_unmapped = 3` | Yes | Counts decisions |
| `steps_since_new_score` reset site | Yes — **improves neutrality** | Keyed on `score_events` growing, not on `_EXPERIMENT_TOOLS`, so a validator that produces scores through a different tool is counted with no edit |
| `needs_guidance` pause + suggestion | Yes | `record_suggestion` is the generic "say what was lacking" channel |
| Plateau noise floor | Yes, **after §8.3** | Was the one hard scale assumption in this milestone |
| `goal_progress` direction | Yes | Reads `ScoreEvent.maximize` (N2) — which is already M12's prescribed fix for "`maximize` is a competition property today"; the series is the carrier |
| `goal_progress` percent-closed | Yes, within one limit | Assumes a single scalar objective — inherited from `ScoreEvent`, not introduced here (§5) |
| `goal_progress` line content | Yes (N6) | Names a metric, a target and a count. No submission, leaderboard or competition appears |

### 10.3 Kaggle concepts deliberately left alone

`submission_budget`, `submit_tools_allowed` and the `submissions` counter are
Kaggle-shaped and pre-existing; M12 puts them behind the Kaggle validator. This
milestone neither extends nor depends on them — the goal line carries no
submission concept, so a campaign with no notion of "submitting" reports its
progress complete. Likewise `_MEASUREMENT_PREFIXES` carries `lb_`
(leaderboard); `goal_progress` reuses `metric_names_match` as-is and adds no
prefix of its own.

---

## 11. Observability

This milestone is largely an observability change, so the surfaces are the
deliverable rather than instrumentation around it.

| Signal | Where | Answers |
|---|---|---|
| `goal <metric>: best … → target … · N% closed · R result(s) · S since improvement` | every step *before* its stop evaluation, and `conduct status` | How far along, and whether the last runs moved anything |
| `stop_reason` | session metadata via `save_checkpoint` | Which condition ended it — `max_steps` should become rare |
| suggestion of kind `needs_guidance` | `conduct status`, gap ledger | What the operator must fix before resuming |
| `last_metric` | the raw `budget_state` line in `conduct status` | The reading `metric_target` is actually evaluated against |
| `steps_since_new_score`, `consecutive_unmapped` | the stop's `observe` payload | Stuck producing nothing, or stuck choosing nothing |
| resolved noise floor | logged once per plateau evaluation that fires | Whether a plateau was a real flattening or a floor set too wide |

The failure this must not repeat: a run ending with a bare reason and no account
of why. `failing` already carries its counters and last errors into the
rationale; `needs_guidance` does the same.

---

## 12. Testing strategy

**Unit — scale invariance (F6).** The same three readings scaled by 1e3 and by
1e-3 produce the same `plateau` verdict. This is the test that would have caught
the absolute epsilon, and it is written first.

**Unit — `goal_progress`.** Every row of §8.4, plus a series whose metric changed
mid-campaign rendering from the comparable tail only, and a `maximize` series
rendering the same shape with the sign flipped.

**Unit — `evaluate_stops` ordering.** A state satisfying both `needs_guidance`
and `plateau` returns `needs_guidance`; one satisfying `metric_target` and
`needs_guidance` returns `metric_target`; `None` on either threshold disables it.

**Unit — counter lifecycle.** `steps_since_new_score` resets on an appended
`ScoreEvent` and **not** on a successful execution that appended none — the
defect the counter exists for, asserted directly. `consecutive_unmapped` resets
on a mapped plan.

**Unit — the unbounded loop terminates.** `max_steps=None` against a registry
that maps nothing reaches `needs_guidance` within `max_consecutive_unmapped`
steps. This is the one test that fails as a hang rather than an assertion, so its
fake store carries an explicit ceiling that raises.

**Unit — no domain units (N1).** Assert that no `BudgetConfig` field this
milestone adds is denominated in seconds, currency or metric magnitude — a
guard against the next threshold being calibrated to whichever workspace is open.

**Integration — resumability (F4).** A `needs_guidance` pause is picked up by
`conduct continue` with no `--session` **and the resumed run dispatches a tool
without re-firing the stop**. The second half is the assertion that matters: a
test checking only that the session is findable passed for a resume that took
zero steps.

**Contract ([M15](../10-capability-audit.md)).** Two different score series must
produce two different `goal_progress` lines. A renderer that ignores its input is
the hollow-layer failure this roadmap is named for.

---

## 13. Rollout

**Prerequisites, in order.**

1. **[PR #142](https://github.com/sadiq18/labpilot/pull/142) merges first.** It
   rewrites objective resolution and touches `cli/conduct.py`, where several of
   this milestone's edits land. It also refuses to launch on both live
   workspaces — rogii's `evaluation_metric` is `None` — so until that is
   resolved there is no workspace on which a `metric_target` campaign can
   honestly run, and criteria 1–3 all need one.
2. ~~**The baseline step-burner is fixed.**~~ **Downgraded to known waste
   during implementation — a correction to the paragraph that used to sit
   here.** It claimed the burner was "an infinite loop that `needs_guidance`
   does **not** catch — every step succeeds" and then, in the same sentence,
   that `steps_since_new_score` stops it at step 6. Both cannot be true, and
   the second is: `generate_plan` is not an experiment tool, so a step spent on
   it grows no series and never resets that counter. The burner costs six
   steps, not a campaign. Fixing it remains worth doing —
   [deferred/baseline-plan-step-burner.md](../deferred/baseline-plan-step-burner.md)
   holds the analysis — but it does not gate the unbounded default, and
   pretending it did would have blocked this milestone behind a campaign-policy
   decision that its own document says needs measurement first.

**Landing order.** Scale-free noise floor (§8.3) → counters and `evaluate_stops`
ordering → `goal_progress` and its two call sites → the unbounded default last.
The floor comes first because plateau is the terminator of record once the step
bound is gone; shipping the unbounded default onto an under-firing plateau
substitutes a clock for a step counter and calls it progress.

**Not behind a flag.** An unbounded default that ships off is a default nobody
exercises. What limits blast radius instead: `--max-steps N` keeps working, the
library default is untouched (N3), and every new threshold has a `None` opt-out.

**Rollback** is `--max-steps 8` — the current behaviour, one flag away, with no
migration to undo (N5). The noise-floor change is rolled back by setting
`plateau_rel_epsilon: 0` (and `improvement_rel_epsilon: 0`), which restores the
absolute comparison exactly.

**Roadmap note.** The README puts the ground-truth phase (M22→M26) ahead of this
work, on the argument that optimising against an unverified target is "a more
sophisticated way of optimising the wrong problem". M17 makes campaigns run
*longer* against that target. Building it now is defensible — it is phase 3 and
unblocked — but the sequencing is a deliberate choice, and PR #142's preflight is
what makes it survivable.
