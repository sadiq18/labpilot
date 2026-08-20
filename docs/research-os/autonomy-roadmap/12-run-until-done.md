# M17 — Run until plateau or goal, and show progress

**Status:** implementation shipped 2026-08-20 — steps 2–5, plus a scale-free
plateau this plan did not ask for and could not have fired without ·
**exit criteria 3, 4, 5 demonstrated** over seven campaigns on 2026-08-20, and
**`max_steps` ended none of them**, which is what this milestone is for ·
**1 and 2 undemonstrated** — three blockers found and fixed on the way; what
remains is a codegen defect that leaves no `train.py` on the `run_experiment`
path, so every execution there fails before it can score
([design §13](design/12-run-until-done.md)) ·
**Blocker cleared:** [M7](01-technique-to-model.md) done 2026-08-07; step 1
landed with M8's score writer (PR #125) ·
**Design:** [design/12-run-until-done.md](design/12-run-until-done.md)

---

## Purpose

A campaign ends because it ran out of **steps**, not because it ran out of
**ideas**. `--max-steps` defaults to 8; every campaign this session was stopped
by that counter or by the policy declaring itself finished — never by the
objective.

Worse, the two stop conditions that *should* govern a long run are dead code.

### The finding: metric-driven stops can never fire

`BudgetState.metric_history` and `BudgetState.last_metric` are **read in four
places and written in none**. Confirmed by inspection:

```
grep -rn "metric_history|last_metric" src/labpilot/
  budgets.py   — declares and reads them
  loop.py:49   — reads last_metric (goal persistence)
  conduct.py   — prints last_metric in status
  (no writer anywhere)
```

`evaluate_stops` therefore evaluates:

| Stop | Fires? | Why |
|---|---|---|
| `submission_budget` | yes | counter is incremented on submit |
| `wall_time` | yes | derived from `wall_started_at` |
| `cost_budget` | yes* | *if anything ever records LLM cost |
| **`metric_target`** | **never** | `last_metric` is always `None` |
| **`plateau`** | **never** | `metric_history` is always empty |

So "stop when the goal is reached" and "stop when improvement flattens" — the
only two conditions a research campaign genuinely needs — are unreachable. Goal
persistence was added this session to stop premature exits; it is a workaround
for the same missing wiring.

## Goal

`research conduct "<goal>"` runs until the objective is met, improvement
plateaus, a budget binds, or it genuinely needs human guidance — and the
operator can see how far along it is at any moment.

## Approach

**1. Harvest the metric after every experiment.** The one change that makes two
existing stop conditions live. Read the value from the tool result (or its
`metrics_path`), append to `metric_history`, set `last_metric`, persist.

Metric-key drift matters here: templates emit `cv_<metric>` and the selector
defaults `tabular_regression` to `rmse` while a competition may score `mse`.
Resolution order should be `cv_<target>` → `<target>` → generic fallbacks, and
mixing keys into one series must be prevented (see
[M8](02-objective-loop.md)).

**2. Make `--max-steps` a cap, not the terminator.** Default to unbounded; keep
the flag for bounded debugging runs. `--max-wall-s` becomes the natural backstop
for an unattended campaign.

**3. Add stops for "cannot progress unaided".** An unbounded loop needs these or
it spins forever on a broken tool:

| Condition | Threshold | Action |
|---|---|---|
| Consecutive dispatch failures | 3 | pause, record suggestion |
| Steps producing no new metric | 6 | pause, record suggestion |
| No eligible tools available | — | pause, record suggestion |

Pause rather than complete, so `conduct continue` resumes after the operator
intervenes. Each records *why* through `record_suggestion`, which is already the
mechanism for "the system telling you what it lacked".

**4. Show goal progress every step.** One line, answerable at a glance:

```
goal mse: best 120 → target 5 · 41% closed · 3 result(s) · 0 since improvement
```

"Percent closed" is the fraction of the distance from the *first* result to the
target that has been covered — it can go negative, which is itself informative.
`steps since improvement` is the plateau signal made visible before the stop
fires.

A prototype of this rendering was written and validated during the session;
`goal_progress(config, state)` producing the line above is a known-good shape.

**5. Surface it in `conduct status` too**, so a detached campaign can be checked
without tailing logs.

## Exit criteria

1. A campaign with an unreachable target stops on **`plateau`**, not on
   `max_steps`.
2. A campaign with a reachable target stops on **`metric_target`** and reports
   `REACHED`.
3. Killing the trainer mid-campaign produces a **`needs_guidance`** pause with a
   recorded reason, not a spin.
4. Every step prints a goal-progress line; `conduct status` shows the same.
5. `metric_history` is non-empty after any campaign that ran an experiment.

## Traps

- **Plateau is meaningless before [M7](01-technique-to-model.md).** Every
  experiment currently returns MSE 194.80, so a plateau stop would fire on step
  2 of every campaign and look like correct behaviour while hiding the real
  defect. Wire the harvesting now; do not enable the plateau stop as a default
  until techniques can actually move the score.
- **Unbounded means unbounded.** Without the `needs_guidance` stops, a campaign
  that cannot dispatch anything will loop forever burning LLM budget on policy
  calls. The failure counters are not optional.
- **Do not conflate "no metric this step" with failure.** Analysis, planning and
  reflection legitimately produce no metric. The counter is about *consecutive
  steps with no new result*, which is why the threshold is 6 and not 1.
- **`llm_cost_usd` is also never written**, so `cost_budget` is effectively dead
  too. Worth wiring alongside, especially once [M10](04-llm-tiering.md) routes
  paid providers.

## Related code

- `src/labpilot/research_engine/conductor/budgets.py` — `BudgetConfig`, `BudgetState`, `evaluate_stops`
- `src/labpilot/research_engine/conductor/loop.py` — `run_until_stop`, the `max_steps` bound, `_objective_unmet`
- `src/labpilot/research_engine/conductor/checkpoint.py` — `load_budget_pair`, `persist_budgets`
- `src/labpilot/research_engine/conductor/metrics.py` — `record_suggestion`
- `src/labpilot/cli/conduct.py` — `--max-steps`, `conduct status`
