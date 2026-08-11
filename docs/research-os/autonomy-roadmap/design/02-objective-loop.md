# Design — M8: close the objective feedback loop

**Plan:** [../02-objective-loop.md](../02-objective-loop.md) · **Status:** design ·
**Owner:** unassigned · **Shares wiring with:** [M17](../12-run-until-done.md)
(`metric_history` / `last_metric`)

---

## 1. Problem

Three gaps, each confirmed against the current tree rather than assumed.

**1. The fields the policy needs are read in four places and written in
none.** `BudgetState.metric_history` and `.last_metric`
([budgets.py:70](../../../../src/labpilot/research_engine/conductor/budgets.py#L70))
are read by `evaluate_stops` for `metric_target`
([budgets.py:148](../../../../src/labpilot/research_engine/conductor/budgets.py#L148))
and `plateau`
([budgets.py:153](../../../../src/labpilot/research_engine/conductor/budgets.py#L153)),
by `_objective_unmet` in the run loop
([loop.py:119](../../../../src/labpilot/research_engine/conductor/loop.py#L119)),
and printed in the CLI status line
([conduct.py:554](../../../../src/labpilot/cli/conduct.py#L554)). Grepping the
tree for an assignment to either field (`metric_history\s*=` /
`last_metric\s*=`) outside tests returns nothing. `_record_experiment_outcome`
— the one place in the loop that already folds an experiment's result into
`BudgetState` — records `succeeded`/`error` only
([loop.py:82](../../../../src/labpilot/research_engine/conductor/loop.py#L82)).
Every reader has been dead code since the fields were added.

**2. The tree already has four independent "primary metric" resolvers, and
they disagree.** Adding a fifth ad hoc one to feed the writer would repeat a
mistake already made three times over:

| Resolver | Competition-aware? | Returns key? |
|---|---|---|
| `outcome._primary_metric` | No — fixed priority list | No, value only |
| `builder._primary_cv_keyed` | No — fixed priority list | Yes |
| `comparator.resolve_primary_metric_key_and_direction` | Yes — reads `competition.json`, prefers `cv_<spec.key>` | Yes, + direction |
| `ranking.resolve_primary_metric_key` | Yes — same idea, different call shape | Yes |

The cost of this is measured, not hypothetical: `builder._primary_cv_keyed`'s
own docstring records that on rogii 2026-08-07, **six evidence cards recorded
a "gain" of −194.30** by comparing a stub run's `cv_accuracy: 0.5` against a
real run's `cv_rmse: 194.80` — an accuracy read as an RMSE because the
resolver in use there doesn't know the two keys move in opposite directions.
This is the exact failure mode the plan's "beware metric-key drift" trap
describes, already realized once.

**3. One reflection-authored hypothesis path is dead; a sibling one is alive
and already covers a single bad experiment — but nothing covers a *run* of
them.** `StructuredReflection.new_hypotheses: list[HypothesisDraft]`
([models.py:106](../../../../src/labpilot/research_engine/shared/experiments/models.py#L106))
and `KnowledgeBase.update_from_reflection`
([knowledge.py:186](../../../../src/labpilot/research_engine/shared/experiments/knowledge.py#L186))
exist to turn a reflection's drafted hypotheses into knowledge-base entries —
but `StructuredReflection(...)` is constructed nowhere in production code, and
`update_from_reflection` has no caller outside its own module and tests. It
is dead machinery for a step that never runs. *(An earlier draft of this
design stopped here and concluded "nothing calls `HypothesisStore.create()`
from reflection" — that's true of this specific path and false of the system
as a whole; corrected below.)*

`maybe_mint_improvement_hypothesis`
([outcome.py:580](../../../../src/labpilot/research_engine/execution/outcome.py#L580))
is a sibling mechanism, and it *is* wired: called from
`update_hypothesis_from_local`
([outcome.py:1101](../../../../src/labpilot/research_engine/execution/outcome.py#L1101)),
itself called from `record_successful_execution`
([outcome.py:1200](../../../../src/labpilot/research_engine/execution/outcome.py#L1200))
right after `run_reflection` — which runs after **every** successful local
execution via `execution/engineer.py:161` — and again from `submit_learn`
([submit_learn.py:387](../../../../src/labpilot/research_engine/execution/submit_learn.py#L387))
on the submission path. When `summary.learning_loss > 0` (a regression
against the immediate parent), it already calls `HypothesisStore.create()`
with `reason=f"Execution {summary.execution_id} lost {loss:.4g} vs prior;
fork an improvement..."` — citing the prior experiment by id in the text —
`created_by=HypothesisCreatedBy.REFLECTION`, `origin=HypothesisOrigin.EXPERIMENT`,
and a structured `evidence=[{"kind": "experiment", "ref": execution_id}]`.
That is exit criterion 1, already met, for a single execution's regression —
with one inherited caveat: `summary.learning_loss` traces back through
`build_evidence_card`
([builder.py:318](../../../../src/labpilot/research_engine/evidence/builder.py#L318))
to `_primary_cv_keyed`
([builder.py:53](../../../../src/labpilot/research_engine/evidence/builder.py#L53))
— the same non-competition-aware resolver §1.2's table lists and the one
whose docstring records the measured −194.30 defect. The specific
accuracy-vs-rmse mismatch that caused that defect is now guarded
(treated as `missing_control`), but the resolver can still silently pick a
*self-consistent but wrong* key on both sides of a comparison (e.g.
`cv_accuracy` when the competition's real metric is `cv_rmse`, if both
happen to be present). This mechanism is solid ground to build Component 3
on, not because it's immune to §1.2's problem, but because it's already
running today regardless of what M8 does — flagged here, not re-fixed, per
§6's out-of-scope call on 4-resolver consolidation.

What it does not do: react to **sustained** stagnation across several
experiments that each look fine individually — `maybe_mint_improvement_hypothesis`
only ever compares one execution to its immediate parent. A campaign that
tries five different techniques, each a small loss or a wash against *its own*
parent, never crosses this function's `loss > 0` bar as a *pattern*, and
nothing today reads `steps_since_improvement` across the whole series to
notice five flat experiments in a row. That's the real remaining gap — not
"reflection produces nothing," but "reflection only looks one step back."

The other symptom the plan names still holds independently: what
`run_reflection` ([pipeline.py:34](../../../../src/labpilot/research_engine/reflection/pipeline.py#L34))
itself returns is a `CriticAssessment.next_steps: list[str]` (free text) and
a `recommend_next_experiment` result
([next_experiment.py:14](../../../../src/labpilot/research_engine/reflection/recommendation/next_experiment.py#L14))
— a `RecommendationDraft` whose `action="analyze_or_hypothesize"` fallback
literally tells the operator to run `research hypothesize list`. Nothing
about that return value itself becomes a `Hypothesis` row; it's
`maybe_mint_improvement_hypothesis`, a separate function running alongside it
in the same caller, that does.

---

## 2. The change

Three components, built and landed in this order because each one is a
prerequisite for verifying the next:

```
①  score writer            "what happened, comparably, durable"
        │
        ▼
②  observe bundle fields    "does the policy see it"
        │
        ▼
③  stagnation → hypothesis  "does a *run* of flat results change what's tried next"
```

① has to exist before ② can be tested (nothing to surface). ② has to exist
before ③ can fire at all — ③ hooks the same `stagnant` signal ② computes, and
before ③'s exit criterion 3 can be checked ("a campaign with history picks a
different first move") — that check reads the same observe bundle. A single
bad experiment already mints a follow-up hypothesis today (§1.3); ③ is
narrower than an earlier draft of this design scoped it, because that part of
the gap turned out to already be closed.

---

## 3. Component 1 — the score writer

**Where it hooks.** `_record_experiment_outcome`
([loop.py:82](../../../../src/labpilot/research_engine/conductor/loop.py#L82)),
called from both the success and exception paths around `_EXPERIMENT_TOOLS`
dispatch ([loop.py:630](../../../../src/labpilot/research_engine/conductor/loop.py#L630),
[loop.py:647](../../../../src/labpilot/research_engine/conductor/loop.py#L647)).
This is already the single funnel for "an experiment tool finished" — extend
its signature rather than add a second call site, or the two will drift the
way the breaker counters didn't.

**Where the value comes from.** `run_plan`'s `ToolResult.data`
([run.py:32](../../../../src/labpilot/research_engine/tools/handlers/run.py#L32))
carries `execution_id`/`status`/`error`, not metrics. The metrics live on the
execution artifact; `metrics_as_experiment`
([builder.py:527](../../../../src/labpilot/research_engine/evidence/builder.py#L527))
already assembles an `Experiment`-shaped view from a raw metrics dict for
exactly this kind of downstream consumption — reuse it rather than re-reading
`metrics.json` a fifth way.

Traced concretely, the same way Component 2's new parameter is traced back to
`loop.py`'s local scope: the raw dict `metrics_as_experiment` needs comes from
`workspace.root / "metrics.json"`
([outcome.py:302](../../../../src/labpilot/research_engine/execution/outcome.py#L302)
shows the same read) — `workspace` is already in scope at the
`_EXPERIMENT_TOOLS` call sites, and this is the very file the plan's §1
"metrics.json is overwritten in place" observation is about, so the writer's
read and that overwrite race are the same file by construction.
`hypothesis_id` and `technique`/`combo_techniques` are not on `ToolResult` or
on the assembled `Experiment` either — `run_plan`'s `step_args`
([loop.py:565](../../../../src/labpilot/research_engine/conductor/loop.py#L565),
already local at the call site as `task.args`) carries `plan_id`;
`PlanArtifacts(...).get(plan_id)` gives `Plan.hypothesis_id`
([models.py:59](../../../../src/labpilot/research_engine/planner/schemas/models.py#L59));
`HypothesisStore(...).get(hypothesis_id)`
([hypothesis.py:181](../../../../src/labpilot/research_engine/shared/experiments/hypothesis.py#L181))
gives the `Hypothesis`, and its `technique`/`combo_techniques` populate the
event (see §3's `ScoreEvent` fields below — `combo_techniques`, not
`technique_stack`; the distinction matters, see the correction after the
model).

**Key resolution.** Use `comparator.resolve_primary_metric_key_and_direction`
— it is the one resolver in the table above that is both competition-aware
and returns direction, which the writer needs to know whether a delta is an
improvement. Do **not** add a new resolver. Consolidating the other three into
callers of this one is real cleanup but out of scope here — flagged in §6.

Its signature is built for an A/B pair (`base`, `compare_exp`) and intersects
their metric keys; the writer only ever has **one** finished execution to
record. Call it with the same `Experiment` on both sides —
`resolve_primary_metric_key_and_direction(exp, exp, competition_dirs=...)`.
`shared = set(base.metrics) & set(compare_exp.metrics)` then degenerates to
`exp.metrics`'s own keys, which is exactly the lookup the writer needs; the
function's `competition.json`-driven preference for `cv_<spec.key>` doesn't
depend on there being a second run, so this is identical behavior on a
campaign's first (baseline) event and every later one. Note the call site,
don't leave it implicit — this exact kind of unstated adaptation between a
comparator API and a single-value lookup is how the four resolvers in the
table above ended up disagreeing in the first place.

**What gets written, and where.** Per the plan: `(experiment_id,
hypothesis_id, technique, metric_name, value, timestamp)`. `BudgetState`
already has `metric_history: list[float]` and `last_metric: float | None` —
those stay as the plateau/target detector's cheap flat view, but a flat float
list cannot carry `experiment_id`/`hypothesis_id`/`technique`, which exit
criterion 1 needs (a hypothesis citing a prior *experiment's* result by id).

Proposed: a new `ScoreEvent` row, appended once per successful experiment
outcome, held in a new `BudgetState.score_events: list[ScoreEvent]` field —
on `BudgetState` itself, not a sibling structure in session metadata, so it
serializes through the existing `budgets_to_metadata`/`budgets_from_metadata`
path ([budgets.py:181](../../../../src/labpilot/research_engine/conductor/budgets.py#L181))
with no new metadata key to name, and so [M17](../12-run-until-done.md)'s
`goal_progress(config, state)` — which already receives a `BudgetState` — can
reach the series without a second parameter (§4 relies on this). `BudgetState.
metric_history`/`.last_metric` are then **derived** from `score_events` on
write (`last_metric = score_events[-1].value`, `metric_history =
[e.value for e in score_events]`) — recompute, not step, per
[AGENTS.md](../../../../AGENTS.md)'s rule 2 (the `apply_card_to_beliefs`
lesson: a stepped counter is wrong forever once one input turns out to have
been wrong).

```python
class ScoreEvent(BaseModel):
    experiment_id: str
    hypothesis_id: str | None
    technique: str | None              # single-technique experiments
    combo_techniques: list[str] = []   # combo experiments (M19 §5's shape)
    metric_name: str          # normalized key, e.g. "cv_rmse" — not raw
    value: float
    maximize: bool            # from the same resolver call, so the sign is never re-derived
    timestamp: str
```

`technique` alone loses information for a combo experiment — but the field
that fixes it is `combo_techniques`
([models.py:83](../../../../src/labpilot/research_engine/shared/experiments/models.py#L83)),
not `technique_stack`
([models.py:81](../../../../src/labpilot/research_engine/shared/experiments/models.py#L81)).
Both live only on `Hypothesis` — `Experiment` has neither, which is why §3's
data-path paragraph above traces `hypothesis_id` back to a `Hypothesis`
lookup rather than reading either off the assembled `Experiment`. The two
`Hypothesis` fields mean different things and an earlier draft of this
section conflated them: `technique_stack`'s own docstring scopes it to
*cumulative lineage* — "techniques already assumed in the pipeline (parent
stack + this change)" — so a 5-generation single-technique chain would carry
`[A,B,C,D,E]` there even though only one new technique was tested this time.
`combo_techniques`'s docstring is scoped to exactly what this design needs —
"techniques applied together in a combination experiment (size 2-3)" — and is
the field Component 3's own worked example (`[target_encoding, mixup]`,
"unclear which of the two caused it") actually describes. Using lineage where
combo was meant reintroduces a milder version of the same misattribution M19
§5 fixed: it would spread blame for a one-technique regression across every
ancestor in the chain instead of naming the one thing that changed.

Only a **successful** execution with a resolvable primary metric appends an
event — a placeholder/stub run (`is_placeholder_metrics`,
[builder.py:42](../../../../src/labpilot/research_engine/evidence/builder.py#L42))
must not enter the series, for the same reason it must not enter an evidence
card. If the key can't be resolved (no shared/`competition.json` key —
distinct from the placeholder case), skip the append but log the reason at
`logger.info`, the way `should_gather_evidence`'s caller already logs its skip reason
([policy.py:520](../../../../src/labpilot/research_engine/conductor/policy.py#L520))
— a silent skip here is undiagnosable from outside, and "the gate exists but
its input was wrong" is the exact failure shape [AGENTS.md](../../../../AGENTS.md)'s
rule 3 warns about.

**Scope: local metric only, deliberately.** `run_plan`/`run_experiment`
produce a local CV score; `submit`/`submit_learn` produce a separate
leaderboard score on a different scale, and `submit`/`submit_learn` are not
`_EXPERIMENT_TOOLS` — they never reach this hook. That's intentional, not an
oversight: `evaluate_stops` has always compared `target_value` against the
*local* metric, and a campaign iterates on local CV between submissions,
which are comparatively rare and expensive. Mixing a local-CV `ScoreEvent`
and a public-LB `ScoreEvent` in the same series without a
`metric_name`/`maximize` distinction would be the metric-key-drift bug from
§1.2 in a new costume — comparing across metric *kinds*, not just naming
variants of the same one. The leaderboard number is not unhandled, just
handled by a different, already-working path: `submit_learn` already writes
it to `Hypothesis.public_score` and folds it into `HypothesisStatus`
(see §5's correction) — this writer does not need to duplicate that.

**Series growth.** Unlike `BudgetState.recent_failures` — explicitly bounded
to the last two entries because it is "a stop *reason*, not a log" — the
`ScoreEvent` series is not truncated. Exit criterion 1 needs to cite an
arbitrary prior experiment by id, so evicting old events would silently break
citations to anything before the cutoff. As a `BudgetState` field, the series
lives in the same JSON metadata blob `BudgetState` already does, rewritten
whole on every append
([store.py](../../../../src/labpilot/research_engine/conductor/store.py) —
`update_session_metadata`); at one row per successful experiment, and each
experiment already costing real training/LLM time, this is a few hundred
small JSON rows even on a long M17 "run until plateau" campaign, not a
performance risk. If a campaign's step count ever makes the whole-blob
rewrite measurable, the fix is a proper indexed table for `ScoreEvent` (an
`INSERT`, not a blob rewrite) — not truncation, which would take exit
criterion 1 down with it.

**Recompute hook.** `persist_budgets`
([checkpoint.py:48](../../../../src/labpilot/research_engine/conductor/checkpoint.py#L48))
is the one place that already turns `(config, state)` into session metadata
via `budgets_to_metadata`; it recomputes `metric_history`/`last_metric` from
the current `ScoreEvent` series immediately before that call, so both stay
derived rather than stepped.

---

## 4. Component 2 — score history in the observe bundle

`build_observe_bundle`
([policy.py:46](../../../../src/labpilot/research_engine/conductor/policy.py#L46))
already has the pattern to follow: `viable_hypotheses` /
`untested_hypotheses` are one store read, attached under names that say
exactly what they hold, with the naming-collision history spelled out in a
comment right above the assignment
([policy.py:102](../../../../src/labpilot/research_engine/conductor/policy.py#L102))
— the plan's part 2 is the same mechanism, applied to `ScoreEvent`s instead of
the hypothesis pool. Add, computed from the same series read once:

- `best_so_far` — max/min over the series per the stored `maximize` flag
- `last_3_scores` — tail of the series, most recent last
- `delta_vs_best` — how far the latest reading sits behind the record, read
  the same way whichever direction the metric runs. Since `best_so_far`
  includes the latest, it is `0.0` at a record and negative behind one, and
  **never positive** — an earlier draft of this line promised the opposite,
  which the implementation cannot produce. "Did the latest run improve" is
  `steps_since_improvement == 0`
- `steps_since_improvement` — **count of `ScoreEvent` entries** (completed
  experiments), not conductor steps, since the last one whose `delta_vs_best`
  exceeded `plateau_epsilon`. Named to sit next to `BudgetState.steps_since_success`,
  which already draws this exact line for a different pair — that field's own
  docstring distinguishes it from `consecutive_failures` because "a campaign
  can burn steps without ever reaching an execution"; `steps_since_improvement`
  needs the same explicit scope so it isn't read as counting `reflect` /
  `analyze_competition` steps that never touch the series. A campaign that
  spends ten steps reflecting between two experiments must still report `1`,
  not `11`.

These four are computed by one function, `score_summary(state, config) ->
ScoreSummary`, not inlined in `build_observe_bundle`. [M17](../12-run-until-done.md)
needs the same numbers — its CLI line already reports "best," "target,"
"percent closed," and plateau state, and its own plan names `goal_progress(config,
state)` as "a known-good shape" validated by a prototype
([12-run-until-done.md:92](../12-run-until-done.md)) — building the numbers
twice, once per milestone, is exactly the kind of drift §1 documents for the
primary-metric key. `score_summary(state, config)` is written to match that
already-validated `(state, config)` shape exactly (argument order aside) so
M17's `goal_progress` can call it directly rather than reinvent it — which is
also why §3 puts the `ScoreEvent` series *on* `BudgetState` itself rather
than a sibling structure: `goal_progress` already takes a `BudgetState`, and
a shared function needs the series reachable from the one object both
consumers already receive.

**The exit criterion is not "the field exists," it's "a decision changes,"**
and the mechanism matters more than the doc originally said. The plan's own
trap calls this out: the metric has been in `evaluate_stops` since M3 and
changed no decision in nine campaigns. `decide_next`
([policy.py:692](../../../../src/labpilot/research_engine/conductor/policy.py#L692))
is LLM-driven (`llm_next_action`,
[policy.py:394](../../../../src/labpilot/research_engine/conductor/policy.py#L394)):
putting `steps_since_improvement` only in the observe bundle and hoping the
model reasons about it is the same bet that already lost for nine campaigns —
a number present but unread. `should_gather_evidence`
([policy.py:585](../../../../src/labpilot/research_engine/conductor/policy.py#L585))
is the right model to copy, but *because of how it's consumed*, not because
of its OR-gate shape: it isn't read by the LLM at all. `available_tools`
([policy.py:495](../../../../src/labpilot/research_engine/conductor/policy.py#L495))
calls it at [policy.py:518](../../../../src/labpilot/research_engine/conductor/policy.py#L518)
to remove tools from the allowlist *before the prompt is built* — a
deterministic, testable gate, not a hint inside the prompt.

Commit to the same shape: extend `should_gather_evidence`'s independent-OR
gate (`thin` OR `stale`) with a third clause, `stagnant` —
`steps_since_improvement >= config.plateau_window` also returns
`(True, "N experiments with no improvement")`. This reuses the existing
justification for reopening evidence gathering ("a queue of stale ideas is
the strongest reason to go and find better ones") for exactly the case this
milestone adds: a backlog that keeps producing hypotheses that don't move the
score is indistinguishable, at the tool-allowlist level, from a thin or stale
one. Reusing `plateau_window` (default 3) rather than a new threshold is a
deliberate choice, not an unexamined one: `evaluate_stops` already uses it to
mean "this many flat experiments is enough to stop the campaign"
([budgets.py:153-159](../../../../src/labpilot/research_engine/conductor/budgets.py#L153)).
The two are not the same formula, so "strictly weaker" is a claim about
intent, not a proven subset relationship worth overstating: `plateau` checks
near-literal flatness of the last `n` raw values
(`max(hist[-n:]) - min(hist[-n:]) <= plateau_epsilon`, with a default epsilon
tight enough that this has fired on essentially nothing in nine campaigns,
per the plan's own trap), while `stagnant` checks steps since the last event
that beat the *global* best. A series oscillating below best without
converging (100, 90, 95, 85 against a best of 100) makes `stagnant` fire
while `plateau` structurally cannot.

**Measured when M8-5 shipped, because this paragraph asked for it.** The
"easier bar" reading is wrong on a flat series, and in the one direction that
matters: `steps_since_improvement` counts transitions since the last record
while `plateau_window` counts readings, so with `plateau_window=3` a
perfectly flat series stops on `plateau` at three readings while the gate
would only open at four. `evaluate_stops` runs at the top of the step and
breaks the loop, so on that path `decide_next` is never reached and the gate
cannot act at all.

It was left that way rather than lowered. `plateau` needs a spread within
`plateau_epsilon` (1e-6) — near-exact ties that real CV scores do not
produce, which is why it has fired on essentially nothing in nine campaigns.
On the realistic drifting-worse series `plateau` never fires and `stagnant`
is the only signal. Firing earlier would mean gathering after a single
non-improving experiment, to serve a case that does not occur.

**Plumbing.** `should_gather_evidence` gains a second parameter,
`budget_state: BudgetState | None = None`, defaulted so its 8 existing
1-arg call sites (`policy.py:518` in production, plus 5 in
`tests/unit/test_conductor.py` and 2 in `tests/unit/test_campaign_harness.py`)
keep passing unchanged — `None` skips the `stagnant` clause and preserves
current behavior exactly. `available_tools`
([policy.py:495](../../../../src/labpilot/research_engine/conductor/policy.py#L495))
needs the *same* defaulted treatment independently, for its own separate set
of 2-arg call sites — `tests/unit/test_implement_is_gated_and_honest.py`
(three call sites) and `tests/helpers/campaign_harness.py:374` call
`available_tools` directly, not `should_gather_evidence`; an earlier draft of
this paragraph conflated the two functions' call-site lists. `available_tools`
gains the same `budget_state: BudgetState | None = None` parameter and
threads it through at its one call to `should_gather_evidence`
([policy.py:518](../../../../src/labpilot/research_engine/conductor/policy.py#L518)).
`available_tools` is itself only called from inside `decide_next`
([policy.py:692](../../../../src/labpilot/research_engine/conductor/policy.py#L692),
calling `available_tools` ~18 lines in), so `decide_next` needs the same new
parameter too, one layer further out — it doesn't call `should_gather_evidence`
directly, but it does need to forward `budget_state` down to the
`available_tools` call it already makes. `decide_next`'s own two production
call sites in the run loop
([loop.py:445](../../../../src/labpilot/research_engine/conductor/loop.py#L445),
[loop.py:662](../../../../src/labpilot/research_engine/conductor/loop.py#L662))
already load `budget_cfg`/`budget_state` into local scope for other reasons —
`loop.py:445` sits right after `budget_state.steps_since_success += 1;
persist_budgets(...)` — so both are passing a value already at hand, not
fetching a new one.

This makes exit criterion 2 a pure code test — `available_tools(workspace,
allowlist, budget_state=stagnant_state)` differs from `available_tools(workspace,
allowlist)` (or the same call with a fresh `budget_state`) — with no LLM
sampling involved, which the original "the chosen `ResearchAction` must
differ" framing did not guarantee. §7 and §8 below are written against this
signature.

---

## 5. Component 3 — stagnation → hypothesis

**Revised from an earlier draft.** This section originally proposed hooking
`run_reflection` to mint a hypothesis on any single regression. §1.3 found
that `maybe_mint_improvement_hypothesis`
([outcome.py:580](../../../../src/labpilot/research_engine/execution/outcome.py#L580))
already does exactly that, wired and running today. Rebuilding it would be
the surgical-change rule broken in the other direction — touching code that
isn't broken. What's below is the part that's actually still missing: a
mint triggered by *several* experiments each individually clearing
`maybe_mint_improvement_hypothesis`'s per-execution bar but the campaign
still not moving.

**Why this can't live where the existing mechanism lives.**
`maybe_mint_improvement_hypothesis` and its callers
(`update_hypothesis_from_local`, `record_successful_execution`, `submit_learn`)
live in the execution/outcome layer — they take `knowledge_dir`/`competition`/
an `ExecutionOutcomeSummary`, with no `ConductorStore` or `BudgetState` in
scope, and comparing "this execution vs. its immediate parent" doesn't need
either. Detecting a five-experiment plateau *does* need the `ScoreEvent`
series, which lives on `BudgetState` (§3) — a conductor-session object this
layer doesn't have and, per the plan's own layering, shouldn't be given just
for this. This mint belongs in the conductor loop instead, next to where the
series is already being read.

**Where it hooks.** In `loop.py`, immediately after `_record_experiment_outcome`
appends a `ScoreEvent` (§3) and `score_summary` is recomputed (§4) — the same
`stagnant` condition Component 2 uses to reopen evidence gathering
(`steps_since_improvement >= config.plateau_window`) also triggers this mint.
Reusing the identical threshold means the two respond to the same event: a
campaign becomes eligible to look for new evidence and to get an explicit
"here's a candidate, stop repeating the pattern" hypothesis at the same
moment, not on two different schedules that can drift apart. `Workspace`
([workspace_facade.py:26](../../../../src/labpilot/research_engine/workspace_facade.py#L26))
already carries `.knowledge_dir`/`.competition`, and the technique sourcing
goes through `generate_candidates`, which takes no `llm_client` at all — so,
unlike `maybe_mint_combo_from_success`'s `llm_client`-or-raise risk
(`update_hypothesis_from_local`'s own docstring warns about this for a
different function), this mint has no LLM dependency to thread down from the
loop and nothing new to configure at this hook.

`evaluate_stops` reads `budget_state` at the *top* of each step
(`loop.py:388`, before this step's dispatch), while this mint fires *after*
`_record_experiment_outcome` appends the new event, later in the same
iteration — so a campaign-ending `plateau` stop reacting to this step's new
data can only be observed at the *next* step's `evaluate_stops` call, one
full step after this mint already ran. The mint is therefore always
sequenced before any stop that could react to the same data, but that's a
consequence of the loop's existing read-then-append ordering, not something
Component 3 arranges — worth stating plainly rather than leaving an
implementer to re-derive it from the loop's control flow.

**Edge-triggered, not level-triggered.** Fire only on the step where
`stagnant` flips `False → True`, not on every subsequent step it stays `True`
— `steps_since_improvement` only grows while a campaign is stuck, so a
level-triggered mint would fire every remaining step of a long plateau,
flooding the backlog with near-duplicate hypotheses (`_already_covered_by_proposed`,
which `maybe_mint_improvement_hypothesis` already uses for its own dedup,
guards content but not this timing). Track this the same way
`BudgetState.consecutive_failures` resets on the first success
([budgets.py:83-89](../../../../src/labpilot/research_engine/conductor/budgets.py#L83)):
a `stagnation_mint_fired: bool` (or equivalent) on `BudgetState` — named for
what's true while it's set (a mint already fired *this* plateau, suppress
further ones), not for an action still to come, since "pending" reads as the
opposite of a suppression latch. Set when the edge fires, cleared on the next
real improvement, so a second plateau later in the same campaign mints again
rather than staying silently suppressed forever.

**Satisfying exit criterion 1 for the multi-experiment case.** (The
single-experiment case is already satisfied today, per §1.3 — this is the
part M8 actually adds.) The reason string cites every experiment in the
stagnant window, not one: `"E-040 (target_encoding), E-041 (mixup), E-042
(feature_interactions) — none improved best_so_far=194.80, last set at E-037;
propose isolating validation-scheme changes instead of another feature
technique"`. `evidence` carries one `HypothesisEvidenceRef(kind=
HypothesisOrigin.EXPERIMENT, ref=...)` per experiment in the window, not just
the most recent — a reader checking "did this cite its evidence" should be
able to resolve every id, not just the last one.

Two cases the worked example doesn't cover, same shape as the prior draft but
now over the window rather than one experiment:

- **Combo regression inside the window** (`ScoreEvent.combo_techniques`
  non-empty for one of the window's events): cite the full combination for
  that event, not one name from it — do not pick one technique out of a
  combo to blame; that's the misattribution M19 §5 fixed for evidence cards.
- **No clean technique label for the whole window** (every event's
  `technique`/`combo_techniques` empty — config/data-only deltas throughout):
  cite the experiment ids and the flat scores, but don't fabricate a
  technique to propose against. Fall through to `recommend_next_experiment`'s
  existing `analyze_or_hypothesize` action instead.
- **Mixed window** (some events single-technique, others combo): the
  substitute-technique exclusion set passed to `generate_candidates` is every
  individual technique named anywhere in the window — combo members
  included, not just whole combos — so a technique already tried as part of
  one experiment's combination isn't re-proposed alone in the next
  hypothesis as if it were untested.

Technique sourcing for the proposal reuses `generate_candidates`'s existing
substitute-technique reasoning
([candidates.py:114](../../../../src/labpilot/research_engine/intelligence/hypothesis/candidates.py#L114)),
scoped to exclude the techniques already present anywhere in the stagnant
window — the same "don't re-propose what was just tried" logic
`tried_techniques` already gives it, just fed the window's techniques instead
of the whole campaign's.

`created_by=HypothesisCreatedBy.REFLECTION`, `source="reflection"`,
`origin=HypothesisOrigin.EXPERIMENT` — the same enum values
`maybe_mint_improvement_hypothesis` already uses for the single-experiment
case, kept consistent rather than inventing a parallel vocabulary for "mint
triggered by a window" vs. "mint triggered by one execution."

**Guardrail.** M18's review already found the failure mode this kind of write
path can cause: consumer filters turned `dormant` into a closed loop and
planner-visible techniques dropped 116 → 1
([13-technique-vocabulary.md §"What the review of the first attempt
caught"](../13-technique-vocabulary.md)). To be precise about which status
this applies to — `HypothesisStatus` (`PROPOSED | TESTING | CONFIRMED |
REJECTED | INCONCLUSIVE`, defaulting to `PROPOSED`) has no `candidate` member;
`candidate` is a status on the *technique vocabulary*, not on `Hypothesis`.
The guardrail is: the **technique this hypothesis names** must reach the
planner through the vocabulary's own `candidate` state via
`recompute_technique_status()` the same way every other technique does — a
stagnation-authored hypothesis must not name a technique that skips that
recompute just because the hypothesis itself is new. The `Hypothesis` row is
created with `status=PROPOSED` like any other; nothing here changes
`HypothesisStatus`'s meaning.

---

## 6. Design choices & tradeoffs

| Choice | Rejected alternative | Why |
|---|---|---|
| `ScoreEvent` series as a `BudgetState` field, `metric_history`/`last_metric` derived from it | Write `metric_history`/`last_metric` directly, no new type | Can't carry `experiment_id`/`hypothesis_id`/`technique` — exit criterion 1 needs the id, and a flat float list has nowhere to put it |
| Reuse `comparator.resolve_primary_metric_key_and_direction` | Write a fifth metric-key resolver scoped to the writer | The four existing resolvers disagreeing is the measured −194.30 defect in §1; a fifth one is the same mistake with better intentions |
| Scope Component 3 to *stagnation* (a window), not single-execution regression | Rebuild a `run_reflection`-hooked single-regression mint as originally drafted | `maybe_mint_improvement_hypothesis` already does the single-execution case, wired and running (§1.3) — rebuilding it duplicates working code instead of touching the actual gap |
| Hook Component 3 in the conductor loop (`loop.py`, next to §3/§4) | Hook it inside `record_successful_execution` / `update_hypothesis_from_local` (execution/outcome layer), next to the existing single-execution mint | That layer has no `BudgetState`/`ScoreEvent` series in scope, and giving it one just for this crosses a layer boundary the existing code deliberately doesn't cross |
| Edge-triggered mint (fires once when `stagnant` flips `False→True`) | Level-triggered (fires every step `stagnant` stays `True`) | A long plateau would otherwise mint a near-duplicate hypothesis every single step, flooding the backlog the way M18's ungated miner did |
| Leave the 4-resolver consolidation out of scope | Fold it into this milestone | Real cleanup, but touches `outcome.py`/`builder.py` call sites this milestone doesn't otherwise need to change — a surgical-change violation for a design that's already touching three subsystems |
| Gate `steps_since_improvement` through `should_gather_evidence` → `available_tools` (allowlist, pre-prompt) | Put it in the observe bundle only and rely on `decide_next`'s LLM prompt to act on it | `decide_next` is LLM-driven; a number the model *might* reason about is the same bet that already left `evaluate_stops`'s metric unread for nine campaigns. An allowlist gate is deterministic and testable without any LLM call |
| `resolve_primary_metric_key_and_direction(exp, exp, ...)` — same `Experiment` on both sides | Write a single-`Experiment` overload, or a new lookup function | The two-sided call degenerates correctly (`shared` reduces to `exp.metrics`'s own keys) and needs no new code; a new overload is one more resolver-shaped thing to keep in sync with the other four |
| One shared `score_summary(state, config)` — matching `goal_progress`'s own validated shape — for M8's observe bundle and M17's `goal_progress` | Let M17 compute its own version when it lands | Both need `best_so_far`/`steps_since_improvement` from the same series; computing them twice is the resolver-drift pattern from §1, one milestone later |

---

## 7. Exit criteria (from the plan, made checkable)

1. **A campaign log in which a hypothesis's `reason` field cites a prior
   experiment's result by id.** Already satisfiable today via
   `maybe_mint_improvement_hypothesis` for a single execution's regression
   (§1.3) — check: `grep` a real campaign's `hypotheses.jsonl` for a
   `created_by=reflection` row whose `reason` string contains an `E-\d+`-shaped
   id that resolves to an existing execution. What M8 adds is the same check
   holding for a *window*: a `hypotheses.jsonl` row produced by Component 3
   whose `evidence` list resolves to more than one prior execution id, not
   just the most recent.
2. **`steps_since_improvement` visible in observe, and a policy decision that
   demonstrably changes when it is high.** Check:
   `available_tools(workspace, allowlist, budget_state=stagnant_state)` where
   `stagnant_state.score_events` puts `steps_since_improvement` past
   `config.plateau_window` must differ from
   `available_tools(workspace, allowlist, budget_state=fresh_state)`
   (`analyze_competition` present vs. absent) — a pure code assertion, no LLM
   involved. This is deliberately a stronger check than "the chosen
   `ResearchAction` differs": that would only prove the *offline* fallback
   path reacted, or require asserting on LLM output, neither of which is what
   `evaluate_stops`'s nine-campaign failure actually needed fixed.
3. **Re-running a campaign on a workspace with history picks a different
   first move than on an empty one.** Check: same workspace snapshot, once
   with `budget_state` cleared, once with a real `ScoreEvent` series —
   `decide_next`'s first action must differ.

---

## 8. Testing strategy

Per [AGENTS.md](../../../../AGENTS.md) rule 4 — prove the test fails without
the fix:

- **Writer:** assert `metric_history` is non-empty *and* the compared-in
  isn't a placeholder — a test that only checks "the field didn't error"
  would pass on an unwritten empty list, the exact vacuous-pass shape the
  rule warns about.
- **Metric-key drift:** a regression test seeding a `cv_accuracy` stub result
  next to a `cv_rmse` real one, asserting the writer refuses to compare them
  (mirrors the fix `_primary_cv_keyed` already made for the evidence-card
  path — the writer must not reopen the hole that one closed).
- **Observe → decision:** the exit-criterion-2 check above (`available_tools`
  diff), run as an actual test rather than eyeballed once. Because the gate
  lives in `should_gather_evidence`/`available_tools` rather than the LLM
  prompt, this test fails before the fix (the parameter doesn't exist / the
  clause isn't there) and passes after by construction — it does not inherit
  LLM-sampling flakiness the way asserting on `decide_next`'s chosen action
  would have.
- **Stagnation → hypothesis:** seed a `BudgetState` with a `score_events`
  window that crosses `plateau_window` with no improvement, run the loop step
  that would append the next `ScoreEvent`, and assert a `Hypothesis` row
  exists afterward with `created_by=reflection` and a `reason`/`evidence`
  that resolve to every execution id in the window — not just that the mint
  function returned without raising. A second test asserts the edge-trigger:
  the same stagnant state held for two more steps must **not** mint a second
  hypothesis, only the first transition does.
- **Existing single-execution mint unaffected:** a regression test that
  `maybe_mint_improvement_hypothesis`'s own behavior (and its test suite) is
  unchanged by Component 3 landing — the two are independent gates on the
  same `HypothesisStore.create()`, and this milestone must not be the one
  that quietly changes what the already-shipped path does.
