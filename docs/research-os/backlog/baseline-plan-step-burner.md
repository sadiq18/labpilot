# Backlog — the baseline guard has a hole

**Status:** Backlog · **Kind:** live defect, not a deferred feature ·
**Found:** PR #112 review, 2026-08-08 · **Site:**
[`conductor/actions.py:194-197`](../../../src/labpilot/research_engine/conductor/actions.py#L194)

Deferred out of PR #112 deliberately: the fix is a campaign-policy decision, and
that PR was about delta consistency. Patching campaign policy blind, inside an
unrelated PR, is how the *next* step-burner gets introduced.

## Problem

`resolve_step_args` switches `generate_plan` off `baseline` once a baseline plan
exists — but only when a hypothesis is available to switch *to*:

```python
if tool == "generate_plan" and resolved.get("baseline") and baseline_plan_exists:
    if next_hypothesis_id:
        resolved.pop("baseline", None)
        resolved["hypothesis_id"] = next_hypothesis_id
```

With `baseline_plan_exists=True` and `next_hypothesis_id=None`, nothing is
stripped. The step then runs `generate_plan(baseline=True)`, `compile_baseline_plan`
is idempotent, no new plan appears, and **the step reports success**. The
campaign can select it again next step, and again.

The function's own docstring states the invariant it fails to hold:

> a campaign that only ever asked for a baseline could never mint a second plan
> — and therefore could never run a second experiment

`max_steps` defaults to 8. A campaign that burns steps this way exhausts its
budget having run one experiment, and nothing in the transcript says why: every
step succeeded.

## This is defect #7, fixed once, incompletely

[`evidence-log.md`](../autonomy-roadmap/evidence-log.md) records it from the
2026-08-02 rogii run:

| # | Defect | Fix |
|---|---|---|
| 7 | Campaign could never run a 2nd experiment — `generate_plan` hardcoded `baseline=True`; baseline is idempotent | Resolve to top proposed hypothesis once a baseline exists |

That fix is the code above. It closed the case where a hypothesis exists and left
the case where none does. Same family as the `has_plan` step-burner, and the same
shape as most defects in this codebase: **the guard exists and its condition is
wrong.**

## When `next_hypothesis_id` is None

[`loop.py:105`](../../../src/labpilot/research_engine/conductor/loop.py#L105)
returns the highest-confidence `PROPOSED` hypothesis, or `None` when there are
none. So the hole is open whenever the ledger holds no proposed hypothesis —
early in a campaign, or after every proposal has been tested or rejected. That is
not an exotic state; it is the normal state right after the baseline lands.

## Why it is not a one-line fix

Dropping the flag is not sufficient.
[`handlers/plan.py`](../../../src/labpilot/research_engine/tools/handlers/plan.py)
raises on receiving neither:

```python
if not baseline and not hypothesis_id:
    raise ValueError("pass baseline=True or hypothesis_id")
```

So stripping the flag with nothing to replace it converts a silent loop into a
raised error. That is an improvement — a loud failure beats a quiet one — but
whether the campaign loop handles it gracefully needs a real run to confirm, and
it is probably not the right answer anyway.

| option | effect | cost |
|---|---|---|
| strip the flag regardless | `ValueError`, step fails loudly | campaign stops on a state that is normal, not exceptional |
| leave `baseline=True`, mark the step a no-op | no loop, honest transcript | still spends a step doing nothing |
| **do not select `generate_plan` at all** | conductor proposes a hypothesis first | correct, but the fix moves into policy selection, not arg resolution |

The third is most likely right: with a baseline present and no hypothesis
proposed, the next action is `propose_hypothesis`, not `generate_plan`. That
makes this a defect in **what the Conductor chooses**, and `resolve_step_args` is
only where it becomes visible.

## Measurement

Do not fix this blind — the whole point of the M14 phase 2b discipline is a
number before a behaviour change.

1. Run a rogii campaign at the current default `max_steps` and count steps whose
   tool is `generate_plan(baseline=True)` **after** a baseline plan exists. That
   is the waste rate today.
2. Apply the chosen option.
3. Re-run and count experiments completed per campaign, not steps saved — the
   step count is the symptom, the second experiment is the goal.

## Test to pin it

`tests/unit/test_conductor.py` already covers `resolve_step_args`. The missing
case is exactly the hole: `baseline_plan_exists=True`, `next_hypothesis_id=None`,
asserting whatever the chosen policy is — and asserting it is *not* an
unchanged `baseline=True`, which is the current behaviour.
