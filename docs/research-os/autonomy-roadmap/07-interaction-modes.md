# Interaction modes — auto / accept-edits / plan

**Status:** not started · **Blocked by:** M7 (modes are meaningless while every
path leads to the same model)

---

## Purpose

Researchers will not hand a system four hours of compute on trust. They need the
same range Claude Code offers: watch it plan, approve each change, or let it
run.

## Goal

Three modes over **one loop**, not three systems.

| Mode | Behaviour |
|---|---|
| **Plan** | Run the policy, render the task DAG and rationale, dispatch nothing |
| **Accept-edits** | Gate the tools that change the world (`implement`, `submit`); everything else proceeds |
| **Auto** | Gate nothing; run to budget or goal |

## Approach

The seam already exists and is load-bearing today:

- `--autonomy 0|1` — `0` gates plan batches *and* submits, `1` gates submits only
- `maybe_approve(...)` in `conductor/approvals.py` — the gate itself
- `--yes` — auto-approve
- `DecisionRecord` — already carries rationale and the task DAG

So:

- **Plan mode** = existing loop with dispatch suppressed, decisions rendered.
  `decide_next` already returns a rationale; nothing new is required to explain
  intent.
- **Accept-edits** = `--autonomy 1` plus gating `implement`. Once M7 lands,
  `implement` is where a technique becomes a real code change, which is exactly
  the moment a researcher wants a say.
- **Auto** = today's `--yes --autonomy 1`.

## Why deliberately late

Every mode currently produces the same model. Plan mode would render a DAG whose
outcome is known; accept-edits would gate a change that changes nothing. The
modes become meaningful the moment techniques diverge — and trivial to build,
because the gate is already there.

## Exit criteria

1. `research conduct "<goal>" --mode plan` prints the DAG and rationale and
   makes no durable write.
2. `--mode accept-edits` pauses before a technique changes code, showing the
   diff, and honours reject.
3. A rejected step is recorded and influences the next decision — operator
   feedback already flows into `observe`, so this should be verified rather than
   built.

## Traps

- **Do not fork the loop per mode.** Three code paths will diverge and two will
  rot. Mode is a policy over one loop.
- **Plan mode must be genuinely side-effect free.** Analysis currently
  materialises data, writes artifacts and ingests knowledge; "plan only" has to
  mean it. `analyze_without_side_effects` already exists and is the right seam.
- **Reject must teach.** A rejection that does not change the next decision
  trains the operator to stop using the mode.
