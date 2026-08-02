# M13 — Policy reasons about state, not keywords

**Status:** not started · **Partly mitigated:** precondition filtering shipped

---

## Purpose

The Conductor is advertised as *"continuously decides what should happen next"*.
What it actually does:

1. An LLM picks **one tool name** from an allowlist.
2. `map_research_action` matches the intent string against keyword tuples and
   returns a hardcoded tool chain.

```python
_TEMPLATES = [
    (("paper", "literature", "search", "read"), [ToolStep(tool="search_papers", …)]),
    (("plan", "baseline", "hypothesis"),        [ToolStep(tool="generate_plan", …)]),
    (("experiment", "run", "train", "try"),     [...]),
]
```

A novel intent falls through as "no capability". The word *"try"* routes to a
four-step experiment chain regardless of whether an experiment makes sense.
There is no model of **where the research is** and **what transition is legal or
valuable from here**.

Consequence: the policy optimises "have I used each tool?" — which is exactly
why it declared victory at step 4 of 12 with the target 39× away, and why it
chose `generate_plan` on five consecutive steps without running any of them.

## Goal

The Conductor selects a **state transition**, justified by the current research
state and the score history — not a tool name matched on a word.

## What already shipped (partial mitigation)

`available_tools()` filters tools whose preconditions the workspace does not
satisfy (`reflect` needs an execution, `run_plan` needs a plan, evidence
gathering is gated on backlog + freshness). That converts "the model picked
badly" into "that option was never on the table".

It is a **guardrail, not reasoning**. It prevents illegal moves; it does not
choose a good one.

## Approach

**1. Name the research states explicitly.**

```
UNANALYSED → ANALYSED → HYPOTHESES_QUEUED → PLAN_READY
          → EXPERIMENT_RUN → REFLECTED → {IMPROVED | PLATEAUED | REGRESSED}
```

State is derived from the workspace, not stored — it is already computable from
plans, executions, hypothesis backlog and the score series.

**2. Transitions carry preconditions and expected value.** A transition is legal
(preconditions met) and *rated* (what it is expected to buy given history). This
is where scoring enters tool choice, which today it never does:

| State | Signal | Preferred transition |
|---|---|---|
| `PLATEAUED`, 3 runs no gain | score flat | change technique family, not hyperparameters |
| `REGRESSED` after technique X | score worse | reflect → hypothesis excluding X's family |
| `HYPOTHESES_QUEUED`, backlog thin | — | gather evidence |
| `EXPERIMENT_RUN`, unreflected | — | reflect (cheap, unlocks learning) |

**3. The LLM chooses among legal transitions, with the state and score history
in the prompt.** It stops guessing a tool name and starts answering "given we
have plateaued for three runs, which of these five legal moves is worth it?" —
a question a smaller model can actually answer.

**4. Delete the keyword templates.** Once transitions are explicit,
`_TEMPLATES` is dead weight and its failure mode (silent fall-through to "no
capability") disappears.

## Exit criteria

1. A decision record naming the **state** and the **transition**, not just a tool.
2. Two workspaces in different states, same goal, produce different first moves.
3. A plateaued campaign changes technique *family* rather than repeating the
   same class of experiment.
4. `_TEMPLATES` removed with no loss of behaviour.

## Traps

- **This is worthless before [M7](01-technique-to-model.md) and
  [M8](02-objective-loop.md).** `PLATEAUED` cannot be detected while every
  experiment returns the same score, and "change technique family" means nothing
  while techniques do not reach the model.
- **Do not encode the transition table as a fixed workflow.** The point is a
  *legal move set* the policy chooses within — re-introducing
  Analyze→Plan→Run→Reflect as a hard sequence would undo the milestone.
- **Precondition filtering must remain.** Reasoning picks among legal moves; the
  filter defines legal. Both are needed.

## Related code

- `src/labpilot/research_engine/conductor/actions.py` — `_TEMPLATES`, `map_research_action`
- `src/labpilot/research_engine/conductor/policy.py` — `available_tools`, `decide_next`, `build_observe_bundle`
- `src/labpilot/research_engine/conductor/loop.py` — `_latest_plan_id`, `_next_hypothesis_id`, `_baseline_plan_exists` (state derivation already lives here)
