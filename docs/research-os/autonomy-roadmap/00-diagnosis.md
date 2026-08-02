# Diagnosis — why the loop cannot optimise

**Date:** 2026-08-02 · **Method:** nine `research conduct` campaigns against
`rogii-wellbore-geology-prediction`, fixing each blocker as it surfaced.

---

## Symptom

The Conductor runs. It gathers evidence, proposes hypotheses, builds plans
against them, trains real models, reflects, and loops. Then it stops with the
goal 39× away, or cycles producing the same number forever.

## Root cause 1 — the hypothesis never reaches the model

`technique` is recorded on the plan, the hypothesis is marked `testing`, the run
executes, reflection files a result. Meanwhile codegen falls back to
`tabular_regression_partitioned`, which has no concept of a technique.

Every experiment therefore produces **MSE 194.80**. The system has twelve
hypotheses and exactly **one reachable model state**.

This is why it cannot "keep optimising": there is nothing to optimise over. The
loop turns and the wheel never touches the road.

→ [01-technique-to-model.md](01-technique-to-model.md)

## Root cause 2 — no objective feedback

`evaluate_stops` reads a metric to decide *when to quit*. Nothing feeds score
history into the *decision about what to try next*. The policy's observe bundle
never contains "your last three experiments scored 194.8, 194.8, 194.8".

So the Conductor optimises **process completion** — "have I used each tool?" —
which is precisely why it declared victory at step 4 of 12 with:

> "All tasks related to analyzing the competition, generating a plan, and
> running the plan have been completed. There is no immediate next step
> outlined in the allowlist."

Claude Code works because `Edit` really edits and tests really fail. The
equivalent ground truth here is the CV score, and it was wired to nothing that
decides.

→ [02-objective-loop.md](02-objective-loop.md)

## Root cause 3 — silent success at every layer

Six distinct instances found in one day, each reporting success while doing
nothing:

| Layer | Reported | Actually |
|---|---|---|
| Profiler | success | 0 rows / 0 columns → wrong target, wrong baseline |
| Codegen | success | fake metrics (`cv_accuracy: 0.0` on a regression task) and a submission with the wrong header |
| Micro agents | success | LLM output discarded, rule engine used instead |
| `run_experiment` | completed in 0.9s | rendered code, skipped training entirely |
| Validation | `cv_mse 3892` | shuffled split across 773 wells — the number measured nothing |
| Test suite | 608 passing | silently making real network calls; machine-dependent |

Every layer above these reported success. The system was structurally incapable
of telling the operator it was broken.

The **mechanism** is a design decision, not an accident: 20 micro agents each
carry a `_run_rule_engine` fallback, and `BaseMicroAgent` catches any LLM or
parse failure and quietly uses it. The system runs deterministic rules while
looking like it is reasoning.

→ [03-verification-first.md](03-verification-first.md) ·
[09-llm-required.md](09-llm-required.md)

## Root cause 4 — the policy matches keywords, it does not reason

The Conductor is described as continuously deciding what happens next. In
practice an LLM picks one tool *name*, and `map_research_action` then matches the
intent string against keyword tuples to select a hardcoded chain. There is no
model of where the research is or which transition is worth making.

That is why it chose `generate_plan` on five consecutive steps without running
any of them, and why "have I used each tool once?" reads to it as success.

→ [08-policy-reasoning.md](08-policy-reasoning.md)

## Root cause 5 — the capability layer is hollow

Of ten catalog tools, **one** (`run_plan`) can move the score, and it has exactly
one reachable configuration. `implement()` is named like an action and renders a
fixed template. `reflect()` is real but nothing it writes feeds a later decision.

A named tool implies a capability. The control plane grew rich enough to decide
"try a CNN" while nothing underneath could produce one.

→ [10-capability-audit.md](10-capability-audit.md)

## Root cause 6 — the substrate cannot deliver the reasoning

The architecture assumes Claude-Code-grade reasoning at three points: conductor
policy, hypothesis generation, code generation. The local `qwen2.5-coder:14b`
returned **English prose** where the analyzer required JSON:

> `'These rules outline the guidelines and conditions for participating in a
> Kaggle competition. Here are some key points: 1. **Competition Data**...'`

Because every micro agent catches a parse failure and falls back to its rule
engine, this failed invisibly — the LLM was paid for and thrown away. Codegen
produced no usable training code at all, which is the only reason a template
fallback exists.

→ [04-llm-tiering.md](04-llm-tiering.md)

---

## The generalisation

> Every milestone shipped its structure but not its function.

- Memory has four layers — nothing reads them to change a model.
- The event bus exists — no agent reacts to `ExperimentCompleted`.
- M5 shipped parallel workers — campaigns run strictly sequential.
- `implement()` is in the tool catalog — it renders a fixed template.
- Hypotheses, beliefs, claims, techniques all persist — and never alter an outcome.

The architecture is **not wrong**. Conductor, tools, agents, memory, event bus,
workspace are the right decomposition and are well built. What is missing is
that almost nothing downstream of a decision can *act differently* as a result
of it.

## What this predicts

Any future milestone that adds a store, a registry, or an agent type will
produce the same outcome unless it ships with a check that an **outcome
changed**. That is why every plan here carries an exit criterion phrased as an
observable difference, never as "component X exists".
