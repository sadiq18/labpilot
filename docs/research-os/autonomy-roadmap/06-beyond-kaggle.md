# M12 — Beyond Kaggle

**Status:** not started · **Blocked by:** M7, M8 (the loop must work once)

---

## Purpose

Kaggle is the proving ground, not the product. The thesis is an autonomous
research engineer that happens to start on Kaggle — extending to ML
benchmarking, paper reproduction, hyperparameter optimisation, AutoML,
simulation-backed science, and eventually general software engineering.

That extension is architecturally cheap **only if** the loop is already
goal-driven and tool-driven. It is expensive if Kaggle assumptions are baked
into the control plane.

## Goal

Replace "train a model and score it" with a pluggable
**`HypothesisValidator`** — anything that can turn a hypothesis into a
comparable number.

## Approach

```python
class HypothesisValidator(Protocol):
    """Turns a hypothesis into a comparable result."""

    def validate(self, hypothesis, workspace, context) -> ValidationResult:
        """Returns (score, direction, provenance, artifacts)."""
```

Implementations:

| Domain | Validator | Score |
|---|---|---|
| Kaggle | CV + leaderboard | competition metric |
| ML benchmarking | standard benchmark harness | benchmark metric |
| Paper reproduction | reproduce reported result | agreement with paper |
| Simulation science | run simulator | domain objective |
| Software engineering | test suite + benchmarks | pass rate, latency |

Everything above the validator — Conductor, hypotheses, beliefs, claims,
reflection, memory, context engine — is already domain-neutral and should not
need to change.

## Why deliberately last

**The abstraction will be wrong if designed before the loop works once.** Right
now there is exactly one working validation path (partitioned CV), and it took a
day to make it honest. Generalising from one example that was broken until
yesterday would encode its accidents.

Wait until M7/M8 have produced several genuinely different experiments, then the
shape of `ValidationResult` will be obvious from what those experiments actually
needed.

## Exit criteria

1. A second validator implemented (a benchmark harness is the cheapest) with no
   change to the Conductor, policy, hypothesis or reflection code.
2. A campaign that runs against it end to end.
3. `research conduct "<goal>"` phrasing unchanged between domains.

## Traps

- **Kaggle assumptions already leaking.** `submission_columns`,
  `sample_submission`, `kernel_output_file`, `max_daily_submissions` and
  leaderboard scoring appear in the competition spec and several capabilities.
  These belong to the *Kaggle validator*, not the core.
- **Metric direction is not universal.** `maximize` is a competition property
  today; it must become a property of the `ValidationResult`.
- **Do not build a plugin system first.** One extra validator, hardcoded, will
  reveal the interface. A registry can come after there are three.
