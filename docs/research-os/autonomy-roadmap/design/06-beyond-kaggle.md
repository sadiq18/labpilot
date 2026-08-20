# Design — M12: Beyond Kaggle

**Plan:** [../06-beyond-kaggle.md](../06-beyond-kaggle.md) ·
**Status:** design · **Owner:** unassigned · **Build phase:** 0–3

> **Unblocked (2026-08-19).** The plan's blockers were M7 and M8 — *"the loop
> must work once"*. Both shipped. The plan also said to wait until several
> genuinely different experiments existed, *"then the shape of `ValidationResult`
> will be obvious from what those experiments actually needed"*. This document
> is that reading, taken from the code rather than from the table in the plan.

---

## 1. The finding this design rests on

**The validator seam already exists. It is unnamed, and its three parts are
assembled from Kaggle sources at three different call sites.**

`build_evidence_card` — the single funnel every verdict passes through — takes
exactly the triple the plan calls a `ValidationResult`:

```python
build_evidence_card(
    treatment_metrics: dict[str, Any],   # the score
    control_metrics: dict[str, Any],     # the score to compare against
    maximize: bool | None,               # the direction
    ...
)
```

That is score + direction, splatted into positional arguments. What makes it
Kaggle-shaped is not the shape — it is **where each part comes from**:

| Part | Comes from today | Why that is Kaggle |
|---|---|---|
| score | `_primary_cv_keyed(metrics)` over `metrics.json` | assumes `cv_`-prefixed keys written by a training template |
| direction | `_resolve_direction(knowledge_dir, competition, workspace_root)` | reads `competition.json`, then an Analyze profile artifact |
| second opinion | `lb_gain` and the leaderboard branch of `_decide` | a public leaderboard exists |

So M12 is not "introduce an abstraction". It is **give the existing triple a
name, and let the thing that produced the score also state its direction**,
instead of a competition file being asked about a number it never saw.

## 2. What is actually coupled, measured

The plan's trap list asserts Kaggle assumptions have leaked into the control
plane. Measured on `main` at `a72a51a`:

| Term | Files | Where |
|---|---|---|
| `kaggle` | 35 | accessor, cli, config, conductor, execution, intelligence, planner, tools |
| `leaderboard` | 17 | accessor, cli, conductor, evidence, execution, intelligence, shared, tools |
| `max_daily_submissions` | 7 | accessor/kaggle, execution, intelligence |
| `lb_gain` | 7 | cli, evidence, execution, intelligence |
| `submission_columns` | 6 | accessor/profiler, execution, intelligence |
| `kernel_output_file` | 3 | intelligence |

The raw counts overstate it. Reading the `leaderboard` hits inside the layers
the plan calls domain-neutral — conductor, evidence, shared — **five of the six
are comments or docstrings**. Exactly one is executable: the
`("submit", "leaderboard", "upload")` keyword tuple at `conductor/actions.py:102`.
The structural coupling above the validator is only:

- `ObservedOutcomes.lb_gain` and the leaderboard branch in `_decide`
- `Experiment.public_score`
- that one keyword tuple

**The plan's claim that the control plane is already domain-neutral is
substantially correct**, and that is the single most important input to this
design: it means M12 is a small change at one boundary, not a refactor of the
loop. The Conductor, hypotheses, beliefs, claims, reflection, memory and context
engine are untouched by every phase below.

## 3. What M22's objective layer already gives us

`ObjectiveSpec` (shipped in #142, hardened in #145) carries:

```python
metric_name, direction, direction_source, source, confidence, evidence
```

That is **four of the five fields** a `ValidationResult` needs, already built,
already tested, and already carrying provenance rather than a bare boolean. And
`probe_direction` measures direction by *running a scorer* —

```
perfect  = score(y, y)
degraded = score(y, worse)
```

— which is defined for any objective, needs no catalogue, and takes no task
label. It is already the domain-neutral core M12 asks for; it was built for a
different reason and reached the shape first.

**This design therefore does not introduce a direction concept.** It moves the
existing one from a competition file onto the result.

## 4. Scope

**In:**

- Name the triple: `ValidationResult`.
- One `HypothesisValidator` protocol, and **two** implementations — the existing
  Kaggle path, extracted unchanged, and one genuinely different second.
- Move `maximize` from "resolved from a competition" to "carried by the result".
- An end-to-end campaign against the second validator.

**Out, deliberately:**

- **No plugin registry.** The plan is explicit: *"One extra validator, hardcoded,
  will reveal the interface. A registry can come after there are three."* Two
  implementations selected by a conditional.
- **No renaming of `competition`.** It is the workspace identifier throughout the
  CLI, the stores and the on-disk layout. Renaming it is a large, purely
  cosmetic diff that would bury the behavioural change. `--competition` reads
  badly for a benchmark; that is a documented wart, not a bug.
- **No removal of `lb_gain`.** A leaderboard is a real second opinion that the
  Kaggle validator genuinely has. It becomes optional rather than assumed.
- **No new `TaskType`.** The DAG and capability registry are already pluggable;
  nothing in this design needs a new node type.

## 5. Design

```
                     ┌──────────────────────────────────────────┐
  Conductor ─────────│ unchanged: hypotheses, beliefs, claims,  │
  policy, reflection │ reflection, memory, context engine       │
                     └──────────────────────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │  HypothesisValidator      │   ← the seam
                        │  .validate(...) -> Result │
                        └───────────────────────────┘
                            │                    │
              ┌─────────────┘                    └──────────────┐
              ▼                                                 ▼
   ┌────────────────────────┐                     ┌──────────────────────────┐
   │ KaggleCvValidator      │                     │ HarnessValidator         │
   │ metrics.json + cv_*    │                     │ result.json, no folds,   │
   │ direction: competition │                     │ direction stated by the  │
   │ + optional leaderboard │                     │ harness that computed it │
   └────────────────────────┘                     └──────────────────────────┘
                            │                    │
                            └────────┬───────────┘
                                     ▼
                     build_evidence_card(result, control_result)
                                     │
                                     ▼
                          _decide(...)  → EvidenceCard
```

### `ValidationResult`

```python
@dataclass(frozen=True)
class ValidationResult:
    """What a hypothesis scored, and everything needed to compare it."""

    score: float | None
    metric: str                      # canonical key or a stable slug
    direction: Direction | None      # "maximize" | "minimize" | None
    source: str                      # which validator produced this
    provenance: list[str]            # how score and direction were established
    artifacts: dict[str, str]        # paths, keyed by role
    raw: dict[str, Any]              # the untouched metrics blob
    secondary: float | None = None   # leaderboard, held-out set, replication
```

Three fields carry the load and each answers a failure this repo has recorded:

- **`metric` travels with the score.** `build_evidence_card` already refuses a
  comparison when two runs report different keys (`mismatched_metric`), and it
  recovers that key by re-reading the blob. Carrying it removes the re-derivation.
- **`direction` is `Direction | None`, not `bool`.** `None` is a real answer.
  `maximize: bool = True` is what recorded rogii's one genuine improvement
  (194.80 → 190.97) as `rejected`; `build_evidence_card` now raises rather than
  defaulting, and a nullable field is how a validator says "I could not tell"
  without inventing a sign.
- **`secondary` replaces `lb_gain` by role, not by name.** A leaderboard, a
  held-out set and a replication attempt are the same thing structurally: a
  second measurement the first cannot see.

### `HypothesisValidator`

```python
class HypothesisValidator(Protocol):
    def validate(self, hypothesis, workspace, context) -> ValidationResult: ...
```

Kept exactly as the plan wrote it. The protocol is three lines because the
interesting content is `ValidationResult`, not the call.

## 6. The second validator

The plan says a benchmark harness is cheapest. Three candidates, and the choice
matters because a second validator that shares the Kaggle one's assumptions
**validates nothing** — it would confirm the interface by not exercising it.

| Candidate | Different in | Runs in CI? | Verdict |
|---|---|---|---|
| Local dataset + supplied scorer | no submission | yes | too close — still CV, still a metrics blob |
| **Script harness (`result.json`)** | **no folds, no submission, no `competition.json`, direction stated by the producer** | **yes, hermetic** | **chosen** |
| Paper reproduction | agreement-with-published as the score | no — needs network and a paper | later |

**Chosen: `HarnessValidator`.** It runs a script in the workspace and reads a
`result.json`:

```json
{"score": 0.82, "metric": "pass_rate", "direction": "maximize"}
```

Every Kaggle assumption is absent: no cross-validation, so `_primary_cv_keyed`
never runs; no `cv_std`, so stability is honestly `UNKNOWN`; no submission and no
leaderboard, so `secondary` is `None` and `_decide` must reach a verdict on one
number; and **direction is stated by the thing that computed the score**, which
is the inversion this whole milestone is about.

That last point is why this beats the "local dataset" option: it is the only
candidate where the direction cannot be recovered from a competition file even
in principle.

## 7. Phases

Each phase is separately mergeable and leaves the suite green.

| # | Ships | Behaviour change |
|---|---|---|
| 0 | `ValidationResult` + `HypothesisValidator`, with the Kaggle path wrapped in `KaggleCvValidator`. `build_evidence_card` gains a `result=` path beside its current arguments. | none — the wrapper produces byte-identical cards |
| 1 | ~~`direction` sourced from the result~~ — landed inside phase 0. Phase 1 instead **routes the production caller through the validator**, so the seam is used rather than merely available. | none — same sources, same order |
| 2 | `HarnessValidator` + `result.json` contract, selected by a conditional. Reads the result rather than running the harness — same split as the Kaggle validator, which does not train either. | new capability |
| 3 | Criteria 1–2 landed with phase 2. Phase 3 is **criterion 3**: the launch gate could only read a `competition.json`, so `research conduct` refused every benchmark workspace outright. | a campaign can start in the other domain |

Phase 0 is the one that carries risk, and it is deliberately a no-op: the test
that matters is that a card built through the wrapper is identical, field for
field, to one built through today's arguments.

## 8. Testing

The exit criteria are testable as written, and two of the three are structural:

1. **"no change to Conductor, policy, hypothesis or reflection code"** — assert
   it, do not claim it. A test pinning those modules' import closure against
   `execution.validators` fails if the seam leaks upward. This is the same
   technique as `test_no_literal_metric_list_survives_outside_this_module`.
2. **"a campaign runs against it end to end"** — the harness fixture makes this
   hermetic: no Kaggle credentials, no network, under CI's existing budget.
3. **"`research conduct \"<goal>\"` phrasing unchanged between domains"** — one
   test invoking the same command string against both workspaces.

Beyond the criteria, the case that must not regress: **a validator that cannot
determine direction must block, not guess.** `ValidationResult.direction is None`
has to reach the same refusal `build_evidence_card` already raises, because a
card whose sign is a guess is the failure that produced fifteen wrong ones.

## 9. Tradeoffs

| Choice | Alternative | Why | Cost |
|---|---|---|---|
| Name the existing triple | Design `ValidationResult` from the plan's table | The table lists five domains we cannot run; the triple is what the working loop actually passes | The name is shaped by one working example plus one new one, not five |
| Direction on the result, competition file as fallback | Direction only on the result | Every existing workspace states direction in `competition.json`; removing the fallback is a migration this milestone does not need | Two sources for one fact, temporarily |
| Two validators, hardcoded | Registry now | The plan's own instruction, and the interface is still a guess until a second one exists | A third validator will need the registry, and that is the right time |
| Keep `competition` as the identifier | Rename to `workspace` / `target` | Renaming touches the CLI, both stores and the on-disk layout for zero behaviour change | `--competition my-benchmark` reads badly |

## 10. The risk worth stating

**The interface may still be wrong.** The plan waited for the loop to work
before designing this, and it was right to; but two examples is two, and the
harness validator was chosen partly *because* it is easy to run, which is a
selection bias toward simple results.

The mitigation is the phase structure rather than more design: phase 0 is a
provable no-op, and phases 1–3 are individually revertible. If the third
validator — paper reproduction is the likely one — does not fit
`ValidationResult`, the cost of having been wrong is one dataclass and one
protocol, not a loop that has to be rebuilt.
