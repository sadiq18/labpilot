# Autonomy Roadmap

Plans produced from a full-day validation session (2026-08-02) that drove
`research conduct` against `rogii-wellbore-geology-prediction` until the
autonomous loop's real limits surfaced.

**Read [00-diagnosis.md](00-diagnosis.md) first.** Everything else follows from
one finding, and the plans are worthless without it.

---

## The finding

> Every milestone shipped its **structure** but not its **function**.

The stores, registries, event bus, memory layers, agent runtime and task queue
all exist and are well built. Almost none of them *change an outcome*. Memory
has four layers and nothing reads them to make a decision that alters a model.
The event bus exists and no agent reacts to `ExperimentCompleted`. M5 shipped
parallel workers and the campaign runs strictly sequential.

The sharpest instance: the Conductor can decide "try a CNN", and nothing
downstream can produce one. Twelve hypotheses were proposed, planned, run and
reflected on — and every experiment scored **MSE 194.80, identically**.

You built the brain and the nervous system. You did not build hands that can do
more than one motion.

---

## Critical path

Do not reorder. Each is dead without the previous.

| # | Plan | Unlocks | Status |
|---|------|---------|--------|
| **M7** | [Technique → model](01-technique-to-model.md) | Anything at all. Without it there is nothing to optimise over. | Not started |
| **M8** | [Objective feedback loop](02-objective-loop.md) | The system noticing it is making no progress | Not started |
| **M9** | [Verification-first execution](03-verification-first.md) | Trusting any result | Partly done |
| **M10** | [LLM tiering & free-tier routing](04-llm-tiering.md) | A model capable enough for M7 | Core built |
| **M11** | [Parallel branches](05-parallel-branches.md) | Iteration speed | Not started |
| **M12** | [Beyond Kaggle](06-beyond-kaggle.md) | The actual product thesis | Not started |
| — | [Interaction modes](07-interaction-modes.md) | Auto / accept-edits / plan UX | Not started |

**M10 before M7 in practice.** M7 needs a model that can write code; the local
14B produced none. Do M10's remaining wiring first, then M7.

---

## How to use these documents

Each plan carries:

- **Purpose** — the concrete failure it removes, with evidence
- **Goal** — what becomes possible
- **Approach** — design decisions *and their rationale*
- **Exit criteria** — a check that cannot be satisfied by accident
- **Traps** — approaches already tried and rejected, so they are not retried

[evidence-log.md](evidence-log.md) records every defect found in the session,
how it was found, and what it cost. When a plan's rationale seems excessive,
the evidence log is why.

---

## The one habit to break

M5 shipped parallel agents before the sequential loop could run a single real
experiment. Breadth before depth is exactly how a beautiful control plane ended
up driving a one-motion data plane.

M7 is unglamorous — a recipe table and some plumbing — and it is worth more than
every remaining milestone combined. If you find yourself adding a seventh
provider adapter before a single technique has changed a CV score, that is the
same trap wearing a new hat.
