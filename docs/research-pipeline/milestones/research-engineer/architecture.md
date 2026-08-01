# Research Engineer — Architecture

Back to [README.md](README.md) · [capabilities.md](capabilities.md) ·
[runtime-and-recovery.md](runtime-and-recovery.md).

**Status:** Design Phase A.

---

## 1. One orchestrator

The **Research Engineer** is a deterministic workflow controller. It resembles a
Kubernetes controller + GitHub Actions + ML workflow engine — **not** an autonomous AI
employee with chat memory.

```
Approved ResearchPlan
        │
        ▼
Research Engineer
        │
        ▼
   Task Queue  (topo order from plan deps)
        │
        ▼
Capability Registry  →  CapabilityExecutor
        │
        ▼
Task Evidence
        │
        ▼
Verification Engine
   │            │
 verified     failed
   │            │
   ▼            ▼
 next task   Recovery Controller → retry / rollback / stop
        │
        ▼
Experiment Artifact → Reflection / Belief update
```

The Engineer owns:

- Loading and validating an approved plan
- Creating a durable **execution** attempt (`E-xxx`)
- Queueing tasks by topological levels
- Dispatching to capabilities
- Recording status transitions and evidence
- Invoking verification and typed recovery
- Resume / idempotency
- Emitting the final experiment artifact

The Engineer **never** uses an LLM to choose the next task.

---

## 2. Roles vs agents (locked)

**Rejected pattern:**

```
Planner Agent → Coder Agent → Reviewer Agent → Executor Agent → Reflection Agent
```

Those are **roles**, not agents.

**Adopted pattern:**

```
Planner (compiler) → ResearchPlan tasks → Research Engineer → Capability Executors
```

Example:

| Task type | Capability |
|-----------|------------|
| `WRITE_CODE` | Code Engineering |
| `RUN_UNIT_TEST` / `RUN_SMOKE_TEST` | Verification |
| `RUN_TRAINING` | Training |

Many tasks; few stable capabilities. See [capabilities.md](capabilities.md).

---

## 3. CapabilityExecutor contract

Every capability implements:

| Method | Responsibility |
|--------|----------------|
| `supported_task_types` | Which `TaskType` values it handles |
| `prepare(context)` | Resolve paths, env, prior artifacts |
| `execute(context)` | Perform the side effect (or call one micro-agent then apply) |
| `verify(context, result)` | Produce typed evidence; pass/fail |
| `rollback(context)` | Undo when required |
| `collect_evidence(…)` | Persist structured evidence for the platform |

Capabilities are **tools**. They do not own the global queue or plan lifecycle.

---

## 4. TaskContext (bounded, ephemeral)

Memory belongs to the **platform**, not to micro-agents.

Each attempt receives a `TaskContext` roughly containing:

- Plan goal + current task (id, type, description, inputs/outputs, verification, retry)
- Competition / execution / workspace paths
- Relevant files (usually **≤ ~20** — never the whole repo)
- Dependencies, constraints, expected outputs
- Related hypothesis (if any)
- Selected runtime target
- Prior attempt evidence for this task

Finish the attempt. Forget. Stateless workers are easier to debug.

---

## 5. Micro-agents (inside capabilities only)

Micro-agents are **stateless reasoning functions** used only where judgement is required
(code strategy, patch proposal, research review, recovery *suggestions*, test generation).

Contract (same family as Planner / Intelligence):

```
TaskContext → (optional) Micro Agent → typed artifact → deterministic apply → verify
```

Principles:

- No session memory, no tool-calling loops, no autonomy
- At most **one** specialist micro-agent call per attempt inside a capability
- Typed output must validate before any write
- Soft-fail to `rule_engine` / templates where a deterministic path exists

**Never** put an LLM in: training launch, file I/O as SoR, Kaggle upload, metrics,
checkpoints, GPU scheduling, status machine, retries, artifact sync.

---

## 6. Verification Engine

Every task emits evidence. Research engineers verify; they do not “hope.”

Examples:

| Kind | Evidence |
|------|----------|
| Code | Diff applied; project compiles; unit tests green after change |
| Research Review | Typed findings; critical findings block progression |
| Smoke | 2 batches, 1 epoch, 1 validation; no crash; memory OK; inference + submission shape OK |
| Training | Loss finite / decreasing; checkpoint exists; GPU healthy |
| Experiment | Baseline vs new metric delta recorded |

Verification failure routes to Recovery — not to a free-form agent chat.

---

## 7. Recovery Controller

Typed policies, preferably deterministic:

| Failure | Recovery |
|---------|----------|
| CUDA OOM | Reduce batch size → retry (bounded) |
| Validation worse vs gate | Stop full train → save evidence → mark task/plan accordingly |
| Dependency conflict | Create / switch isolated environment → retry install |
| Flaky remote | Retry with backoff → fail with evidence |
| Code review critical | Block → Code Engineering fix attempt (bounded) → re-review |

Recovery suggestions from an LLM are allowed; applying them is still a controlled,
verified capability path.

---

## 8. Experiment Artifact Generator

When the plan completes (or stops with evidence), produce a durable experiment result that
feeds reflection — e.g.:

```json
{
  "id": "exp_…",
  "execution_id": "E-001",
  "plan_id": "P-001",
  "hypothesis_id": null,
  "competition": "…",
  "changes": ["baseline pipeline from template X"],
  "result": { "cv": 0.849, "delta": null },
  "artifacts": ["checkpoint.pt", "logs", "submission.csv"],
  "task_evidence_refs": ["…"]
}
```

Map into DB `experiments` (+ disk workspace). See [schema.md](schema.md).

---

## 9. Relationship to today’s Pipeline

Today’s [`orchestrator/pipeline.py`](../../../src/labpilot/orchestrator/pipeline.py) is a
linear stage DAG. After Phase B:

- **Research Engineer** becomes the SoR for plan-driven runs
- Stage implementations (`training/`, `submission/`, …) **migrate or wrap** as capabilities
- Obsolete linear orchestration for the research path is **deleted**
- `research improve` either becomes “compile hypothesis plan → run Engineer” or is slimmed —
  exact cut decided in Phase B (design intent: one plan-driven path)

---

## 10. Explicit architectural non-goals

- Multi-agent orchestration replacing the Engineer
- ReAct / tool-calling loops at the top level
- Per-task dedicated agents (capabilities scale; tasks proliferate)
- Unbounded repository mutation
- LLM-owned control flow
