# Research Engineer — Capabilities

Back to [README.md](README.md) · [architecture.md](architecture.md).

**Status:** Design Phase A.

---

## 1. Capabilities scale; tasks proliferate

**Wrong:** one micro-agent per `TaskType`.  
**Right:** one **capability** per stable skill; many task types map to it.

Example:

| Tasks | Capability |
|-------|------------|
| `WRITE_CODE`, `READ_CODE`, code-oriented `MODIFY_CONFIG`, fix attempts | Code Engineering |
| `RUN_UNIT_TEST`, `RUN_SMOKE_TEST` (+ future integration) | Verification |

Aim for **~8–12 capabilities**, not 15+ agents and not hundreds of agents as tasks grow.

---

## 2. Stable capability inventory

### 1. Workspace

- Create isolated workspace, configs, experiments, logs directories  
- Download / attach data via deterministic adapters  
- Verify layout and dataset access  
- Recovery: recreate missing dirs  

### 2. Code Engineering

- Serves `READ_CODE`, `WRITE_CODE`, code-related `MODIFY_CONFIG`, bounded fixes  
- LLM: implementation strategy, patch proposal, bug-fix proposal, test generation  
- Deterministic: apply patch, syntax check, rollback snapshot  
- Controlled ops only (allowed targets + intent + validation)  
- Today also owns template **offline codegen** (`offline_codegen/`) including tabular
  default model params (moved out of deleted `improvement/` in Reflection Plan 9)

> **TODO (follow-on):** When template-based offline codegen is removed, delete the
> offline Jinja renderer defaults (`DEFAULT_TABULAR_MODEL_PARAMS` and related wiring).
> See [research-reflection/plan-9-legacy-cleanup.md](../research-reflection/plan-9-legacy-cleanup.md).

### 3. Research Review

- Reviews every material code/config change for **research correctness** (not style)  
- Example: “EMA only active during training?”  
- Typed findings; **critical** findings block progression  
- LLM judgement allowed; blocking is deterministic policy  

### 4. Dependency

- Serves `INSTALL_PACKAGE`  
- Isolated env when needed; verify import; record lock/evidence  
- Recovery: isolate env / pin versions  

### 5. Verification

- Serves `RUN_UNIT_TEST`, `RUN_SMOKE_TEST`  
- Unit: pytest (or project test command) exit 0  
- **Smoke ★:** production-shaped gate — ~2 batches, 1 epoch, 1 validation; no crash;
  memory OK; inference works; submission shape generable  
- Only then allow full training tasks  

### 6. Runtime

- Discover resources (local Mac/CPU/GPU, Docker, Kaggle, configured cloud)  
- Deterministic selection policy (e.g. local smoke → Kaggle/cloud full train → local
  inference when that is the policy)  
- Dispatch, poll, pull artifacts, health evidence  
- Details: [runtime-and-recovery.md](runtime-and-recovery.md)  

### 7. Training

- Serves `RUN_TRAINING`  
- Launch job; monitor loss / heartbeat / GPU  
- Verify checkpoint + finite/decreasing metrics  
- OOM → reduce batch → retry (bounded)  
- **No LLM** for train/schedule  

### 8. Inference & Evaluation

- Serves `RUN_INFERENCE`, `EVALUATE`, `COMPARE`  
- Deterministic predictions, CV/metrics, reproducibility check, plots, baseline delta  

### 9. Submission

- Serves `BUILD_SUBMISSION`  
- Format validate; package kernel/source if required; **Kaggle upload**; status pull  
- Kernel packaging helper (unused by SoR today): `accessor/kaggle/exporter.py` —
  wire here when kernel-mode submission lands  
- Evidence of upload result  

### 10. Reporting & Memory

- Serves `GENERATE_REPORT`, `UPDATE_BELIEF`, `CREATE_HYPOTHESIS`, `REFLECT`  
- Experiment artifact/report generation  
- Existing reflection micro-agent for judgement slices; platform persists  

---

## 3. TaskType → capability map (v1)

| TaskType | Capability |
|----------|------------|
| `read_code` | Code Engineering |
| `write_code` | Code Engineering |
| `modify_config` | Code Engineering (or Workspace if pure config scaffold — Phase B picks one) |
| `install_package` | Dependency |
| `run_unit_test` | Verification |
| `run_smoke_test` | Verification |
| `run_training` | Training (+ Runtime for where it runs) |
| `run_inference` | Inference & Evaluation |
| `build_submission` | Submission |
| `evaluate` | Inference & Evaluation |
| `compare` | Inference & Evaluation |
| `generate_report` | Reporting & Memory |
| `update_belief` | Reporting & Memory |
| `create_hypothesis` | Reporting & Memory |
| `reflect` | Reporting & Memory |

Research Review is invoked as a **gate capability** after Code Engineering tasks that
mutate code/config (may appear as an explicit plan node or an Engineer-enforced edge —
Phase B prefers an explicit DAG node for observability).

Runtime selection may be an explicit plan node or a Training.prepare step — prefer
explicit node when the plan must show “where we train.”

---

## 4. LLM vs deterministic boundary

**LLM allowed**

- Implementation strategy  
- Bounded workspace planning (suggestions only)  
- Code generation / modification / bug-fix **proposals**  
- Error explanation / recovery **suggestions**  
- Test generation  
- Research correctness review  

**Deterministic only**

- Training, file I/O / patch apply as SoR  
- Kaggle upload  
- Config parsing, dataset loading  
- Metric calculation, experiment logging  
- Checkpoint management, GPU scheduling  
- Process control, status transitions, retries, artifact sync  

---

## 5. Micro-agent principles (per capability)

Every reasoning slice satisfies:

```
Input → Output → Verification → Failure Recovery
```

No memory on the agent. Platform supplies `TaskContext`. Forget after the call.

---

## 6. Non-goals

- One agent per task type  
- Top-level multi-agent bus  
- LLM inside Training / Submission upload / metrics  
- Unbounded file access in Code Engineering context  
