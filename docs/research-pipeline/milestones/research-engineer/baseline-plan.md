# Research Engineer — Baseline plan (P-001)

Back to [README.md](README.md) · Planner: [../research-planner/README.md](../research-planner/README.md).

**Status:** Design Phase A.

---

## 1. Why a baseline plan

Baseline quality varies by **problem type** (tabular vs image vs audio, metric, rules).
The old `research run --competition` path baked baseline selection into a linear Pipeline.
Under the Research Engineer, baseline is a **first-class ResearchPlan**:

- Inspectable DAG  
- Same verification / resume / evidence path as hypothesis experiments  
- Conventionally **`P-001`** for each competition  

This milestone owns both **emitting** that plan and **implementing** it via
`research run --plan P-001`.

---

## 2. CLI

```bash
research plan create <competition> --baseline
```

Rules:

1. Requires completed Analyze context (competition/dataset artifacts, profile signals,
   metric/rules as available under `knowledge/<slug>/research/`).
2. Refuses if any research plan already exists for the competition (baseline must be first).
3. Allocates **`P-001`**.
4. Sets `metadata.plan_kind = "baseline"`.
5. Links reserved hypothesis id **`H-BASELINE`** (created if missing; not `H-001`).
6. Optional Planning Engine LLM may refine the slim draft; soft-fail keeps rule_engine
   baseline template (same Option B posture as Planner).

Hypothesis plans remain:

```bash
research plan create <competition> --hypothesis H-xxx
```

Those compare against P-001 (or current best experiment), not rebuild a baseline.

---

## 3. Baseline DAG shape (instruction set)

Illustrative topological spine (exact keys in Phase B templates):

```text
L0  workspace / data prep          (download, layout, profile hooks as tasks)
L1  read_code / select baseline    (registry + problem type)
L2  write_code + modify_config     (scaffold train/infer pipeline)
L3  research_review                (correctness gate on scaffold)
L4  install_package (if needed)
L5  run_unit_test
L6  run_smoke_test                 ★ production-shaped gate
L7  runtime select / provision
L8  run_training                   (full / budgeted)
L9  run_inference → evaluate
L10 build_submission → upload
L11 generate_report → reflect → update_belief
```

Smoke gate (before full train) must verify roughly:

- Runs small batches / 1 epoch / 1 validation without crash  
- Memory healthy  
- Inference path works  
- Submission artifact shape is generable  

Only then allow full training.

---

## 4. Problem-type variance

Baseline **selection** (model family, template, CV strategy) uses deterministic inputs:

- Competition / dataset Analyze artifacts  
- `baseline/` registry mappings (existing)  
- Metric and rules  

The DAG structure stays on the same instruction set; descriptions/inputs change by
problem type. The Engineer does not invent a second pipeline language.

---

## 5. Execution

```bash
research run --plan P-001
```

1. Load plan; require `status=ready` (or explicit approve step if Phase B adds one).  
2. Create execution `E-xxx`; set plan `in_progress`.  
3. Research Engineer walks tasks → capabilities → verify → recover.  
4. On success: experiment row + artifacts + report; plan `done`.  
5. Operator may leave the machine; resume with `research resume --execution E-xxx`.

---

## 6. Relationship to hypothesis plans

| Plan kind | Source | Compares to |
|-----------|--------|-------------|
| `baseline` (P-001) | Analyze + registry | N/A (establishes floor) |
| hypothesis | `H-xxx` | P-001 metrics / best so far |

Planner templates for SpecAugment-style experiments already assume a baseline metric
input — P-001 makes that real.

---

## 7. Non-goals

- Multiple concurrent baselines per competition in MVP  
- Baseline without Analyze context  
- Keeping slug-only Pipeline as a permanent alternate baseline path after capstone  
- Auto-running `research run` immediately after `--baseline` without an explicit run command
  (unless a later UX flag is approved)
