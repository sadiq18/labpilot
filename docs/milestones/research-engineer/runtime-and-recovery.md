# Research Engineer — Runtime and recovery

Back to [README.md](README.md) · [capabilities.md](capabilities.md) ·
[architecture.md](architecture.md).

**Status:** Design Phase A.

---

## 1. Why this doc

A research engineer spends real time on environments, GPUs, and failures. The Runtime
capability and Recovery Controller make that **observable and resumable**. Today’s
[`runtimes/`](../../../src/labpilot/runtimes/) package is **config-only**; this milestone
adds dispatch, polling, artifact pull, and health evidence.

---

## 2. Runtime capability responsibilities

1. **Discover** available runtimes (local, Docker, Kaggle kernel, configured cloud).  
2. **Match** task needs (VRAM estimate, CPU-only smoke, competition kernel rules).  
3. **Select** a target deterministically (policy table — not LLM).  
4. **Dispatch** the job (local subprocess, Kaggle kernel push, remote API).  
5. **Poll** until complete or timeout; record heartbeats.  
6. **Pull** artifacts (checkpoints, metrics, logs, submission).  
7. **Verify** health (process exit, GPU errors, missing checkpoint).  

Example policy (illustrative):

```text
Smoke / unit          → local
Full training         → Kaggle GPU or configured cloud if local VRAM insufficient
Inference / submit    → local or same remote as train (artifact locality)
```

If model needs 24GB and only 16GB local → reduce batch (Training recovery) **or** select
remote runtime (Runtime selection) — policy order documented in Phase B.

---

## 3. Smoke gate before full train

Smoke is part of **Verification**, but Runtime must support “tiny job” semantics:

- 2 batches, 1 epoch, 1 validation (as configured)  
- Catch CUDA OOM, import errors, path bugs, submission wiring **before** burning quota  

Full training tasks stay `pending` / become `skipped` if smoke fails and recovery exhausts.

---

## 4. Training health verification

After / during training:

- Loss finite (and ideally decreasing over the smoke/full window)  
- Checkpoint exists  
- GPU / process healthy (no silent hang — heartbeat / log tail)  

Failures feed Recovery.

---

## 5. Recovery catalog

| Failure | Primary recovery | Escalate |
|---------|------------------|----------|
| CUDA OOM | Reduce batch size; retry N times | Switch runtime; fail with evidence |
| Dep conflict | Isolated env + reinstall | Fail task |
| Smoke crash | Code Engineering fix attempt (bounded) + re-smoke | Fail plan early |
| Val worse vs gate | Skip gated full-train; save evidence | Mark inconclusive |
| Remote 429 / transient | Backoff retry | Fail execution |
| Upload reject | Fix submission format; retry | Fail submission task |
| Review critical | Code fix + re-review | Block progression |

LLM may **suggest** recovery actions; the Controller applies only typed, verified paths.

---

## 6. Resume and idempotency

- Durable `research_executions` id (`E-xxx`)  
- `research resume --execution E-xxx` continues from first non-terminal task  
- Capability `execute` must be safe to retry (or declare non-idempotent with explicit
  resume hooks — e.g. remote job id in metadata)  
- Never restart a completed Kaggle job blindly; poll existing job id when present  

---

## 7. Relationship to deferred “P2 execution”

Historical TODO called remote dispatch “P2 execution.” This milestone **absorbs** the
product need (dispatch/poll/pull) into the Research Engineer Runtime capability rather
than a separate forever-deferred track. Scope is still bounded by configured runtimes
in the registry — not every cloud vendor on day one.

---

## 8. Non-goals

- LLM picking GPUs  
- Unbounded spend / quota without local policy caps  
- Replacing Kaggle’s API with scraping  
- Guaranteeing free-tier success (evidence must show quota/failure clearly)  
