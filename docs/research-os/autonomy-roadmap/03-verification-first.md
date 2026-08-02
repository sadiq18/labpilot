# M9 — Verification-first execution

**Status:** partly done · **Applies to:** every capability, permanently

---

## Purpose

The dominant failure mode in this codebase is **silent success**: a component
does nothing, reports success, and every layer above it agrees. Six distinct
instances were found in a single day, and each was invisible until a file
timestamp or row count was inspected by hand.

This is worse than crashing. A crash costs a run; a silent no-op costs the
credibility of every result in the research memory.

## Goal

A tool that changed nothing did not succeed.

## Approach

Make postcondition assertion a **rule for every capability**, not a patch
applied where a bug was noticed:

| Capability | Postcondition |
|---|---|
| `prepare_workspace` | data files exist and profile has non-zero rows |
| `write_code` | a file changed on disk (digest differs) |
| `run_training` | metrics written **and** newer than the run start |
| `run_inference` | submission row count matches the sample submission |
| `build_submission` | header matches `submission_columns` |
| `reflect` | at least one durable record written, or an explicit skip reason |

Two design rules learned the hard way:

1. **Assert the effect, not the call.** The test that concealed the dry-run bug
   asserted `"dry_run" not in args` — the *absence of a key* — rather than the
   effective value. It passed while the behaviour was wrong. Assert
   `.get("dry_run") is False`.
2. **A degraded path must be labelled, not silent.** Rule-engine fallbacks,
   template fallbacks, and downgraded models are all legitimate — but the result
   must carry which path produced it, or later analysis is unattributable.

## Already done (2026-08-02)

- `ExperimentProducedNoMetricsError` — a non-dry run writing no metrics now
  raises rather than returning success.
- Codegen refuses to continue on the emergency stub for a real run.
- `research doctor` verifies LLM provider reachability *and* that the model is
  pulled.
- Test suite made hermetic — it was silently making real network calls, so
  results were machine-dependent (611 passing / 30s, from 3 failed / 28m40s).
- Progress reporting with elapsed time, so a slow step is distinguishable from a
  hung one.

## Remaining

- Postconditions for the capabilities not yet covered (table above).
- A `--strict` campaign mode where any degraded path fails the step rather than
  proceeding, for use when validating the loop itself.
- Provenance stamped on every experiment: which model, which template, which
  code path (llm / recipe / template / stub).

## Exit criteria

Deliberately break each capability (empty data dir, unwritable metrics, mismatched
submission header) and confirm the campaign **fails at that step** rather than
completing and reporting a result.

## Traps

- **Soft-fail is the right default for *analysis*, wrong for *execution*.** An
  analyzer that cannot reach arXiv should degrade and continue. A training step
  that produced no model must not. The two were treated identically.
- **`except Exception` misses `SystemExit`.** The Kaggle client calls
  `sys.exit()` on auth failure, which aborted a command with empty stdout and
  exit code 0 — indistinguishable from success.
