# Plan 5 — Write hooks (persist on completion)

Back to [README.md](README.md).

## Goal

Automatically persist Experience Records when experiments complete, without
bypassing Conductor strategy.

Preferred path (when M5 event bus is available):

```text
ExperimentCompleted (Blinker)
  → subscriber calls ExperienceExtractor
  → ExperienceStore upsert
```

Fallback: Reflection / Engineer reporting completion hook invokes the same
extractor callable.

Rules:

- Subscribers **only write memory**; they must not enqueue tasks or change
  Conductor policy.
- Upserts remain idempotent (plan 1 / 2).
- Failures in extraction log and do not fail the experiment pipeline.

## Acceptance

- [ ] Completion of an experiment produces/updates an Experience Record
- [ ] Re-running completion path does not duplicate rows (idempotent)
- [ ] Event subscriber (or documented fallback) does not schedule Conductor work
- [ ] Records include artifact links; `git_commit` when present on experiment
- [ ] Unit/integration test: complete fixture experiment → store has one record

## Out of scope

- Capstone multi-competition smoke (plan 6)
- Auto warm-start on new campaign
- Peer agents proposing next work from experience events (forbidden; stays Conductor)
