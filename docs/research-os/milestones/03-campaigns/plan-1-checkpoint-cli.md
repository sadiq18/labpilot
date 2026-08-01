# Plan 1 — Checkpoint + conduct CLI lifecycle

Back to [README.md](README.md).

## Goal

Durable campaign checkpoint and CLI:

```text
research conduct continue|pause|resume|status [--session S-xxx]
```

Resume defaults to the latest active session for the competition.

## Acceptance

- [x] Pause/resume/continue restore queue + session status
- [x] `status` shows session summary
- [x] `--session` overrides latest-active default
