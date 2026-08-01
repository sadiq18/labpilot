# Plan 4 — Capstone

Back to [README.md](README.md).

## Goal

End-to-end offline conduct loop with durable queue/log; M3 handoff checklist.

## Acceptance

- [x] Offline loop runs ≥2 catalog tools
- [x] Queue + decisions survive reopen
- [x] Checklist that M3 can add dynamic tasks on the same queue

## M3 handoff checklist

- [x] Same `os_tasks` / `os_sessions` tables can store new dynamic tool names later
- [x] `research conduct` is the product entry; M3 adds continue/pause/resume
- [x] Operator feedback comments already feed policy observe
- [ ] Dynamic task *creation* (not just catalog selection) — M3
- [ ] Hard budget / plateau stop policies — M3
