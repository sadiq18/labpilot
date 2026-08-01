# Plan 2 — Constrained policy + scheduler

Back to [README.md](README.md).

## Goal

Scheduler dispatches ready tasks to ToolRegistry. Policy LLM selects NextAction
only from the registered catalog (offline fallback for tests).

## Acceptance

- [x] Allowlist rejects invented tools
- [x] Offline deterministic next-tool order
- [x] No tool→tool chaining from handlers
