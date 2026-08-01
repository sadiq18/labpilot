# Plan 5 — retrieve / explain CLI

Back to [README.md](README.md).

## Goal

User-facing trust/debug: dump ranked evidence and why items were included.
Secondary to Conductor quality — not the primary M4 deliverable.

## Acceptance

- [x] CLI builds `ContextBundle` (not only RI `ResearchContext`)
- [x] Explain-oriented view shows inclusion reasons / scores
- [x] Works offline against fixture knowledge DB

## Commands

```text
research context retrieve <slug> -q "…" [--format text|json]
research context explain  <slug> -q "…"
```

Legacy ``research retrieve`` remains the Plan 9 RI ``ResearchContext`` path.
