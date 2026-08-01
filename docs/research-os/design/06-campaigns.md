# Design — Campaigns

Back to [../README.md](../README.md) · Milestone: [../milestones/03-campaigns/](../milestones/03-campaigns/).

**Milestone:** M3 · **Impl branch:** `research-os-m3-campaigns`

---

## Goal

Turn the Conductor kernel into **goal-driven campaigns**: dynamic tasks, approval
ladder, stop conditions, and a thin goal-oriented CLI.

---

## Dynamic scheduling (Strangler Phase C)

Conductor may insert tasks beyond the fixed V1 sequence when useful, e.g.:

- `fetch` / paper search
- hypothesize / rank
- extra validation experiment
- reflect before next plan

Still: strategy in Conductor; execution in tools/Engineer.

---

## Autonomy ladder

| Level | Gates |
|-------|-------|
| 0 (default early) | Pause before new plan batch + LB submit |
| 1 | Pause before submit only |
| 2 | Pause only on budget/policy changes |
| 3 | No pauses (hard budgets still stop the loop) |

---

## Stop conditions

- Goal metric reached (e.g. LB ≥ target)
- Submission / time / $ budget exhausted
- Plateau policy (optional)
- Operator pause / cancel

---

## Goal CLI (extends M2 `research conduct`)

```text
research conduct "<goal>"     # product entry (ships in M2)
research continue
research status
research pause
research resume
```

`continue` / `resume` **restore checkpoints** (queue, objective, workspace refs) —
not a new chat transcript. Legacy `analyze` / `plan` / `run` / `reflect` remain for
debug and power users. Future IDE clients call the same runtime APIs.

---

## Non-goals

- Full `explain` / `benchmark` / `replay` suite (can follow as small CLI polish;
  replay reads episodic memory — [10-memory-os](10-memory-os.md))
- Parallel multi-agent trees (M5)
- Cross-competition transfer (M6)
