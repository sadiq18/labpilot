# Research Reflection — CLI

Back to [README](README.md).

---

## 1. Primary commands (target)

```bash
# Auto path: Reporting tasks inside an Engineer plan
research run --plan P-001 --competition <slug>

# Explicit re-run of reflection for an execution
research reflect --execution E-001 --competition <slug>
research reflect --experiment <id> --competition <slug>   # optional alias

# Human-readable research memory
research journal --competition <slug>
research journal --competition <slug> --format markdown   # default
research journal --competition <slug> --json              # machine

# Claims (Plan 7+)
research claims list --competition <slug>
research claims show C-001
```

---

## 2. `research reflect`

| Flag | Role |
|------|------|
| `--execution` / `-e` | Prefer: reflection from Engineer evidence pack |
| `--competition` | Required for belief/hypothesis scoping |
| `--offline` / rule_engine | Force non-LLM critic path |
| `--dry-run` | Compute assessment; do not write beliefs/claims |

**Effects (when not dry-run):** write `experiment_evidence`, run Critic, mutate
beliefs (+ audit), evaluate linked hypothesis, optionally update lessons/claims.

---

## 3. `research journal`

Answers for a competition:

1. Strong / moderate / weak / rejected evidence (from `experiment_evidence`)
2. Open questions (unresolved hypotheses + contested claims)
3. Current beliefs (top by confidence)
4. Supported claims
5. Recommended next experiment (from recommendation module)

Journal is a **projection** — not a second SoR. Pipeline-era
`research report` / `labpilot.report` are **removed in Plan 9**; existing
`runs/*/report.html` files may remain on disk as historical artifacts only.

---

## 4. North-star loop (post M6)

```bash
research analyze <slug>
research plan create <slug> --baseline    # or --hypothesis H-xxx
research run --plan P-001 --competition <slug>
research journal --competition <slug>
```

Legacy `init` / `build` / plan-less `run` are **retired** (see Research Engineer
pipeline deprecation). Do not document them as the north star.
