# Legacy Pipeline deprecation

Back to [README.md](README.md) · Capstone:
[capstone-notes.md](capstone-notes.md).

**Status:** Done — Analyze → `plan create` → `run --plan` is SoR.
Linear `Pipeline` (`init` / `build` / `improve`) has been **removed**.

---

## Happy path (SoR)

```bash
research analyze <competition>
research plan create <competition> --baseline   # or --hypothesis H-xxx
research run --plan P-001 --competition <competition>
research resume --execution E-001 --competition <competition>
```

---

## Phases

| Phase | Goal | Exit criteria | Status |
|-------|------|---------------|--------|
| 0 | Soft deprecation + docs | Docs match SoR; deprecation tracked here | Done |
| 1 | Workspace download + profile | `run --plan --dry-run` works without `research init` | Done |
| 2 | Retire `build` | Build superseded by Engineer; BaselineSelector in Code Engineering | Done |
| 3 | Retarget `improve` | Improve → `plan create --hypothesis` → `run --plan` | Done |
| 4 | Delete Pipeline | No `orchestrator/pipeline.py`; CLI init/build/improve gone | Done |
| 5 | Package cleanup | Leftover Pipeline-only packages slimmed/quarantined | Done |

Kernel-mode export/upload remains a follow-on (was Pipeline-only); CSV competitions are the removal gate.

---

## Migration map

| Legacy | Replacement |
|--------|-------------|
| `research init` | Workspace capability (dirs + data + profile) + `research analyze` |
| `research build` | `research plan create --baseline` + `research run --plan` |
| `research improve --run-id` | `research plan create --hypothesis H-xxx` + `research run --plan` |
| `research run -c` (no plan) | Already rejected — use `--plan` |
| `research resume --run-id` | `research resume --execution E-xxx` |
