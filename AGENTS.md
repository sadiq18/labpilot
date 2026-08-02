# AGENTS.md

Committed agent guidance for LabPilot. Local Cursor/Claude copies may live in
`.cursor/`, `.claude-plugin/`, `skills/karpathy-guidelines/`, and `CLAUDE.md`
(gitignored); this file is the repository source of truth.

## Behavioral rules (Karpathy)

Synced from [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills):

1. Think before coding — surface assumptions and tradeoffs.
2. Simplicity first — minimum code that solves the ask.
3. Surgical changes — touch only what you must.
4. Goal-driven execution — define verifiable success criteria and loop until checked.

Local mirrors (gitignored): `.cursor/rules/karpathy-guidelines.mdc`,
`skills/karpathy-guidelines/SKILL.md`, `CLAUDE.md`.

## Growth areas (required)

1. **Post-AI verification** — After AI-produced architecture/analysis/reflection
   outputs, run an explicit accept / reject / spot-check path
   (`verify_ai_artifact`). Do not treat first-pass LLM findings as final.
2. **Repo hygiene** — No real secrets in the tree (test fakes like `sk-test` are
   OK). Prefer conventional commits (`feat:`, `fix:`, `test:`, `docs:`…).
   Do not mass-split oversized modules unless asked.
   Keep `.cursor/`, `.claude-plugin/`, `skills/karpathy-guidelines/`, and
   `CLAUDE.md` gitignored; do not commit local agent tooling.
3. **Discoverable tests** — Put automated checks under [`tests/`](tests/) so CI
   and scanners find them. Prefer `tests/unit/` for fast unit coverage.

## Verification commands

Matches [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```bash
uv sync --extra dev
uv run pytest -m "not llm and not image and not deep"
```

LLM / image / deep jobs are separate markers — see [`tests/README.md`](tests/README.md).

## Secrets

- Never commit `.env`, API keys, or tokens.
- Unit tests may use obvious fakes (`sk-test`, `fake-key`).
- If a real secret appears: rotate it outside the repo, then remove it from git history.

## Oversized modules (backlog — do not split in drive-by PRs)

Known hotspots (>500 lines), deferred:

- `src/labpilot/research_engine/planner/templates.py`
- `src/labpilot/research_engine/intelligence/knowledge/store.py`
- `src/labpilot/research_engine/intelligence/literature/clients.py`
- `src/labpilot/research_engine/memory/extractor.py`
- `src/labpilot/research_engine/intelligence/hypothesis/candidates.py`
- `src/labpilot/research_engine/planner/planner.py`
- `src/labpilot/research_engine/intelligence/analyzers/competition.py`

Split only with an explicit scoped task and tests before/after.
