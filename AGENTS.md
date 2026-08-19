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

## Working on this codebase

Four rules earned from real defects here. Each cost a bug to learn.

1. **Never edit a competition workspace.** Validate against a sandbox copy:
   `cp -R "$WS/knowledge/research" "$SANDBOX/kb/$COMP/research"`, then pass
   `$SANDBOX/kb` as the knowledge dir (`ResearchPaths` expects
   `<knowledge_dir>/<competition>/research/knowledge.db`). If a workspace needs
   migrating or cleaning, make labpilot do it on the next run — do not hand-edit
   artifacts or the DB.

2. **Recompute, never step.** Derived state must be a function of current
   inputs, so it stays correct after those inputs are repaired.
   `apply_card_to_beliefs` stepped once per card; repairing a card afterwards
   changed nothing, and the one technique that improved the metric stayed
   recorded as harmful. See `evidence/belief_repair.py` for the shape.

3. **Check the field the bad record actually uses.** Nine defects have been
   "the guard exists and its input is wrong" — a check on
   `normalize_label(name)` that strips the colon it tests for; a filter on
   `effect` when the bad claims have `effect=''`; a metrics guard asking
   "is there a file?" when the question was "did *this run* write one?".
   Before trusting a guard, feed it a real bad record.

4. **Prove your test fails without your fix.** Several tests here passed
   vacuously — one compared two renders in *different directories* while the
   renderer bakes the directory into its output. If a test could pass on an
   empty list, assert the list is non-empty first.

## Verification commands

Matches [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```bash
uv sync --extra dev
uv run pytest -m "not llm and not image and not deep"
```

LLM / image / deep jobs are separate markers — see [`tests/README.md`](tests/README.md).

Before pushing anything with concurrency, CPU budgeting, or CLI output, also run:

```bash
scripts/hostile-test.sh
```

It re-runs the unit suite under a faked core count (1 and 2) and a 40-column
terminal. A green mutation sweep proves an assertion is wired to its code; it
cannot prove the assertion means the same thing on another machine. Four tests
on the M11 fan-out branch passed on a ten-core laptop and failed on a two-core
CI runner — a hardcoded core count, a `--help` string Rich truncates when
narrow, and two frozen race outcomes.

`tests/unit/test_tests_do_not_assert_the_machine.py` catches the other half
structurally: a test that compares *equality* against `available_cpus()`,
`gethostname()`, `now()` and friends is asserting the box, not the behaviour.
Neither guardrail subsumes the other — a hardcoded literal only the hostile run
sees, an ambient read only the structural rule sees.

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
