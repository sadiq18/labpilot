# Parallel work plan — #11, #25, #39

Three independent lanes for another agent. A fourth (#40) was worked
concurrently and is in review as PR #99 — **do not touch
`src/labpilot/research_engine/conductor/policy.py`** until it merges.

Each brief below is self-contained. You do not need the conversation that
produced it.

---

## Lane ownership — read this first

Merge conflicts are the main risk of running these in parallel. File ownership:

| Lane | Owns | Must not touch |
|---|---|---|
| **#40** (done — PR #99) | `conductor/policy.py`, `tests/unit/test_conductor.py` | — |
| **#11** | `execution/capabilities/code_engineering/**`, `**/templates/**/*.j2`, `execution/technique/resolver.py` | `conductor/policy.py`; avoid `micro_agents/code_engineer/agent.py` if you can (see #39) |
| **#25** | `execution/technique/` (new `vocabulary.py`), `intelligence/hypothesis/candidates.py`, `accessor/sqlite/schema.sql`, `accessor/sqlite/migrate.py`, `conductor/loop.py` (one call) | `conductor/policy.py` |
| **#39** | all 20 `**/agent.py` + `**/micro_agent.py`, `accessor/common/micro_agents.py`, `tests/conftest.py`, ~28 test files | everything else |

**Sequencing:** #11 and #25 are safely concurrent. **#39 should start last** — it
edits every agent file and `tests/conftest.py`, so it will conflict with anything
still in flight. If you must overlap, run #39 after #11 has merged.

Base all branches on `main` (or on `research-os-m14-phase3` if #98 has not merged
yet — it carries the `structured_output` precondition that #39 depends on).

---

## Conventions that are not optional here

These are hard-won on this repo. Ignoring them is how the bugs below were made.

1. **Never edit the competition workspace.** Validate against a *sandbox copy*:
   ```bash
   # $WS = the competition workspace (the directory holding labpilot.yaml).
   # $COMP = its slug, e.g. rogii-wellbore-geology-prediction.
   mkdir -p "$SANDBOX/kb/$COMP"
   cp -R "$WS/knowledge/research" "$SANDBOX/kb/$COMP/research"
   ```
   Then pass `$SANDBOX/kb` as the knowledge dir. `ResearchPaths` expects
   `<knowledge_dir>/<competition>/research/knowledge.db`, which is why the copy
   is nested that way.
   If a workspace needs migrating or cleaning, make labpilot do it on the next
   run — do not hand-edit artifacts or the DB.

2. **Recompute, never step.** Derived state must be a function of current inputs,
   so it stays correct after those inputs are repaired. `apply_card_to_beliefs`
   stepped once per card; repairing a card afterwards changed nothing, and `SWA`
   stayed recorded as harmful. See `evidence/belief_repair.py` for the shape.

3. **Check the field the bad record actually uses.** Six bugs this month were
   "the guard exists and its input is wrong" — a check on `normalize_label(name)`
   that strips the colon it tests for, a filter on `effect` when the bad claims
   have `effect=''`. Before trusting a guard, feed it a real bad record.

4. **Prove your test fails without your fix.** Several tests here passed
   vacuously — one compared two renders in *different directories* while the
   renderer bakes the directory into its output. If a test would pass on an
   empty list, assert the list is non-empty first.

5. **Run the default slice before claiming done:**
   ```bash
   uv run pytest -m "not llm and not image and not deep"
   ```
   Also run it with provider keys *set*
   (`GROQ_API_KEY=x OPENROUTER_API_KEY=x …`) — one test was env-dependent.

6. **Commit messages: short subject, why not what.** Detail belongs in
   `docs/`, not the git log.

---

## #11 — techniques do not reliably change the model

**Why it matters most.** A campaign now runs 30 steps. If choosing technique X
does not change the generated `train.py`, those steps produce identical scores
and the research loop is theatre. This happened: 12 hypotheses once produced
byte-identical training code and a single score.

**What is already true** (do not redo):

- A technique resolves to one of five statuses — `none` / `applied` /
  `not_applicable` / `candidate` / `rejected` — in
  `execution/technique/resolver.py`.
- `execution/technique/registry.py` holds only techniques a **template gate** can
  execute deterministically. It is deliberately *not* a vocabulary; a name absent
  from it means "no deterministic recipe", never "not a technique".
- The `candidate` path goes to LLM codegen with the hypothesis description. It
  works: `SWA` produced MSE 194.80 → 190.97 that way (five LightGBM seeds
  averaged).
- `templates/tabular_regression_partitioned/train.py.j2` has real gates for
  `lag_features`, `rolling_features`, `aggregation_features`, each using
  `_driver_columns()` so they never touch the target.

**The task.** Establish, with evidence, whether a chosen technique changes the
emitted code — and fix the cases where it does not.

Suggested approach:

1. **Measure first.** For each technique the resolver can return, render the
   plan twice — once with it, once without — into the *same* directory (the
   renderer bakes `run_dir` into its output, so different directories make any
   two renders differ and the test proves nothing). Hash the result. Build a
   table: technique → does the code differ.
2. Expect two failure classes: (a) `applied` techniques whose template gate does
   not exist, so the flag is set and nothing reads it; (b) `candidate` techniques
   where codegen received the name but not enough description to act on.
3. Fix the ones that are genuinely broken. For (a), either add the gate or make
   the resolver return `not_applicable` — silently accepting a technique that
   changes nothing is the defect.
4. A technique that legitimately cannot alter this template must say so. That is
   information, not a failure.

**Verify:** the differ-table above, plus a campaign producing at least
two *distinct* scores from two different techniques. Baseline to beat: MSE
190.97.

**Pitfall:** do not "fix" this by making codegen more verbose. The question is
whether the *emitted program* differs, not whether the prompt did.

---

## #25 — technique vocabulary store

**Design is written:** `docs/research-os/autonomy-roadmap/design/10-technique-vocabulary.md`.
Read it first; it has the schema, the four statuses and their derivation rules,
and the tradeoffs already decided.

**Ship step 1 only** — schema + recompute + a **report**. No filtering. The whole
point is to read what the rules *would* do before anything acts on them. If the
report proposes demoting `SWA` (the one technique with real measured effect), the
rule is wrong and nothing has been lost finding out.

Step 2 (consumers filter by status) is a separate PR, after a human reads step
1's output.

**Concretely:**

1. `ALTER TABLE techniques ADD COLUMN status TEXT NOT NULL DEFAULT 'candidate'`
   plus `technique_status_history`. Bump `SCHEMA_VERSION` in
   `accessor/sqlite/migrate.py` (currently `10`; migration is
   `CREATE TABLE IF NOT EXISTS` throughout and re-runs safely).
2. `execution/technique/vocabulary.py` — derive status from evidence cards.
   **Reuse `ClaimPromoter.measured_effect`** rather than writing a second
   computation; two definitions of the same quantity is a recurring defect here
   (there were three of "runnable plan", two of "belief identity").
3. A report function and a CLI surface to print it.
4. One call in `conductor/loop.py`'s repair chain, after
   `rederive_beliefs_from_cards`. Order matters and is established there.

**Verify on a sandbox copy of the workspace** (rogii today: 116 techniques, 124 beliefs, 15 evidence
cards). Expected: `SWA` → `confirmed`; `the`, `Breath Focus practice`,
`3D garment modeling` → not `confirmed`; recompute twice changes nothing the
second time.

**Pitfall:** the tempting rule is "drop anything unusual". That deletes exactly
the novel techniques the system exists to find. §3 of the design fixes *unchanged
evidence count* as a success metric for this reason — the change must not be able
to look good by making the system smaller.

---

## #39 — test doubles, then delete the rule engines

**Start last.** This touches every agent file and `tests/conftest.py`.

**Context.** M14 phase 3 removes the 20 `_run_rule_engine` implementations. They
fired **zero** times across 73 micro-agent invocations on two different models,
and the failure they exist to catch (a model answering a JSON-only prompt in
prose) is unreachable while every role requires `structured_output` — which PR
#98 made non-relaxable. So deletion is safe.

**The blocker is the test suite, not production.** Deleting them (mechanical, via
AST — the removal itself is minutes) produces:

```
96 failures, 28 test files, 16 agents
LLMUnavailableError: CodeEngineerAgent requires an LLM and none is configured
LLMUnavailableError: RootCauseAgent requires an LLM and none is configured
…
```

`tests/conftest.py` sets `LABPILOT_DETERMINISTIC=1`, and roughly 80 tests
exercise agent logic with no LLM by falling through to the rule engine. **The
rule engines are load-bearing for the test suite despite never firing in
production.**

**The trap.** Every agent `output_model` is constructible with no arguments, so a
single generic conftest double turns the suite green in one edit. **Do not do
this.** Tests that assert real deterministic content would silently start
asserting empty defaults — 28 files of hollow tests, which is worse than leaving
the engines in place.

**Do instead:** give each of the 16 agents a stub client returning *meaningful*
JSON for its output model. Group by agent, not by test file. For each one, check
what the tests actually assert before choosing the stub's content — if a test
asserted the rule engine's specific output, either the stub reproduces something
equivalent or the test's intent needs restating.

**Then** delete, with this script:

```python
import ast, pathlib, subprocess
files = subprocess.run(["grep","-rl","def _run_rule_engine","src/labpilot"],
                       capture_output=True, text=True).stdout.split()
files = [f for f in files if "accessor/common/micro_agents.py" not in f]
for f in files:
    p = pathlib.Path(f); lines = p.read_text().splitlines(keepends=True)
    node = next(n for n in ast.walk(ast.parse("".join(lines)))
                if isinstance(n, ast.FunctionDef) and n.name == "_run_rule_engine")
    start, end = node.lineno - 1, node.end_lineno
    while start > 0 and lines[start-1].strip() == "": start -= 1
    p.write_text("".join(lines[:start] + lines[end:]))
```

Then in `accessor/common/micro_agents.py`: remove the fallback tail of `run()`
(it must raise `LLMDegradedError` instead), the abstract `_run_rule_engine` stub,
and `strict_llm()` / `STRICT_LLM_ENV` / `deterministic_allowed()` /
`DETERMINISTIC_ENV`, all of which become moot with nothing to fall back to.

**Verify:** full suite green *without* a generic double; spot-check three of the
converted test files to confirm they still assert something real; one campaign completing.

---

## #26 — blocked

LLM adjudication of candidate techniques. Needs #25's status store to exist
first. Do not start.
