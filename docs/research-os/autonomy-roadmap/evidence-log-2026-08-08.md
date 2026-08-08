# Evidence log — 2026-08-08

Everything found by restarting `research conduct` against
`rogii-wellbore-geology-prediction` after it had been dead for a day, and
driving it until it produced research again.

**Branch:** `docs/backlog-deferred-from-112` · **Result:** 1364 tests passing
(from 1257), 18 commits, [PR #113](https://github.com/sadiq18/labpilot/pull/113).

**Method:** nine campaign runs. Fix what the last run exposed, run again. Every
defect below was found by *running* the system; none was found by reading it.

---

## The state it was in

The workspace had not produced a result since 2026-08-06 21:37.

| | |
|---|---|
| Consecutive failed executions | **108** (E-033 → E-139, then more) |
| Every one | `ModuleNotFoundError: No module named 'catboost'` at `train.py:19` |
| Time to fail | ~33 ms |
| Sessions burned | 4 (S-021…S-024), **80 conductor steps**, 104 LLM calls |
| Stop conditions fired | **none** |
| `tasks_failed` recorded | **0**, on all four |
| `CodeEngineerAgent` invocations in the whole window | **1** |

The fix for the undeclared dependency had been on `main` since 2026-08-08
(`6539d4d`, generated code declares its own dependencies). It could not be
reached: the only step that would have applied it was marked `done`.

---

## Defects found

| # | Defect | How it presented | Fix |
|---|---|---|---|
| 1 | Record references reach agents' system prompts | `.labpilot/skills/*.md` carried `Keep: hyp:H-010` into six agents on every run | `is_record_reference` at the overlay write site (`e8fc78f`) |
| 2 | Overlays keep teaching a verdict the card reversed | Every overlay said `Avoid: SWA` — the only technique that ever improved the metric | Re-derive overlays from current cards at campaign start (`cea05b2`) |
| 3 | Derived files indistinguishable from the source of record | 19 of 19 plan projections said `ready`; the DB said `done=16, abandoned=3` | Stamp projections and card dumps as derived (`413d30d`) |
| 4 | `max_submissions=0` means "stop", not "do not submit" | Campaign ended at step 0 with one `stop` decision and no research | Cap fires only on a spent budget; submit removed from the allowlist (`ffd7d63`) |
| 5 | A crashed run publishes a completed one's metrics | E-147 died on `import catboost` and reported `rmse 13.957107` — E-003's figure from six days earlier | Do not emit `ExperimentCompleted` for a failed run (`c4f2036`) |
| 6 | Retry asks which task failed, not whether the artifact works | `run_smoke_test` **passed** a 624-byte `train.py`; only `run_training` failed, which is excluded from the code-suspect set | `_train_script_is_unrunnable` (`97b7751`) |
| 7 | The prompt cache is thread-confined | Every LLM micro agent "skipped"; codegen silently rendered a Jinja template for four campaigns | `check_same_thread=False` + `RLock` (`617cb1d`) |
| 8 | `ast.parse` stands in for "runs" | 624 bytes — a docstring and half a comment — accepted and written | PEP 723 termination + `__main__` guard at apply time (`ac15949`) |
| 9 | Failure text keeps the head, not the tail | Stored error was **1523 characters of `Loading train: 96%`** and no diagnosis | `failure_excerpt` collapses tqdm frames, keeps the tail (`5563cf8`) |
| 10 | The smoke gate does not run what training runs | Bare `python train.py` vs `uv run --script`: it could not see dependency faults | Smoke uses `training_command` (`5563cf8`) |
| 11 | A stdlib module declared as a dependency | Codegen declared `glob`; uv rejected all six dependencies | Strip stdlib names via `sys.stdlib_module_names` (`d1f9bec`) |
| 12 | A broken script on disk does not trigger a rebuild | The stdlib fix could not reach a file already written | On-disk validity feeds the retry decision (`d2680e7`) |
| 13 | The re-ask is silent | Re-queued `write_code` was told nothing about why the last attempt failed | `retry_reason` into the codegen prompt (`df19e59`) |
| 14 | A missing script counts as runnable | `except OSError: return False` answered "yes, it runs" for a `train.py` that is not there — defect 6's loop by another door | Return `True` (`2d62c2d`) |
| 15 | A rejection without a tool name crashes the offline policy | `f["gated_tool"]` beside `f.get("decision")` on the same row | `.get` on both (`2d62c2d`) |

---

## One shape, eight times

**A gate that does not test what it claims to test.**

| The gate | What it actually tested |
|---|---|
| `run_smoke_test` | that `python train.py` exits 0 — not that the training command works |
| `ast.parse` | that the file is spelled correctly — not that it does anything |
| `error[:1500]` | the *first* 1500 characters — where the progress bar is, not the traceback |
| retry logic | which task reported the failure — not whether the artifact is still broken |
| `_train_script_is_unrunnable` | whether a readable file is valid — not whether a file exists |
| `EXPERIMENT_COMPLETED` | that an execution ended — not that it succeeded |
| evidence-card dumps | what was believed when written — presented as current |
| plan projections | the status at creation — presented as the status |

Each reads as correct. Each says **pass**. That is what makes the shape
expensive: a check that crashes gets fixed, and a check that wrongly passes gets
*trusted* — and its existence stops anyone looking. "We have a smoke test" ended
the enquiry for four campaigns.

This is the same family as the nine counted in
[evidence-log-2026-08-07.md](evidence-log-2026-08-07.md) — *the guard exists and
its input is wrong* — seen from the other side: **the guard exists and its
question is wrong.** [M20](15-gates-must-fail.md) exists to close it.

## The absence that cost the most

Not a wrong check — **a missing one.** `evaluate_stops` can stop a campaign for
submissions, wall time, cost, metric target, plateau, operator pause and step
count. None of them is *nothing is working*.

So 108 failures in ~33 ms each looked exactly like a campaign in progress. There
is no failure-rate stop, `os_capability_gaps` has **0 rows** across all 33 sessions,
and `os_suggestions` has 0 — the machinery to record "I could not do this"
exists and nothing writes to it. Assigned to
[M19 step 1c](14-experiments-as-deltas.md), because a delta campaign that cannot
notice its own failure rate cannot produce the measurement step 2 decides on.

## Three corrections to my own diagnoses

1. **"The evidence cards are inverted; the knowledge base is untrustworthy."**
   Wrong. `repair_card_directions` had already run: EV-012 (SWA) was `accepted`
   in `research/evidence/`. I had read `artifacts/evidence_card_*.json`, a
   write-only dump nothing reads. Defect 3 above is the fix, and the misread is
   *exactly* correction #1 in the 08-07 log repeating.
2. **"`generate_plan` passing `write_projections=False` is the bug."** Wrong —
   line 58 already writes them via `upsert`. The real gap is
   `PlanStore.update_plan_status`, which writes the DB row only. I nearly
   "fixed" working code.
3. **"`resolve_maximize` returns `None`, so repair is a no-op."** Wrong — I
   passed `knowledge/research` where `knowledge/` was wanted. It returns
   `False` (minimise), correctly.

Three mistakes in test code, all from assuming a name instead of checking:
`str.splitlines()` splits on `\r` and defeated the tqdm collapse; `ResearchTask`
has no `error` field (it lives in `metadata["error"]`); the ordering field is
`order`, not `order_index`. The first was caught by its own test asserting 34 ≠ 1.

## Where it ended

| | before | after |
|---|---|---|
| Consecutive failures | 108 | **0** |
| P-019 | `abandoned`, failed 107× | **`done`** |
| Latest plan | P-019 (Aug 7) | **P-020 → H-052**, previously untested |
| `train.py` | undeclared `catboost` | PEP 723, runs via `uv` |
| Failure class | infrastructure | **research** |

The last row is the point. The current failure is
`ValueError: pandas dtypes must be int, float or bool. Fields with bad pandas
dtypes: Geology: object` — and the model **receives that exact text** in its
retry prompt and still does not encode the column. That is model capability, not
wiring, and it is the first honest research failure this workspace has produced.
