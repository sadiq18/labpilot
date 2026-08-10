# M20 — A gate that cannot fail is not a gate

**Status:** in progress — the mechanism landed 2026-08-09 · **Evidence:**
[evidence-log-2026-08-08.md](evidence-log-2026-08-08.md) ·
**Generalises:** M9 (verification-first), M15's contract test ·
**Blocked by:** nothing — every item is independent of the research loop

---

## Purpose

Fifteen defects were found on 2026-08-08 by driving `research conduct` nine
times. **Eight were one shape:** a gate that tests something easier than what it
promises, and passes.

| The gate | What it actually tested |
|---|---|
| `run_smoke_test` | that `python train.py` exits 0 — not that the *training command* works |
| `ast.parse` | that the file is spelled correctly — not that it does anything |
| `error[:1500]` | the first 1500 characters — where the progress bar is, not the traceback |
| retry logic | which task reported the failure — not whether the artifact is still broken |
| `_train_script_is_unrunnable` | whether a readable file is valid — not whether a file exists |
| `EXPERIMENT_COMPLETED` | that an execution ended — not that it succeeded |
| evidence-card dumps | what was believed when written — presented as current |
| plan projections | the status at creation — presented as *the* status |

Every one reads as correct. Every one says **pass**.

That is what makes the shape expensive. A check that crashes gets fixed within
the hour. A check that wrongly passes gets *trusted*, and its existence ends the
enquiry — "we have a smoke gate" was true for four campaigns while the smoke
gate was approving a file with no code in it.

## The relationship to M9 and M15

Neither is new. M9 says *"a tool that changed nothing did not succeed"*; M15's
contract test says *"different input must produce a different artifact"*. Both
are right, and both were pointed at **capabilities**.

M20 points them at the **verifiers**. The smoke gate is itself a tool that
changed nothing and reported success; nobody had asked it M9's question because
it *is* the thing that asks M9's question of everything else.

## The insight

**A guard is only proven by the failure it rejects.** Not by the passes it
allows, and not by review — four of the nine defects in the 2026-08-07 log were
guards that could never fire, and each had been read and approved.

`AGENTS.md` already records the countermeasure — *feed a guard a real bad record
before trusting it* — written after exactly that. It is advice, and advice does
not hold: three of the eight gates above were written *after* it.

So M20 makes it mechanical.

## What changes

### 1. A guard ships with the failure it rejects

Every check, gate, validator or guard carries a test that feeds it **the real
bad artifact** and asserts rejection. If that test cannot be written, the guard's
question is not yet understood.

The bar is red-then-green, not green: a test that passes both with and without
the fix has proven nothing. On this branch that was done by hand — stash the
fix, confirm the test fails, restore — and it caught one inert test.

### 2. Verification calls production, never resembles it

The smoke gate built its own command instead of calling `training_command`. Two
implementations of one idea, drifting apart exactly where it mattered.

The rule: a check invokes the same code path the real run uses. Where it cannot,
it says so. This is grep-able — look for anything test-shaped that constructs a
command, path or payload by hand.

### 3. Real failures become fixtures

Every fix on 2026-08-08 used the actual bad artifact: the 624-byte truncated
file, the tqdm flood, `Keep: hyp:H-010`, `glob` in a dependency list. They are
better inputs than anything invented, and they are currently inline across nine
test files.

A corpus — dated, sourced, reusable — makes the next guard cheap to validate and
stops the evidence being lost.

### 4. A derived artifact re-derives, or says it is derived

Three defects were one shape: a file written once from a source that later
changed — plan projections, evidence-card dumps, skill overlays — each silently
disagreeing with the database. Two separate wrong diagnoses came from reading
one, six days apart, by the same author.

The rule catches the fourth instance before it is written.

## Exit criteria

1. Every module under `execution/capabilities/` that reports pass/fail has at
   least one **red-then-green** test proving it rejects a real bad artifact.
2. No verification path constructs a command that production builds elsewhere;
   the duplicates are removed, not documented.
3. `tests/fixtures/real_failures/` exists, is dated and sourced, and the
   2026-08-08 corpus is in it.
4. Every derived artifact either re-derives at read time or carries a stamp
   saying it does not — enforced by a test over the writers, not by review.
5. A deliberately broken artifact fails the campaign **at the gate that owns
   it**, not three steps downstream. This is the check that cannot be satisfied
   by accident: it requires the gate to be both present and correct.

## Where this stands

| exit criterion | state |
|---|---|
| 1 — every pass/fail module has a red-then-green rejection test | **done, at verdict-site granularity.** The enumerator was per-*module* for one round, which let one marker stand for four gates; keyed on `capability:check` it surfaced **20 sites nobody had shown could say no**. Eight check nothing and declare it on their own evidence; twelve have a rejection test, each verified red-then-green. `_UNPROVEN_SITES` is empty |
| 2 — no verification path rebuilds a command production owns | not started |
| 3 — `tests/fixtures/real_failures/`, dated and sourced | **done.** The 2026-08-08 corpus, previously inline across nine test files |
| 4 — a derived artifact re-derives or says it is derived | not started |
| 5 — a broken artifact fails at the gate that owns it | not started |

### Three of the first nine rejection tests proved nothing

The sweep the criterion asks for — disable the guard, confirm the test goes red,
restore — was run over all nine. Three stayed green, and **review had passed all
three**:

| test | why it proved nothing |
|---|---|
| `code_engineering` | refused at an earlier precondition (*"missing dataset profile"*) and never reached the branch it claimed to test |
| `training` | a second, weaker copy of a test that already existed properly elsewhere. The marker moved to the real one; the copy went |
| `research_review` | drove `force_block`, a **test hook** — proving a gate through its own hook is close to circular, and with the hook disabled the capability failed anyway, for an unrelated reason |

That is this milestone's own claim landing on itself: each read as correct, and
each said pass.

**The lever matters as much as the test.** `training`'s first sweep disabled
`metrics=metrics if fresh else {}` and stayed green — that line blanks the
figure, while the verdict lives one branch up in `if ok and not fresh:`. A
red-then-green run against the wrong line proves nothing just as surely as a weak
test does, which is worth saying because the sweep is the thing everything else
here rests on.

### Five gates that cannot fail, found on the first day

The enumerator asks the question of the *code*, not of its tests: does this
capability have any path to `passed=False`?

| capability | what it reports | what it tests |
|---|---|---|
| `reporting` | 4 return sites, every one `passed=True` | nothing — it writes a summary and calls it verified |
| `runtime` | 2 return sites, both `passed=True` | nothing — it provisions a runtime and cannot report that it did not |
| `stub` | always passes | that it ran. Always passing is what a stub is *for*, and that is the point: on the card it is indistinguishable from a capability that verified something, which is how four campaigns ran with codegen silently falling back |
| `submission` | `passed=packaged.is_file()` | that a file exists — one **it wrote itself** moments earlier. Passes on a workspace with no model, no predictions and no data |
| `workspace` | `passed=passed` | that the directories exist. Run without Kaggle credentials it reports `passed=True` carrying `download_skipped: no_kaggle_config` and `profile_skipped: no_data` in its own metadata: it says it skipped everything, and passes |

**All five are fixed.** They were held as strict xfails for about an hour, which
turned out to be the right shape for exactly as long as it took to fix them — the
markers had to come off the moment the verdicts started meaning what they
promised, because a strict xfail that passes is a failure.

* `runtime` refuses to substitute the local default for a runtime that was asked
  for and could not be resolved;
* `workspace` separates *skipped because asked to* from *skipped because
  unable* — both were `None`, and the verdict read anything-but-False as done;
* `submission` no longer fabricates `id,prediction\n0,0` on a real run, so the
  file it checks for is one it did not write;
* `reporting`'s four verdicts ask whether there was anything to report, reflect
  on, believe, or suggest — the hypothesis one previously counted its own canned
  fallback string as a suggestion;
* `stub` declares `verifies = False` and stamps `stub_no_verification` on its
  evidence, which is M20's other option taken in the open.

The `workspace` fix immediately made three existing tests fail. All three were
about directory creation and idempotency and never wanted data — they simply had
never had to say so, because the old gate accepted silence. Declaring
`skip_download` in those three is the fix, and it is the gate working: an
assumption that was invisible is now written down.

Worth naming what the first three have in common with the eight in the table
above: none is a naming problem, and none would be caught by review. `reporting`
is honestly named and does report. The claim it makes is `passed`.

### The same shape one layer up: a fault that reads as an answer

The gates were the surface this milestone named. Sweeping the *decisions* built
on them found 137 handlers in the research loop that swallow without logging,
and seven where the swallowed value is what the conductor acts on:

| read | a fault used to mean | what that did |
|---|---|---|
| `_write`'s parent read | `prior_train = ""` | **an unreadable parent became "no parent"** — `_propose_delta` declines without it, so a permissions problem turned a delta experiment into a whole-file rewrite, on a card that said the step passed. M19's premise, lost to an `except` clause |
| `has_runnable_plan` | *nothing runnable* | a locked database stopped a campaign and looked like a finished one |
| `untested_hypothesis_count`, `viable_hypothesis_count`, `pool_counts` | *nothing queued* | M21's gathering gate stuck shut — the failure that module's own docstring exists to prevent, arriving through its error path |
| `_latest_plan_id`, `_next_hypothesis_id`, `_baseline_plan_exists`, `_latest_execution_id` | *nothing exists yet* | the conductor rebuilds a baseline over whatever is already there |
| `measured_effect` | `(0, 0.0)` | not "unknown" — *measured, and it was zero*. A claim about evidence, made because the evidence could not be read |
| `EvidenceCardStore.get` / `list` | the card is not there | a corrupted verdict reads as no verdict, and the promoter, the belief updater and the planner all act on the difference |

Every one carried a comment saying *"absent store means nothing yet"* — true of
the case its author had in mind, false of every other one.

**The fix is not a narrower `except`.** Absence is asked *first*, so the negative
answer is reached without an exception at all, and the handler is left holding
only genuine faults — which are then logged with their traceback, because the
value returned after one is a guess and the log is the only place that says so.

Three things that only appeared once the tests were written:

* store **construction** sat outside the `try` in three of them, so a corrupt
  database escaped the handler written for exactly that case;
* `HypothesisStore` is file-backed, not SQLite, so asking `knowledge.db` about
  hypotheses would answer *"none"* for a workspace full of them — the same
  mistake in the other direction. Absence has to name the store it stands for;
* `Path(None)` raises, and a `TypeError` escaping a question this calm would
  crash the conductor, so "no knowledge directory" is an answer rather than an
  error.

### What proving the twenty sites turned up

Two real defects, neither found by reading the code — both surfaced by trying to
write a test that made the gate say no:

* **`evaluation:compare` reported success on a card that compared nothing.**
  `passed=True` was unconditional, so a comparison with no control, or over
  placeholder metrics, passed — and the card is the thing COMPARE exists to
  produce.
* **`_infer` fabricated the artifact it then checked for.** It wrote
  `id,prediction\n0,0` when there was nothing to infer from, and reported
  `passed=pred.is_file()` on the file it had just written. The same defect as
  `submission`, one capability over, three weeks later.

And three tests of mine went green for the wrong reason before landing —
`code_engineering:apply` twice, because the proposal reached the `last_resort`
branch and the step failed *before* apply. The red-then-green sweep caught each;
reading them did not.

**Eight sites check nothing, and now say so.** *"no requirements file; skipped
install"*, *"no unit tests; skipped"*, *"runtime job already active"* — their
`passed=True` is honest about the step and dishonest about the card, where it
reads identically to a gate that looked and found nothing wrong. They stamp
`no_verification` in their `checks`, and the enumerator reads that from the
source rather than from a list in a test file, which would drift the moment a
branch changed.

## Traps

**Do not add a linter rule for "call it a gate".** The defects are not naming
problems; `run_smoke_test` is honestly named and still wrong.

**Do not test the guard against a synthetic bad input when a real one exists.**
A hand-written "truncated file" would have had no `# /// script` block, passed
the check, and taught nothing. The real one was truncated *inside* the block —
which is why it slipped through `ast.parse`.

**Do not fold this into M9.** M9 is a standing practice for capabilities and is
partly done; folding M20 in would let "M9 is partly done" absorb work that has
not started. They share a principle and not a surface.

**Resist widening a check to catch the last failure.** Twice on 2026-08-08 the
first fix I reached for was "add this task type to the suspect set" — and twice
the general form was to ask a different question (*does the artifact run?*)
rather than to extend the list. A list that grows once per incident is the
curated-set pattern wearing verification's clothes.

## Explicitly not in scope

**The campaign circuit breaker.** The absence that cost the most on 2026-08-08
was not a wrong check but a missing one: `evaluate_stops` cannot stop for
*nothing is working*, so 108 failures at ~33 ms each looked like a campaign in
progress.

That belongs to [M19 step 1c](14-experiments-as-deltas.md), not here. It is a
campaign-policy control rather than a verification one, and 1c needs it directly:
a delta campaign that cannot notice its own failure rate cannot produce the
measurement step 2 decides on.
