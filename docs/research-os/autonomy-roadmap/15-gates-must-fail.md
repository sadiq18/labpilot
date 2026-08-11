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
| 1 — every pass/fail module has a red-then-green rejection test | **done, and the markers are now earned rather than declared.** The requirement was per-*module* for one round, which let one marker stand for four gates; keyed on `capability:check` it surfaced **20 gates nobody had shown could say no**. Eight check nothing and declare it on their own evidence; twelve have a rejection test, each verified red-then-green. Every `rejects` marker is checked against the verdicts the run actually produced — see *The parser that had to go*, below |
| 2 — no verification path rebuilds a command production owns | **done.** The command was already shared; the *environment around it* was not. All **three** places that execute model-written code — both verification gates and `pip install` — now strip credentials the way `TrainingRunner` does, and all three are bounded in time with the timeout reported as a verdict rather than raised. See *The half of the command nobody shared*, below |
| 3 — `tests/fixtures/real_failures/`, dated and sourced | **done.** The 2026-08-08 corpus, previously inline across nine test files |
| 4 — a derived artifact re-derives or says it is derived | **in progress.** One stamp helper and one reader (`accessor/common/derived.py`) instead of a copy per writer, applied at the five **write sites** and enforced by reading the files back. Enforcing it found **four more unstamped views already shipped** — see *Four more of the same shape*, below. Auto-discovery of a future writer is not built |
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

### Four more of the same shape

Criterion 4 named four artifacts and each had been answered individually —
`repair_card_directions`, `rederive_beliefs_from_cards`, `repair_skill_overlays`,
and a staleness stamp on plan projections. What none of them had was a *rule*, so
the criterion asks for one enforced over the writers.

Writing that rule found four more views, already shipped and unstamped:

* **`comparison.md`**, written beside `comparison.json` by `write_comparison` —
  whose own docstring calls the JSON *"(source of truth)"* and the markdown
  *"(view)"*. The author knew.
* **`profile.md`**, beside the `profile.json` the same call writes.
* **`research_brief.md`**, rendered from `analyze.json` and *not written with
  it*: `research analyze --skip-hypothesize` rewrites the JSON and skips the
  brief, so the previous run's file survives and the next `plan create` feeds it
  forward. This is the one that matters most: four separate readers feed it to
  an LLM, and it is where the pair genuinely diverges.
* **`<execution>_report.md`**, written once from the workspace's `metrics.json`
  into `reports_dir` — the directory `WorkspaceProvider` rglobs into LLM context,
  so an unstamped copy travels further than any of the others.

A fifth candidate turned out already compliant and is worth recording:
`JournalProjector.render_markdown` is printed by `cli/reflect.py` and never
written to disk, so it takes the criterion's *other* option and re-derives on
every read. "Renders markdown from a source" is not the test; "a file persists
after its source moves" is.

#### The stamp belongs to the writer, and the strip belongs to one reader

Both halves were wrong first, and both were found by review rather than by the
tests written alongside them.

**The stamp went in the renderer.** Two callers render *live* rather than
persisting — `experiments compare --format markdown` recomputes whenever the
stored JSON records a different pair, and `plan show --format markdown` reads the
DB directly — and both were then told to "read the JSON" for a file that may not
exist or may describe something else. A stamp that misdirects is worse than none,
which is the failure it exists to prevent. Moving it to the four write sites
fixed that, and fixed it for plan projections too, where the stale warning had
been printed on live reads since before this branch.

**The strip was per-caller.** `research_brief.md` is the only one of these read
back as *machine* input, and of its three readers two stripped the block and one
did not: the codegen prompt, which spent 277 of its 3000 characters telling the
model to distrust the context it was being handed. The context provider emitted
it twice more, against an 8000-character retrieval budget. `read_derived` is now
the single reader, so the next consumer gets stripping without knowing to ask —
the same argument as the single stamp, on the other side of the file.

#### A stamp must not overstate either

The generic note read *"...which is the source of record **and may have changed
since**"*, and for two of the five that is false: `comparison.md` and
`profile.md` are written from the same object as their JSON, in the same call, so
the pair can never disagree. Worse, `comparison.md`'s warning named
`repair_card_directions` as the thing that would invalidate it — that pass
rewrites evidence cards under `research/evidence/` and never touches this file,
and the `comparison.json` production actually writes comes from
`evidence/builder.py` under a different schema. A reader following that pointer
would have opened the JSON, found the identical verdict, and believed they had
confirmed it against the source of record.

Overstating is the same defect as misdirecting, in the document whose entire
purpose is to be accurate about what it knows. The generic text now says only
what is true of every view — that it is a copy, and of what — and each writer's
`warning` says what its own risk actually is.

#### The stamp got its own source wrong, twice

`comparison.md`'s warning named a repair pass that never touches it. Then the
report added for the fifth view named `metrics.json` — a file that lives in the
*workspace*, is shared by every execution, and is overwritten by the next run, so
a reader following it to check a per-execution report finds different numbers.
Both were caught by review, not by the tests written beside them, and both are
the same mistake: asserting a relationship without checking that it holds. The
report now names `research/executions/<id>/evidence/`, which is per-execution and
durable.

**The routing needed testing as much as the helper.** Reverting any of the four
`read_derived` call sites to a plain `read_text` left the whole suite green,
including the codegen prompt — the regression an earlier round had already fixed
once. Every fixture that reached a reader used an *unstamped* brief, so the strip
path was never entered: the coverage was of the helper in isolation and of call
sites with nothing to strip. All four are now driven over a brief the real writer
stamped, and each reversion goes red.

**What is not built:** discovering a *future* writer automatically. The five are
enumerated by hand, so a sixth added later is invisible until someone adds it —
and the fifth was found by review, not by the rule.
Stated rather than left to be found, the same limit shape as the verdict
observer's.

### The half of the command nobody shared

Criterion 2 was named after the smoke gate building its own command instead of
calling `training_command`. That had already been fixed — and fixing it had
drawn attention to the argv and away from everything else `subprocess.run` takes.

`TrainingRunner` runs generated code with `env=child_environment()`, which strips
the operator's provider and Kaggle keys. Its docstring is explicit: that code
"has no business holding the operator's provider keys or Kaggle credentials."
Both verification gates ran the *same generated code* without it — the smoke gate
passed `{**os.environ, "LABPILOT_SMOKE": "1"}`, and the unit-test gate passed no
`env` at all and inherited everything.

So the first thing to execute model-written code was the most permissive path in
the system, and the one place the rule was written down was the only place it was
applied. Both gates now build from `child_environment()`; the smoke gate layers
`LABPILOT_SMOKE` back on, which is the single difference it is entitled to.

This is criterion 2's real content, and the reason its wording says *command*
rather than *argv* was luck rather than foresight: sharing the argv and forking
the environment is the same defect, and it is the half that decides what a
hostile dependency can reach.

**The test for it captures what the gate passed**, rather than reading the
capability for the right call — criterion 1 spent seven review rounds inside a
source parser learning that. A mutation sweep then caught the fixture: the
comparison against `training_command` was satisfied by a hand-built
`[sys.executable, str(train)]`, because a script with no PEP 723 block produces
that argv either way. Both script shapes are covered now, which is the rogii
2026-08-08 failure — a declared-dependency script the gate ran under bare
`python` — surviving one round inside the test written to catch it.

#### Two of three is not done

Reviewing the above found the third place, and it was the worst: `pip install -r
requirements.txt` passed no environment at all. Installing a package **runs** it
— `setup.py` or a PEP 517 backend executes during the build — the requirements
file is written into the workspace where codegen chooses its own paths, and
`install=True` is the production default. A typo-squatted name was enough to run
arbitrary code holding every key the other two paths had just been taught to
withhold.

The instructive part is not the miss but the framing that produced it. The
criterion says *verification path*, the installer is production, so it was
filed as out of scope and the row was marked **done** — while the argument used
to justify the change ("unreviewed code must not hold the operator's keys")
plainly covered it. A scope that excludes the worst instance of the thing being
fixed is the wrong scope, and "done" on a criterion is read as covering the idea,
not the wording.

#### A bound is half a fix

The same review found the unit-test gate had no `timeout` while its sibling smoke
gate had one, so a generated `while True:` blocked the campaign with no verdict,
no evidence and no failure — the same file under the smoke gate returned in two
minutes. The installer had no bound either.

Adding `timeout=` alone would have been the wrong fix. `subprocess.run` raises
`TimeoutExpired`, and `engineer.py:227` calls `capability.execute` unwrapped, so
the exception escapes and no evidence is written — trading a hang for a vanish,
which is worse and is exactly what this milestone is named after. All three sites
now catch it and return `passed=False` with a `timeout` check, so running out of
time is a decision the card records rather than an absence.

Bounds: smoke 120s (unchanged), unit **600s** — a real generated suite may
legitimately take minutes and the bound is for hangs, not slowness — and install
**900s**, because a source build of a large wheel is slow and being killed
mid-build is a worse failure than waiting. All three are overridable via
`smoke_timeout_s` / `unit_timeout_s` / `install_timeout_s`, and each override is
now driven by a test: nothing had ever read any of those keys, including the
pre-existing one, so a rename would have fallen back to the default in silence.

#### The verdict was empty

Reviewing *that* found the next layer. Both timeout handlers wrote their own
one-line message and dropped `TimeoutExpired.output` — so a suite hanging in
`test_alpha` produced a log reading `pytest timed out after 600s` and nothing
else, while the success path writes returncode, stdout and stderr to the same
file. The harder failure produced the thinner record, which is the asymmetry
PR #121 fixed in `evaluation._infer`, reappearing inside the handler written to
stop a *different* silence. Writing "return a verdict, not an exception" in a
docstring did not prevent shipping a verdict with nothing in it.

Two details the fix turns on, both easy to get wrong:

- `TimeoutExpired.output` is **bytes on POSIX even when `text=True` was passed** —
  the exception comes from the inner `communicate()`, before decoding. Naively
  interpolating it writes a literal `b'collected 3 items\n'` into the log, which
  looks like a record and reads like an escape sequence. `stream_text` in
  `capabilities/_helpers.py` decodes with `errors="replace"`, since output from a
  process killed mid-write can end in half a character.
- `failure_excerpt` takes `stderr or stdout`, which is right for a crash — the
  traceback is on stderr. A timeout has no traceback, and the tail of *stdout* is
  what says how far it got, so a one-line stderr warning was enough to hide the
  test name entirely.

#### Three rounds, one cause: the fixture could not express the defect

Joining the streams did not fix the second point — it moved it. The excerpt keeps
the **tail** within 1500 characters, so a stderr longer than that evicted stdout
again: the same silencing, one layer down, inside the fix written to prevent it.

Ordering cannot solve that; whichever stream goes last wins. `stopped_excerpt`
in `capabilities/_helpers.py` gives each stream half the budget instead, so
neither can silence the other at any volume, and both capabilities call the one
function rather than keeping a copy each — which is criterion 2 applied to the
fix for criterion 2.

**The pattern underneath is worth more than the fix.** Every round of this PR was
verified by a mutation sweep, every sweep went red, and every round shipped the
next defect. The sweeps mutated the *code* and never the *input*:

| round | fixture | the limit it had to exceed |
|---|---|---|
| 1 | a `train.py` with no PEP 723 block | `training_command`'s two branches are identical for it |
| 2 | `TimeoutExpired(output=None)` | nothing to discard, so the discard was invisible |
| 3 | 39 bytes of output | the excerpt budget is 1500 |

A mutation sweep proves an assertion is wired to the code. It says nothing about
whether the *input* can express the failure — and a fixture chosen to look
realistic is almost always too small to. Demonstrated directly on this branch:
shrinking the flood below the budget and reverting the helper to the round-3
version leaves all 39 tests green; restoring the flood fails six.

So the practice, alongside the code sweep: **mutate the input too.** Shrink the
fixture and confirm the test loses power. If it stays green, the fixture was
decorative and the sweep was measuring nothing. For anything that truncates,
selects, or summarises, the input has to exceed the limit by construction rather
than by realism — and the assertion has to cover both directions, since raising
each stream's share from `limit // 2` to `limit` also survived until a test
asserted the *total*.

### The parser that had to go

`@pytest.mark.rejects("<capability>")` was, for seven review rounds, a claim
nobody checked. Deciding which gates *existed* meant parsing the capability
sources for `passed=` and resolving each to a name, and every round of review on
PR #121 landed inside that parser:

| round | what it missed |
|---|---|
| 2 | a `checks` list built as a variable, not a literal |
| 3 | labels joined with `+`, and an `if/else` picking between two |
| 4 | `no_verification` stamped from inside a nested block, exempting a gate that was not the one stamped |
| 5 | `name: str = "..."` — the form `BaseCapability` declares — not matching the pattern for `name = "..."` |
| 6 | `_by_type`, where a later `register()` for the same task type silently drops the earlier capability |
| 7 | one file holding two capabilities; a dict keyed by name dropping a duplicate; `inspect.getfile` raising on a class with no file |

Each fix was correct. Each left the next shape unhandled, and every failure was
silent — a gate the parser could not read was a gate it did not require a test
for. 36% of the file was AST machinery and all seven rounds were spent there.

The question is behavioural — *can this gate say no?* — and the runtime answers
it exactly. Every capability reports through `TaskEvidence`, which carries the
capability, the checks and the verdict, so `tests/helpers/verdict_observer.py`
records them as they happen and a marked test now has to have **caused** the
rejection it claims. The parser and its ~370 lines of tests are gone.

**Switching found two markers that had never been earned**, both previously read
and approved, neither visible to the parser — it could see that a marker existed,
never that the test rejected nothing:

| marker | what the test actually did |
|---|---|
| `code_engineering:modify_config` | greps the capability module for `passed=config_path.is_file()`. No gate decides anything while it runs |
| `dependency` | calls `strip_stdlib_dependencies` directly. A real test of a helper, not of the capability's verdict |

Both markers are gone and both tests stay, each carrying a note saying what it
does and does not prove. Neither capability lost coverage: `code_engineering` is
covered by two `write_code` tests, `dependency` by `pip_install`.

It also left a **finding against production**, not against a test:
`_modify_config` writes `configs/baseline.yaml` and then returns
`passed=config_path.is_file()` on the file it just wrote. No input reaches the
`False` branch — it is a sixth gate that cannot fail, in the same shape as the
five below, and it is not yet fixed.

**Round 8 found four defects in the replacement**, which is worth recording
plainly: leaving a mechanism because it kept producing defects did not stop the
next one producing them. Three were the same silent-claim shape the observer
exists to remove, arriving inside it.

| finding | what it was |
|---|---|
| `KeyError` → `INTERNALERROR` | `item.stash[_OBSERVED]` subscripted a key the line above read with `.get(..., [])`. Unset stash plus an unmet claim killed the whole session, not the test — the same one-guarded-read-one-unguarded shape round 7 was about, two lines apart again |
| a marker with no argument passed | `@pytest.mark.rejects` with nothing to check had nothing checked, so writing it wrong was quieter than not writing it |
| a fixture could earn the marker | the recorder starts before the test's other fixtures, so a rejection during setup satisfied the claim without the body proving anything. Latent when found — 0 markers relied on it |
| three marker forms out of four were invisible | `vars(module)` filtered to `test_*` missed methods of `Test...` classes and module-level `pytestmark`, both of which pytest honours. Latent — the suite has no test classes |

All four are fixed, each with a test written from the failure first and each
confirmed by mutation. Two of those tests were themselves vacuous on the first
pass and the sweep caught them: asserting the phrase *"no argument"* appeared
somewhere passed while the empty-string case was unhandled, because that case
also fails as an unearned marker; and a `MarkDecorator` unwrap survived deletion,
because `MarkDecorator` already proxies `.name` and `.args`. The counter-measure
that works is not *"leave the fragile mechanism"* — it is writing the failing
test first and mutating the fix afterwards.

The partial-install that made the first one reachable is now its own guard: the
observer is a fixture plus two hooks a conftest imports by name, any subset
imports cleanly, and `test_the_conftest_installs_every_hook_this_module_defines`
fails when one is missing. `pytest_plugins` would couple them properly, but
pytest honours it only in the rootdir conftest and this suite's is in `tests/`.

**The limit, stated rather than left to be found.** Observation sees the verdicts
the suite *reaches*. A verdict no test ever exercises produces nothing and is
invisible here — where the parser would have listed it, wrongly or otherwise.
That is a narrower blind spot than the parser's six, and unlike them it does not
report as coverage, but it is real: closing it needs the aggregate question
*"which observed checks were never observed failing?"*, which needs session-wide
collection and is not built.

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

**And the stamp reaches the card**, which it did not for a day. Written into
`TaskEvidence.checks` and read by nothing but a test is the same shape as
`delta_flags` sitting in a file no part of the system opened — the defect this
milestone found two rounds earlier, repeated on the same branch. `EvidenceCard`
now carries `unverified_steps`, and `decision_summary` names them beside the
verdict, so a conclusion drawn from a run whose unit-test step skipped for want
of tests says so where a reader meets it. Derived from metadata, so the three
writers that recompute `decision_reason` cannot drop it.

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
