# M19 — An experiment is a change to its parent

**Status:** steps 0, 1a, 1b, 1c and §6 provenance shipped — **step 2 in progress:
first delta experiment completed end to end, 2026-08-09** ·
**Design:**
[design/14-experiments-as-deltas.md](design/14-experiments-as-deltas.md) ·
**Supersedes:** the Jinja template pack ·
**Subsumes:** the technique registry, the `applied`/`candidate` split

---

## Purpose

Two mechanisms produce training code, and both are wrong for different reasons.

**Templates don't scale.** Seven Jinja templates cover a space that varies per
competition, per problem type, per dataset quirk. Measured on rogii: the registry
declares 12 executable techniques and **7 resolve `not_applicable` purely because
nobody wrote a gate**. The registry exists only to feed those gates, so the whole
`applied` vs `candidate` distinction is an artifact of the mechanism rather than
anything about research.

They are also the third instance of a pattern already rejected twice — a curated
set answering an open-world question, after `KNOWN_TECHNIQUES` and the proposed
package allowlist.

**Whole-file regeneration is wasteful and lossy.** The parent's code goes into
the prompt — measured at **~3,376 tokens, 46% of it** — and the contract then
says *"always emit full overridden train.py"*. The system pays to send the
parent, pays again to receive a near-copy, and every regeneration is a chance to
silently drop what worked. The skill's own instruction to "keep what worked" is
fighting its mechanism.

## The insight

**Express an experiment as a delta and the validation discipline survives
whenever the delta does not touch it.** `partition_suffix_holdout`,
`_driver_columns()` and the leakage gates live in the parent and are never
regenerated. That is not a guarantee — a delta *can* reach into them — which is
why detection is part of the work rather than an optional extra.

That is the only reason templates still looked load-bearing — a leaky score looks
*better*, not worse, so losing that discipline fails silently. Deltas remove the
need for a template to carry it.

It also aligns the code artifact with the model everything else already uses:
evidence cards compare `parent_cv` to `treatment_cv`, the graph is parent →
child, attribution credits the difference. Only the code was a fresh object.

## The edit machinery is bought, not built

A spike on 2026-08-07 ran `aider` against rogii's real 331-line `train.py`, asking
for the `SWA`-style change that produced this system's only genuine improvement:

| | nemotron-super-120b | **nemotron-ultra-550b** |
|---|---|---|
| Delta | +55 / −8 | **+24 / −8** |
| Self-doubt comments in code | 6 | **0** |
| Test half correct | no | **yes** |
| Validation discipline touched | **0 lines** | **0 lines** |
| Cost | $0.007 | $0.02 |

Both runs left `_driver_columns`, `_add_partition_features`, `_known_rows` and
`partition_suffix_holdout` completely untouched — the core requirement above, met
without labpilot writing a line of edit-format code. The variable is **model
quality, not mechanism**, and that is what [M10](04-llm-tiering.md) exists to
manage — though only if aider is pointed at fitroute rather than at the provider
directly. Passing `--model` transfers the selection and bypasses the budget
ledger, rate limiting and failover, so an OpenAI-compatible fitroute proxy is
step 0 of the rollout, not a nicety.

So M19 ships an *adapter*: aider runs in a workspace copy, the diff becomes a
`CodeProposal`, and the existing validation and apply path is unchanged. That
keeps propose-then-apply, the never-edit-the-workspace rule, and M14's
provenance.

## Where this stands

| step | what | state |
|---|---|---|
| 0 | fitroute OpenAI-compatible proxy | **shipped** (PR #110) |
| 1a | `CodeAgent` seam + hypothesis-consistency checks | **shipped** (PR #111) |
| 1b | wire the checks into the whole-file path, observe-only | **shipped** (PR #112) — four of §5's five checks |
| — | **validation-region flagging**, §5's fifth check | **not built** — see below |
| — | per-execution code provenance (§6) | **shipped** (PR #113) — `runs/<execution_id>/` |
| 1c | `AiderAgent` + copy/diff/propose + **campaign circuit breaker** | **shipped** (PR #115) — default off |
| 2 | opt-in via config; measure the failure rate | **in progress** — unblocked by M21; first delta measured, see below |
| 3 | flip the default when the rate justifies it | not started |
| 4 | delete templates in that same change | not started |

Step 1 was split once the consistency checks proved independent of aider: they
compare a parent to a child, and whole-file regeneration already produces that
pair. **1b comes before the adapter because the defect in "The check that
matters most" is already in production** — whole-file regeneration has the same
false-attribution failure and hides it better, so waiting for aider would leave
it running while building its replacement.

**1b found that the planned derivation cannot work.** It was going to read
`technique` off the plan metadata; on rogii's 19 real plans that field holds
hypothesis ids, category names, two techniques glued together, and once the
bare word `the`. The most recent campaign is 5 unusable in 9. Nothing maps a
technique name to a code identifier, and `TechniqueSpec` has no field for one.

The fix was to stop trying to derive the claim and ask the author for it.
`CodeProposal` now carries `kept` / `added` / `combined` as **code
identifiers**, named by the agent that wrote the file, so all four checks run
with nothing to maintain. Every evidence card records the verdict, the
violations, the wide-delta flag and the claim itself.

Self-reported, so a consistently lying model is not caught — but carelessness
is the failure that happens, and the gap between what the author says it did
and what the file does is exactly what makes attribution false.

**The fifth check was never built, and it is the one this milestone rests on.**
§5 lists preservation, addition, combination, confinement and **validation
region**; `consistency.py` implements the first four. Nothing detects a delta
landing in `partition_suffix_holdout`, `_driver_columns` or the holdout
construction — which is the mitigation §8 names for the only risk it calls *the
one that would hurt*, and the property that justifies deltas at all.

It does not depend on aider. Like confinement it needs no claim from the author,
so the argument that put 1b before 1c applies to it unchanged: whole-file
regeneration can silently drop the leakage discipline today, and a leaky score
looks *better*, not worse.

What stopped it is a design question, not effort. Defining the region as a list
of function names is the curated-set-answering-an-open-world-question pattern
this plan has already rejected four times — most recently as the technique→symbol
map that killed 1b's original derivation. The region has to be derived from the
parent or declared by the workspace, and that decision is unmade.

**Also outstanding before step 2 means anything:** the checks have still only
seen samples the author wrote — the same setup that produced both 1a bugs. Nine
campaigns ran on 2026-08-08 and none exercised them, because every `write_code`
in that run regenerated a whole file rather than proposing a delta. The
wide-delta threshold of 5 is calibrated on one 8-function file and stays a guess
until a second competition.

## 1c also owns the circuit breaker

**The absence that cost the most on 2026-08-08 was not a wrong check but a
missing one.** `evaluate_stops` can end a campaign for submissions, wall time,
cost, metric target, plateau, operator pause and step count. None of them is
*nothing is working*.

So 108 consecutive failures at ~33 ms each looked exactly like a campaign in
progress: four sessions, 80 steps, 104 LLM calls, no stop, and `tasks_failed`
recording **0** throughout. `os_capability_gaps` and `os_suggestions` hold 0 rows
across all 33 sessions — the machinery to record "I could not do this" exists and
nothing writes to it.

It lands here rather than in [M20](15-gates-must-fail.md) because it is a
campaign-policy control, not a verification one, and because **1c needs it
directly**: step 2 decides on a measured failure rate, and a campaign that cannot
notice its own failures cannot produce one. `aider_no_edit` and
`aider_syntax_fail` are counters with nothing watching the count.

Shape: stop when N consecutive executions fail, or when M steps pass with no
successful experiment, and say which. Roughly twenty lines beside
`evaluate_stops` — the difficulty is choosing N and M, and the 2026-08-08 record
gives the first data point either way.

**The full account of all fifteen defects is in
[evidence-log-2026-08-08.md](evidence-log-2026-08-08.md).** Thirteen were fixed
in PR #113; the eight that share one shape are why M20 exists.

## What 1c measured

**The adapter works.** Measured 2026-08-09 on rogii's real 331-line `train.py`,
same SWA-style request as the 08-07 spike:

| | delta | discipline touched | tokens | time |
|---|---|---|---|---|
| **diff (pinned)** | **+18 / −7** | **none** | **7.4k** | **16 s** |
| whole (aider's default) | +23 / −7 | none | 9.1k | 48 s |
| *spike, ultra-550b* | *+24 / −8* | *none* | — | — |

Tighter than the design's best recorded result, on a cheaper model, for 19%
fewer tokens — and the call was **metered in the budget ledger**, which was §4's
whole justification for building the proxy as step 0 and had never been tested.

**The edit format had to be pinned.** aider chose `whole` on its own: it has no
context-window or capability data for `labpilot/codegen`, a name that exists
only inside our proxy, so it falls back to the format that always works. Left
alone, 1c would have shipped the adapter and kept the whole-file cost. The
failure mode is a bill, not an error. Worth generalising — **routing by role
hides the model's identity from every client downstream**, so anything else
pointed at this proxy needs its defaults checked rather than trusted.

**§5's claim needed a new source.** aider returns a diff and no structured
claim, so every aider delta landed `delta_unchecked` — three of the four checks
going dark exactly when deltas arrived. `DeltaBriefAgent` now produces the
instruction *and* the claim from the hypothesis **before** aider runs, which is
what keeps the claim independent of the code it checks. Third mechanism proposed
for this job, after `technique` metadata and a technique→symbol map, and the
first that preserves the ordering §5 depends on.

## Why step 2 is blocked, and it is not the adapter

Four campaigns ran with `codegen.strategy: delta`. Not one produced a successful
delta experiment, and aider was right every time:

> *"The code currently: 1 Trains a LightGBM model … 3 Averages predictions from
> both models … Since no modifications are required, there are no
> SEARCH/REPLACE blocks to output."*

The hypothesis was **redundant** — `train.py` already ensembles two models. The
campaign kept re-selecting P-021 because nothing marks a hypothesis as already
implemented, so it stayed `proposed` and was chosen again.

Two consequences for step 2:

1. **`aider_no_edit` conflates two opposite findings** — "the model could not do
   it" and "the change was already there". A redundancy rate read as a failure
   rate would conclude delta does not work, from evidence that it does.
2. **The backlog gate is a ratchet.** `should_gather_evidence` shuts when
   `backlog >= 3`; rogii holds **46 `proposed`**, so `analyze_competition` and
   `search_papers` are removed from the allowlist permanently. The only thing
   that would refresh the pool is disabled *by* the pool. M16 already names this
   — *"a backlog is not a good backlog"* — and it is now load-bearing.

So the measurement step 2 needs is not available until hypothesis selection can
retire an idea it has already implemented. That work is sequenced in
[16-hypothesis-selection.md](16-hypothesis-selection.md).

## Step 2, first measurement — 2026-08-09

Both blockers are gone. `hypothesis_redundant` separates "already there" from
"the model could not do it", and the backlog ratchet is broken: rogii holds 46
`proposed` and `should_gather_evidence` now returns `True`, because the clauses
are ORed and the count is of *viable* rows.

The measurement was taken **outside a campaign** — `research plan create -H
H-015` then `research run -p P-022`. Four campaigns had been spent trying to
observe one delta; the direct run took minutes, because a campaign spends its
steps deciding what to do and this question does not need deciding.

**The adapter produced a delta on the real pipeline.** `agent_invocations`
records `DeltaBriefAgent` (llm) at 00:42:52 and `aider` at 00:44:05 with no
failure. `pipeline/train.py` changed **+34 / −4**: rolling-window statistics
grouped by `partition_id`, replacing a placeholder comment block. Exactly what
H-015 (`rolling_features`) proposed, on a 132-line file, from a hypothesis the
model had never seen a template for.

**And §5 caught that it changed nothing.**

```
delta_claim       = {"kept": [], "added": ["engineer_features"], "combined": []}
delta_consistent  = false
delta_violations  = ["'engineer_features' was supposed to be added,
                     but the result never calls or imports it"]
```

`engineer_features` is *defined* on line 45 and **never called** — `main()`
reads the data and goes straight to `feature_cols`. The rolling features are
dead code. The delta is clean, applies, parses, and does not run.

This is the "added but unused" false attribution the design named, met for the
first time on real output. Had the run completed, the card would have credited
`rolling_features` for a score computed without them — and the number would
have been real, which is what makes it dangerous.

Worth stating plainly because the first instinct was wrong: this reads at a
glance like a false alarm from a check that looks at calls and imports but not
definitions, and "fix the check to count `def`" is a two-line change that would
have passed its own new tests. It would also have blinded the check to the one
thing it exists to see. `tests/unit/test_code_engineering_delta_observe.py::
test_a_helper_that_is_defined_but_never_called_is_a_violation` is what stopped
it — a test written from an earlier rogii observation, holding down a rule
whose next challenger was a plausible-looking delta.

**The pipeline failure is not the delta's.** `run_smoke_test` failed on
`pandas dtypes must be int, float or bool. Fields with bad pandas dtypes:
Geology: object` — the same defect recorded on 2026-08-08, before delta
existed. Training never ran and the plan is `abandoned`, which is the honest
record: the gate refused a pipeline that does not work rather than reporting a
completed experiment.

### A complete experiment, and the eleven fixes it took

The delta above never reached a result. Getting one took a day of running the
real path and fixing what each run exposed — none of it found by review, and
each fix only visible once the previous one stopped hiding it.

**The pipeline could not run.**

| defect | what it did |
|---|---|
| the partitioned profiler read one file of one *kind* | `Geology` lived only in the other kind, so `profile.json` reported thirteen columns and all of them numeric. Codegen wrote "every column except this exclusion list" — correct given that profile, fatal given the data |
| `research ingest` raised on `hyp:H-010` | a record reference merged as a technique; the store's own error said to filter first, and no caller did |
| the codegen prompt never named the output paths | the script invented `/workspace/`, then `./workspace/` when told "relative paths only" — which *is* relative. Training succeeded and wrote its result where nothing reads it |

**The retry loop could not converge.** Four separate reasons, each masking the
next: `retry_reason` never reached aider's instruction; the brief overrode it
with the hypothesis anyway; the reason itself was the *head* of a traceback —
file paths, with the exception past the cut; and a training failure left
`code_is_suspect` false, so every retry rebuilt blind. Three consecutive runs
produced a nil delta while the error sat one field away.

**Verdicts were wrong in four ways.** Three of them are one rule — *a change
that cannot alter behaviour is not an experiment* — arriving through doors that
are invisible to each other:

* **unreachable new code**: aider wrote thirty-four correct lines of rolling
  features into a function `main()` never calls (`check_reachability`);
* **an unreachable parent**: a failed run leaves its edit behind, so the next
  attempt at the same hypothesis found `rolling` and `groupby` already present —
  inside that dead function — and retired it as already implemented
  (`check_redundancy` now judges live code only);
* **identical code**: handed the dtype error, a retry edited the module
  docstring. `touched_functions` compares function bodies, so it reported
  nothing touched; the claimed symbols were present from an earlier attempt; and
  `aider_no_edit` never fired because there *was* an edit (`check_effect`).

The fourth is the one that would have poisoned the measurement. `cv_rmse` went
**194 → 1382**, `comparison.json` recorded `decision: "rejected"`, and the
hypothesis was written **confirmed**. `_map_outcomes` read
`comparison["verdict"]` — a key nothing writes — so the measured verdict never
arrived, then fell through to reading `cv_delta`'s sign as though larger were
always better. Without a decision it now returns `inconclusive`: nothing at that
layer knows the metric's direction, and `confirmed` on a regression poisons
every ranking that reads it afterwards.

And `research run` reported **succeeded** against a `metrics.json` written the
previous evening — the stale-metrics hole `run_experiment` closed with
`_metrics_written_since`, still open on the Engineer path, which is the path a
plan actually takes.

**The completed run.** Execution E-234: aider added `MD_x_GR`, the consistency
checks passed, training produced real metrics over 1.36M rows, and the
comparator correctly rejected the result. First delta experiment to reach a
verdict.

### What this means for step 2

| exit criterion | state |
|---|---|
| 1 — child produces a delta, baseline produces a whole file | **met** |
| 2 — workspace untouched when a proposal is rejected | **not met**: a failed run leaves its edit behind, which is how the redundancy false positive above happened |
| 3 — validation logic survives a feature-adding delta byte-identical | **met** on E-234 |
| 4 — failure rate recorded in `agent_invocations` | **met** — `origin=aider`, brief recorded separately, kinds distinguish redundancy from adapter failure |
| 5 — templates deleted in the same change that flips the default | not started |

One experiment is not a rate. What step 2 still needs is N runs with
`codegen.strategy: delta` and the outcome counts read off `agent_invocations`.
The pool is ready for it — 40 viable hypotheses after `ingest` + `hypothesize
new`, concrete ones like `typewell_gr_mean` and `tortuosity_50` rather than the
`3D garment modeling` that used to fill it.

Two things to know before reading those numbers:

* **The parent is currently weaker than it was.** Forcing a from-scratch rebuild
  to clear the dtype defect discarded H-014's accumulated feature work — twenty
  features became six. Every delta measured against it is measured against a
  crippled baseline. Restore the parent or let the loop rebuild the features
  before treating any comparison as a finding.
* **`partition_suffix_holdout` is not an inverted split.** Training on 27% and
  validating on 73% looks wrong and is correct: `holdout_fraction` is the
  measured `scored_fraction`, and the scheme reproduces the predict-forward gap
  the test set actually has.

### A note on how these were found

Three times a fix that looked obvious was caught by the existing suite:
counting `def` as satisfying "added" would have blinded the check that catches a
dead delta; requiring reachability of every module would have condemned any
library; making `RUN_TRAINING` a code-validation task outright contradicts a
rule that exists so an OOM does not discard working code. All three would have
shipped green under their own new tests. The habit that keeps paying is
reverting the fix and confirming the new test goes red — and the habit that
keeps costing is inferring intent from a number instead of reading where it
comes from.

## Exit criteria

1. A child experiment produces a delta; a baseline still produces a whole file.
2. The workspace is untouched when a proposal is rejected.
3. Validation logic survives a feature-adding delta **byte-identical**.
4. Failure rate is recorded in `agent_invocations`, so making delta the default
   is an evidence-based decision rather than a judgement call.
5. Templates are deleted **in the same change** that makes delta the default —
   the discipline M14 phase 3 established, where a removal and the precondition
   that makes it safe must ship together.

## The check that matters most

A delta must do **what the hypothesis claimed** — not merely apply cleanly.

*"Ensemble LightGBM with CatBoost"* can be satisfied by replacing LightGBM, by
adding CatBoost and never averaging, or by adding CatBoost *and* quietly retuning
LightGBM. All three run and produce a card. The third credits the whole `cv_gain`
to "ensemble" when two things changed — a false attribution that no metric
reveals, because the number itself is real.

Deltas make this checkable for the first time: preservation, addition,
combination and confinement are AST facts about the change, and only labpilot can
test them because only labpilot holds the hypothesis.

## Risk worth naming

A delta makes it *possible* to change validation logic; running in a copy makes
it reviewable, not impossible. The mitigation is detection, not prohibition — flag a delta that lands in the
validation region and record it on the evidence card. A
hypothesis *about* validation is legitimate; one that changes validation while
claiming to test a feature is a false result.
