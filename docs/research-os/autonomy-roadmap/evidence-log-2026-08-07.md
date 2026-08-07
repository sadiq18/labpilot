# Evidence log — 2026-08-06/07

Everything found by taking M10 routing live and then driving `research conduct`
against `rogii-wellbore-geology-prediction` until it produced a result.
Companion to [evidence-log.md](evidence-log.md) (2026-08-02).

**Branch:** `research-os-m7-technique-identity` · **Result:** 922 tests passing
(from 749), 16 commits across PRs #91–#94 and follow-ups. **First distinct
experiment scores in the system's history.**

---

## The headline

> **MSE 194.80084243002463 → 190.97471945924474.**

Every experiment before this returned `194.80084243002463`, identically, across
nine campaigns and twelve hypotheses. That number was the roadmap's founding
evidence that the loop was structure without function.

| | |
|---|---|
| Produced by | `SWA`, resolved as **`candidate`** — no registry recipe |
| Implemented by | LLM codegen, from the hypothesis description (`technique_origin: llm`) |
| Attribution | `{"SWA": -3.826122970779892}` — exact match to the delta |
| What the code did | five LightGBM models on seeds 42–46, predictions averaged |

`SWA` is a neural-network technique and this is a tree model, so the generated
code was read before the result was believed: the model translated the
*principle* (weight averaging) into the tree-model equivalent (seed ensembling),
applied consistently to validation and test. A sound reading, not a literal one.

**This is what the hybrid decision bought.** Under M7's original F4 — "unknown
technique fails loudly" — `SWA` would have raised and aborted the run. §8.7 made
it a `candidate` instead, and it produced the only improvement the system has
ever found. The operator's argument for that design was: *"if some bugs comes in
producer and we start rejecting it, we might lose some good technique."*

Caveats: n=1, cross-validation not leaderboard, and ~2% from seed ensembling is
unremarkable in itself. **The recipe path contributed nothing** — every template
gate built this session remains unexercised in a real campaign.

## One defect, six times

Every wiring defect found had the same shape: **the plumbing is present and one
connection is not made.** That is the roadmap's founding diagnosis recurring at
finer grain, and it is worth naming as a pattern rather than five bugs.

| # | Site | The missing connection | How it presented |
|---|---|---|---|
| 1 | `Scheduler.dispatch` | never given `llm_client`; dispatched tools with `task.args` alone | `CodeEngineerAgent requires an LLM and none is configured` — while `research doctor` reported `codegen -> groq-llama70b` |
| 2 | `check_llm_roles` | called `load_config()`, which reads only the package default | doctor silent about roles in exactly the workspaces that configure them |
| 3 | `_render_template_fallback` | took no `plan_meta`; renderer call discarded all four kwargs | 12 hypotheses, byte-identical `train.py`, one score |
| 4 | `update_hypothesis_from_local` | had no parameter to carry the client to combo minting | a *successful* experiment aborted its learn-from-outcome step |
| 5 | `ClaimPromoter` | gated on confidence; never consulted measured effect | `"vit improves the primary metric"` — `supported`, while both vit runs scored identically to baseline |

| 6 | `revalidate_claims` | counted attribution computed from placeholder scores (`parent_cv=0.0`, `treatment_cv=0.5`) | a +194.8 "improvement" against a control that never ran, which is the whole basis of the vit claim |

| 7 | `build_evidence_card` | `maximize: bool = True` — a default **no call site overrode**, on a competition whose profile said `minimize` | every accept/reject verdict inverted; the one genuine improvement recorded as `rejected` |

Defect 1 is the one that mattered most: **the LLM reached the Conductor's policy
but not its hands.** Defect 6 is mine, and is the reason the repair took three
attempts. Defect 7 was found while addressing review comments on the fix for
defect 6, and is the most consequential of the seven — see below.

### Defect 7: the compass pointed the wrong way

Every conclusion the engine draws is signed. `cv_gain` is `treatment - parent`,
and whether that number is good news depends entirely on the metric's direction.
`build_evidence_card` took the direction as a keyword argument with a default of
`True`, and neither of its two call sites passed one. rogii minimises MSE.

The result, across all 15 cards on disk:

| Card | Measured | Recorded | Should have been |
|---|---|---|---|
| EV-012 | `SWA` cut MSE 194.80 → 190.97 | `rejected` | **`accepted`** — the only genuine improvement the system ever produced |
| EV-015 | MSE rose 194.34 → 194.80 | `accepted` | `rejected` |

The direction was never missing. `metadata.profile.metric.direction = "minimize"`
sat in the Analyze profile artifact the whole time; nothing read it. The
workspace's own `competition.json` has `metric: null`, so the profile artifact is
not a nicety — it is the only source that had the answer.

Fixed by making the default impossible: `maximize` is now `None`, resolved from
the competition profile (workspace copy → knowledge copy → profile artifact), and
**a direction that cannot be resolved raises** rather than assuming. A card is a
durable, signed conclusion; writing one whose sign is a guess is worse than
writing none.

### What defect 7 exposed

Re-orienting the cards flipped six of them from `rejected` to `accepted` — the
`0.50` stub runs, which against a 194.80 baseline now looked like a spectacular
MSE improvement. They had been rejected for the wrong reason, and correcting the
reason made them worse.

The runs themselves said so all along:

```
E-004: {'cv_accuracy': 0.5, 'status': 'dry_run_stub'}
E-001: {'cv_accuracy': 0.0, 'status': 'last_resort_scaffold'}
```

`training/capability.py` stamps `dry_run_stub` on a dry run and the fallback
script stamps `last_resort_scaffold`. Both markers were explicit, machine
readable, and unread — which is why they defeated every downstream check that
only asked whether *a* score was present. Seven of fifteen cards were built from
runs that trained no model, including EV-001, the sole basis of
`"vit improves the primary metric"`.

Two guards now sit at the point cards are minted:

- **`is_placeholder_metrics`** — a run reporting one of those statuses produces
  no verdict and no claim updates.
- **metric-key mismatch** — `_primary_cv` scans a list mixing `cv_accuracy` and
  `cv_rmse` and returns the first hit, so two runs could each answer from a
  different key and the builder would subtract an accuracy from an RMSE. It now
  records *which key* each side used and refuses the comparison when they differ.

This is the upstream fix that `_card_compared_something_real` could only describe.
At the claim layer a stub score is indistinguishable from a real one; at the
build layer the run itself declares what it is.

### The repair, measured

`repair_card_directions` runs at campaign start, immediately before claim
revalidation — cards first, because revalidation reads their verdicts and would
otherwise re-confirm an inverted one. Legacy stub cards are identified by looking
up their execution artifacts, which still hold the original metrics, so the
retirement is evidence-driven rather than a guess from the scores.

Against a sandbox copy of rogii's 15 cards:

| | Before | After |
|---|---|---|
| EV-001 (`vit`) | `accepted`, +194.80 | `inconclusive` — placeholder control |
| EV-002…007, EV-014 | `rejected` / `accepted` | `inconclusive` — placeholder runs |
| EV-012 (`SWA`) | `rejected` | **`accepted`** |
| EV-015 | `accepted` | `rejected` |
| `hyp:H-010` support | obs=5, net −971.50 | **obs=0** |
| `SWA` support | obs=1, net −3.83 | unchanged — a real result, kept |

Four claims contested. No workspace artifact was edited by hand: the campaign
heals its own memory on the next run, which is the standing rule for this system.

### The compass had two needles

Review of the fix found the same defect one layer down. `_claim_updates_from_attribution`
oriented the *sentence* by the direction-corrected `signed` value, but still chose
`ClaimEvidenceKind` and `confidence_delta` from the **raw** credit:

| Field | Was | Should be |
|---|---|---|
| `claim` | `SWA improves the primary metric` | ✓ |
| `evidence` | `contradict` | `support` |
| `confidence_delta` | −0.12 | positive |

`apply_card_to_beliefs` keys both the confidence step and the recorded `effect`
off `evidence`, so the card said SWA helped while teaching the belief store that
SWA is harmful and lowering its confidence. The half a human reads was right and
the half that steers was wrong.

The test that should have caught it asserted only `"improves" in claim` — a
weaker assertion than the defect. Both halves are now derived from `signed`, and
the tests assert polarity and delta through to the stored belief's `effect`.

**The generalisation, third time this session:** a value can be corrected at the
site you are looking at and stay wrong at the site that consumes it. Orienting
`_decide` did not orient the verb; orienting the verb did not orient the belief.
Each fix looked complete from where it was made.

### Guards that looked protective and were not

A sharper sub-class: the check exists, and its *input* is wrong. Three of these
in one session.

| Guard | Why it never fired |
|---|---|
| `ledger.py::_index_technique` | tested `normalize_label(name).startswith("hyp:")` — normalisation strips the colon, so the condition can never be true. Five `hyp:*` rows reached `techniques.name` through it |
| `_resolve_problem_type` | read `competition.json` (`unknown`) while `baseline_choice.json` said `tabular_regression`. An empty modality makes `filter_incompatible_techniques` return early, so the cross-modality filter was **disabled entirely** — `vit` ran on a tabular regression |
| `revalidate_claims` (mine) | keyed on the `effect` column; the one false claim has `effect=''` and carries its assertion in the *statement*. Of 417 claims it could only ever touch 14 — none of them the problem |

The third is mine, made *while* fixing the first two. The lesson generalises:
**check the field the bad record actually uses, not the field you expect to be
authoritative.**

## Fabricated research memory

`hyp:H-010` was the single most common "technique" in rogii's knowledge base —
**11 durable records**, ahead of every real one. The loop:

```
outcome.py appends `hyp:{hypothesis_id}` to an experiment's tags
  -> evidence attribution falls back to tags when no technique is set
  -> a belief is written claiming `hyp:H-010` is a technique
  -> hypothesis generation reads that belief
  -> the planner writes it to plan.metadata["technique"]
  -> codegen is asked to implement "hyp:H-010"
```

Six of ten plans carried one. A belief recorded `hyp:H-010` with
`effect=negative, status=rejected` — a fabricated *failure* of a technique that
never existed. Fixed at both attribution sites via one shared rule
(`shared/labels.py`), which insists on the **raw** label precisely because
normalisation defeated the previous attempt.

Existing records were **quarantined, not deleted** — 8 rows marked
`metadata.quarantined` — on the operator's instruction: the audit trail of what
the loop produced is itself evidence.

## Routing, measured

Free-tier limits were wrong in the catalog and only probing found it.

| Finding | Detail |
|---|---|
| Groq unreachable | Cloudflare rejects urllib's default User-Agent with **403 on every endpoint**. Our adapter sent none, so Groq failed like a bad key. Fixed with an honest `fitroute/0.1` |
| GitHub Models retired | `HTTP 410 github_models_retirement_brownout` — a provider the M10 design named as free **one day earlier** |
| Gemini RPD wrong by 10× | Catalog said 200/day (marked "not measured"); a live 429 gave `limit: 20` |
| TPM binds codegen, not RPM | A codegen call measured **14,437 tokens** against Groq's 12,000 TPM |
| OpenRouter has no TPM at all | Request-limited only (20 rpm; 1000/day with ≥$10 credits). A 15,102-token prompt returned valid JSON in **1.5s** |
| Two catalogued models do not exist | `qwen/qwen3-coder-480b:free` (400) and `deepseek/deepseek-r1:free` (404) — both taken from documentation rather than the live roster |

**Ordering error worth recording.** `default` was given to `groq-llama8b` for its
14,400 requests/day — but the *Conductor's policy* runs on `default`, is the
highest-frequency caller, and sends ~2.9k tokens per decision. Against 6,000 TPM
that is two decisions a minute, and a campaign died at step 7 with
`Limit 6000, Used 4882, Requested 2931`. **Daily budget was the wrong axis; per-minute
tokens is what binds the policy loop.**

### Prompt size

Measured rather than guessed, and the first guess was wrong:

| Field | Tokens | Share |
|---|---|---|
| `profile.json` | ~2,900 | 47% |
| ↳ **`files`: 200 filename strings** | **~1,770** | **61% of the profile** |
| `prior_train.py` | ~3,376 | 46% |

The generated code globs the data directory at runtime, so the filename list is
dead weight in every call. Replacing it with `{count, sample: [5], note}` cut the
profile **58%** with no information loss. The `stats` blobs — my first
hypothesis — were only 9.9%.

## Research memory repairs itself

A design property worth separating from the bug that prompted it. The operator's
instruction was: *"Make labpilot system smart enough to do it. Do it if migration
or cleanup need but learn from it again fix in labpilot so that it could not
repeat."*

So the cleanup is **not a script anyone runs**. `revalidate_claims()` executes at
the start of every campaign and again before every promotion cycle, contesting
any claim that asserts an effect no measurement supports. Three properties make
it safe to run unattended:

- **Contested, never deleted.** What the system once believed, and why it
  stopped, is itself research evidence. All 417 rogii claims survive; one
  changes status.
- **Direction-agnostic.** It refuses only on *no measured effect*, never on the
  sign of one — `SWA` scored **−3.83** for a genuine improvement on MSE, so
  inferring "positive" from sign would be wrong half the time. Judging direction
  needs the metric's optimisation sense and belongs with
  [M8](02-objective-loop.md).
- **Never fatal.** Repair failures are logged and swallowed; a campaign must not
  die because its memory could not be tidied.

The same rule that repairs old records prevents new ones, which is what stops it
being a migration.

## Campaign progression

Each fix moved the wall further out. Step count is *not* the metric — run 4 went
fewest steps and produced the breakthrough — but the stop *reasons* are the
record of what was actually blocking.

| Run | Steps | Stopped by |
|---|---|---|
| 1 | 5 | Conductor's hands had no LLM client |
| 2 | 6 | Gemini 20/day quota |
| 3 | 7 | policy on 6k TPM (routing error above) |
| 4 | 4 | gated tool treated as fatal — **produced 190.97** |
| 5 | 7 | policy asked for a gated tool six times; identical retries |
| 6 | 8 | reasoned stop, but fixated on `vit` with a runnable plan unused |
| 7 | **10** | offline catalog exhausted |

## Open, not fixed

- ~~**Claim revalidation does not fire.**~~ **Fixed.** Three defects, not the
  two first identified, and the third was the one actually blocking:
  1. it keyed on the `effect` column, while all seven effect-asserting claims
     carry `effect=''` and put the assertion in the *statement*;
  2. it ran only via `record_successful_execution`, so a campaign completing no
     experiment never repaired itself — it now also runs at campaign start;
  3. **it counted attribution computed from placeholder scores.** EV-001
     credits `vit` **+194.80** against `parent_cv=0.0`; EV-002–007 credit
     −194.30 against `treatment_cv=0.5`. A control of zero on a metric whose
     baseline is ~195 is a stub run, and EV-001 alone is why the vit claim read
     `supported`. Now gated on the evidence builder's own `decision` plus a
     zero-control check.

  Verified against a *copy* of the live workspace: contests exactly one claim —
  the false one — with all 417 preserved. Worth recording that the guard was
  written three times before it worked, and each failure was the same mistake
  this log documents elsewhere: **the check was sound, its input was not.**
  Tests built on invented fixtures passed throughout; only running it against
  real data exposed them.
- **The modality filter is too blunt.** `_MODALITY_TOKENS["vision"]` contains
  `cnn`, and tabular allows no vision tokens, so convolution is now blocked on
  every tabular problem. rogii is geosteering; convolution over a depth sequence
  is standard practice there. Applicability should be *derived from the data*
  (the profile already records `partitioned` and `partition_kinds`) rather than
  matched against a hand-written token list — the same argument that moved the
  technique vocabulary out of a Python constant.
- **`has_plan` is coarser than the adjacent `has_unrun_plan`**, so `run_plan`
  is dispatched for finished plans and fails. Burns a step per occurrence.
- **No runtime failover.** `RoleBoundClient.complete` records a failure and
  re-raises; selection is predictive only. An upstream 429 ends a campaign.
- **The recipe path is unexercised.** Gates exist and are tested; no campaign has
  applied one.

## Three corrections to my own diagnoses

Recorded because each was stated with more confidence than the evidence
supported, and the correction is the useful part.

1. **"All 17 plans are stuck at `ready`; nothing writes `done`."** Wrong — read
   from `knowledge/research/plans/*.json`, which are stale secondary artifacts.
   The authoritative store showed `done=15, abandoned=1, ready=1`, and
   `update_plan_status` exists and is used. There was no deadlock.
2. **"`_index_technique` correctly filters `hyp:`/`fork:`."** Wrong — the guard
   cannot fire (above). Asserted from reading the code without testing it.
3. **"Qwen3 Coder 480B is the strongest free option."** Taken from a blog post;
   it is absent from OpenRouter's live roster and returns HTTP 400. The live
   `/api/v1/models` endpoint is the only source worth trusting.

A fourth, smaller: the contract test in `test_technique_render_contract.py`
initially rendered variants into *different directories*, and `CodeRenderer`
bakes `run_dir` into its output — so it would have passed on the directory name
rather than the recipe. Caught by its own determinism control.

## What this session proved

The loop can produce a distinct, attributable, better result. That had never
happened before. It required a capable model (M10), the technique reaching the
executor (M7), and — decisively — M14 phase 2a's refusal to run without an LLM,
which converted four silent degradations into loud failures that could be found
and fixed.

What it did not prove: that the *recipe* path works in a campaign, that the
improvement replicates, or that a campaign can run to completion. The furthest
any run reached was 10 of 60 steps.


## M14 completed to the evidence, not past it

Six prerequisites shipped, then the campaign that both remaining phases were
blocked on finally ran. What it produced is more interesting than that it ran.

### The step ceiling broke

| Run | Steps | Ended by |
|---|---|---|
| previous best | 10 of 60 | — |
| S-020 | **27** | offline policy: catalog exhausted |
| S-021 (strict mode) | **30 of 30** | max steps |

Nine provider failovers across S-020 and S-021. Each one, before this session,
was a campaign-ending event.

### Two defects only a real run could have shown

**`returned no choices`.** OpenRouter answers some free models with HTTP 200 and
an empty `choices` array. Not an error status, not a rate limit, and nothing
usable came back — so the failover taxonomy, written from the errors we had
seen, let it through. It dropped the campaign to the offline policy three times
in eight steps while eight other providers sat idle.

**The policy was invisible.** `ConductorPolicy` is the highest-frequency LLM
caller in the system and is not a micro agent, so `BaseMicroAgent.run`'s
provenance never covered it. The first instrumented campaign recorded **one**
invocation across eight steps. The number M14 2b was blocked on was the number
the instrument could not see.

A third, same shape: provenance was installed around the *campaign loop*, so
`analyze` and `hypothesize` recorded nothing. Three campaigns yielded data for
two agents. Most of the 21 rule engines never run under the Conductor at all.

### `cool_down`, again

`BudgetLedger.cool_down` was written, tested, complete — and called from
nowhere. That is the fifth instance this session of *the plumbing is present and
one connection is not made*, and the second where the unconnected piece was
built specifically for the case it was not handling.

### What the measurement says

| | Invocations | Fallbacks | Rate |
|---|---|---|---|
| S-020, strict off | 27 | 3 (all `json_shape`) | 11% |
| S-021, strict on | 30 | 0 | 0% |
| cumulative | 94 | 3 | 3.2% |

2b ships **default off** on that basis. The honest reading of S-021 is narrower
than it looks: 30 of 30 invocations took the LLM path, so it proves strict mode
does not break a clean run, not that it survives a dirty one.

Phase 3's finding is sharper and also incomplete: across 37 micro-agent
invocations, `_run_rule_engine` fired **zero** times — every fallback belongs to
`ConductorPolicy`, which drops to the offline decision order instead. That is
evidence for deleting the rule engines and is not sufficient to act on: 13 of 21
agents were never invoked, and zero fires under M10 routing measures the model
rather than the engine. §11.4's run C is exactly this comparison, and we have
one half of it.


## Where the day ended

| Milestone | Status |
|---|---|
| **M7** technique → model | done — MSE 194.80 → 190.97; differ-table confirms the path is sound |
| **M10** LLM tiering | shipped and live, plus runtime failover |
| **M14** LLM required | **complete** — phases 1, 2a, 2b, 3 |
| **M18** technique vocabulary | shipped — status derived from evidence |
| **M19** experiments as deltas | designed, not started — the next build |

### The campaign, end to end

| Run | Steps | Ended by |
|---|---|---|
| best before today | 10 of 60 | — |
| S-020 | 27 | offline policy exhausted its checklist |
| S-021 (strict mode) | **30 of 30** | max steps |

### The pattern, counted

Nine defects this session shared one shape: **the guard exists and its input is
wrong.** Not nine unrelated bugs — one habit, nine times.

| # | Guard | Wrong input |
|---|---|---|
| 1 | `ledger.py::_index_technique` | `normalize_label` strips the colon it tests for |
| 2 | `_resolve_problem_type` | read `competition.json` while `baseline_choice.json` had the answer |
| 3 | `revalidate_claims` | keyed on `effect`, but the false claims carry `effect=''` |
| 4 | `_card_compared_something_real` | `bool(0.0)` rejects a legitimate zero |
| 5 | `build_evidence_card` | `maximize=True` default no caller overrode |
| 6 | `_claim_updates_from_attribution` | verb oriented, belief polarity not |
| 7 | `test_campaign_runs_are_not_dry_runs` | guarded `_default_args`; the offline policy bypassed it |
| 8 | `run_experiment` metrics check | asked "is there a file?", not "did *this run* write one?" |
| 9 | `test_no_candidate_reason_names_the_filter` | passed on whichever filter fired first on that machine |

Four were introduced by the same work that fixed the others. The countermeasure
that generalises is in `AGENTS.md`: **feed a guard a real bad record before
trusting it.**

### Rejected three times

A curated list answering an open-world question: `KNOWN_TECHNIQUES`, then a
package allowlist, then the Jinja templates. Each time the replacement was the
same shape — derive status from evidence, or bound the blast radius, rather than
enumerate what is permitted in advance.
