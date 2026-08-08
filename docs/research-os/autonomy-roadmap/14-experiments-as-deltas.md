# M19 — An experiment is a change to its parent

**Status:** steps 0, 1a and 1b shipped · **Design:**
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
| 1c | `AiderAgent` + copy/diff/propose + per-execution provenance | not started |
| 2 | opt-in via config; measure the failure rate | not started |
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

**Also outstanding before step 2 means anything:** no campaign has run with 1b
in place. Its whole purpose is a false-positive rate before a check can cost a
step, and the checks have still only seen samples the author wrote — the same
setup that produced both 1a bugs. The wide-delta threshold of 5 is calibrated on
one 8-function file and stays a guess until a second competition.

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
