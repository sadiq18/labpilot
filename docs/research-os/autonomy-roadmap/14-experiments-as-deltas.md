# M19 — An experiment is a change to its parent

**Status:** not started · **Design:**
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

**Express an experiment as a delta and the validation discipline survives by
construction.** `partition_suffix_holdout`, `_driver_columns()` and the leakage
gates live in the parent; a delta that does not touch them cannot lose them.

That is the only reason templates still looked load-bearing — a leaky score looks
*better*, not worse, so losing that discipline fails silently. Deltas remove the
need for a template to carry it.

It also aligns the code artifact with the model everything else already uses:
evidence cards compare `parent_cv` to `treatment_cv`, the graph is parent →
child, attribution credits the difference. Only the code was a fresh object.

## Exit criteria

1. A child experiment emits **anchored edits**, not a whole file.
2. A missed anchor is detected, named, and re-asked — never half-applied.
3. Validation logic survives a feature-adding delta **byte-identical**.
4. `anchor_miss` is recorded in `agent_invocations`, so making delta the default
   is an evidence-based decision rather than a judgement call.
5. Templates are deleted **in the same change** that makes delta the default —
   the discipline M14 phase 3 established, where a removal and the precondition
   that makes it safe must ship together.

## Risk worth naming

Anchored edits make it *possible* to change validation logic; they do not
prevent it. The mitigation is detection, not prohibition — flag a delta whose
anchor falls in the validation region and record it on the evidence card. A
hypothesis *about* validation is legitimate; one that changes validation while
claiming to test a feature is a false result.
