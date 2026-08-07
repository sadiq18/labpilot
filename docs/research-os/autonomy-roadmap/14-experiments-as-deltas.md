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
quality, not mechanism**, and that is what [M10](04-llm-tiering.md) already
manages.

So M19 ships an *adapter*: aider runs in a workspace copy, the diff becomes a
`CodeProposal`, and the existing validation and apply path is unchanged. That
keeps propose-then-apply, the never-edit-the-workspace rule, and M14's
provenance.

## Exit criteria

1. A child experiment produces a delta; a baseline still produces a whole file.
2. The workspace is untouched when a proposal is rejected.
3. Validation logic survives a feature-adding delta **byte-identical**.
4. Failure rate is recorded in `agent_invocations`, so making delta the default
   is an evidence-based decision rather than a judgement call.
5. Templates are deleted **in the same change** that makes delta the default —
   the discipline M14 phase 3 established, where a removal and the precondition
   that makes it safe must ship together.

## Risk worth naming

A delta makes it *possible* to change validation logic; running in a copy makes
it reviewable, not impossible. The mitigation is detection, not prohibition — flag a delta whose
anchor falls in the validation region and record it on the evidence card. A
hypothesis *about* validation is legitimate; one that changes validation while
claiming to test a feature is a false result.
