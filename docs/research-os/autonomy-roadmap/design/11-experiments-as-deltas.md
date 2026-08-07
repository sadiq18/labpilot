# Design — M-11: an experiment is a change to its parent

**Status:** design · **Owner:** unassigned · **Supersedes:** the Jinja template pack ·
**Subsumes:** technique registry, `applied`/`candidate` split, template→labpilot coupling

---

## 1. Problem

Two mechanisms produce training code today, and both are wrong for different
reasons.

**Templates don't scale.** Seven Jinja templates cover a space that varies per
competition, per problem type, per dataset quirk. Measured on rogii: the registry
declares 12 executable techniques and **7 resolve `not_applicable` purely because
nobody wrote a gate**. The registry exists only to feed template gates — its own
docstring says so — so the whole `applied` vs `candidate` distinction is an
artifact of the template mechanism rather than anything about research.

They are also the third instance of a pattern already rejected twice: a curated
set answering an open-world question. `KNOWN_TECHNIQUES` went for this reason;
so did the proposed package allowlist.

**Whole-file regeneration is wasteful and lossy.** The parent's code is sent into
the prompt — up to 120k chars, measured at **~3,376 tokens or 46% of the prompt**
— and the contract then says *"always emit full overridden train.py"*. So the
system pays to send the parent, pays again to receive a near-copy, and every
regeneration is an opportunity to silently drop something that worked. The skill
tells the model to "keep what worked", which is an instruction fighting its own
mechanism.

### Why this matters beyond cost

The partitioned template encodes `partition_suffix_holdout`, `_driver_columns()`
and three leakage gates — validation discipline fixed once, after a real leakage
bug. Under whole-file regeneration that discipline is re-derived on every run and
can be lost without any metric showing it: a leaky score looks *better*, not
worse. That risk is the only reason templates still looked load-bearing.

---

## 2. The change

**An experiment is a diff against its parent, not a fresh file.**

The validation protocol then survives **by construction** — it lives in the
parent and the delta does not touch it. No template needed to carry it, and no
after-the-fact contract to verify.

This also aligns the code artifact with the model the rest of the system already
uses. Evidence cards compare `parent_cv` to `treatment_cv`; the experiment graph
is parent → child; `technique_attribution` credits the difference. Only the code
was a fresh object each time. After this it is `parent + change`, matching what
every downstream consumer already assumes.

```
competition start   ──▶  baseline      whole file, from the dataset profile
                              │
hypothesis 1        ──▶       ├──▶ edit set ──▶ child A
hypothesis 2        ──▶       ├──▶ edit set ──▶ child B
hypothesis 3        ──▶  (on A) ──▶ edit set ──▶ child C
```

---

## 3. Requirements

**Functional**

1. Codegen emits **anchored edits** against the parent, not a whole file, for any
   plan with a parent execution.
2. A baseline (no parent) still emits whole files.
3. An edit that cannot be applied is **detected and named**, never applied
   partially.
4. A failed edit set is re-asked with the reason, then falls back to whole-file
   emission. The result records which path produced it.
5. Changes to validation logic are possible but visible — a delta touching the
   validation region is flagged on the evidence card, because a hypothesis about
   the metric is different from a hypothesis about the model.

**Non-functional**

- Prompt cost for a child experiment drops by roughly the size of the parent
  file: the parent is still sent as context, but the reply is a few hundred
  tokens instead of a few thousand.
- Edit application is deterministic and offline — no LLM in the apply path.

---

## 4. Why anchored edits, not unified diff

| Format | Model reliability | Failure detectable? | Chosen |
|---|---|---|---|
| Unified diff | poor — line numbers and hunk headers drift | partially (patch rejects) | no |
| Whole file | high | n/a — silently loses things instead | baseline only |
| **Anchored edit** — exact `find` text + `replace` | high; no counting required | **yes, exactly** — the anchor is present or it is not | **yes** |
| Structured op list (`add_feature`, `set_param`) | high | yes | rejected: it is a closed vocabulary, the same trap again |

The anchored form is what makes failure *loud*, which is the property this
codebase keeps needing. An anchor that does not match is a fact, not a judgement
— and it retries with the reason named, the same mechanism that fixed prose
replies to JSON prompts (`_is_shape_error` → corrective re-ask).

The structured-op alternative is rejected on the same grounds as templates: it
can only express changes someone anticipated.

---

## 5. Schema

`CodeFileSpec.action` is already `Literal["write"]` — a single-member literal,
which is the natural extension point.

```python
class CodeEdit(BaseModel):
    """One anchored replacement within an existing file."""
    find: str       # exact text from the parent, unique within the file
    replace: str
    why: str = ""   # shown in the failure message when the anchor misses


class CodeFileSpec(BaseModel):
    path: str
    action: Literal["write", "edit"] = "write"
    content: str = ""            # action="write"
    edits: list[CodeEdit] = []   # action="edit"
```

**Application rules**

- Every `find` must occur **exactly once** in the current file. Zero occurrences
  is a miss; more than one is ambiguous and is also a miss — a delta that could
  land in two places is not a delta.
- Edits apply against the **original** text, not against each other's output, so
  the set is order-independent and one edit cannot invalidate another's anchor.
- All-or-nothing. A partially edited `train.py` is worse than an unedited one:
  it runs, produces a number, and the number means nothing.

---

## 6. Failure ladder

```
emit edits ──▶ all anchors unique? ──yes──▶ apply ──▶ syntax check ──▶ run
                     │                                     │
                     no                                    fail
                     ▼                                     ▼
        re-ask, naming the missed anchor            re-ask with traceback
                     │ (bounded)                          (bounded)
                     ▼
        fall back to whole-file emission
                     │
                     ▼
          record generated_by="whole_file_fallback"
```

The fallback is deliberate and recorded rather than silent. Measured precedent:
when the JSON re-ask landed, the campaign that had been dropping to the offline
policy three times in eight steps ran 30 of 30. The same shape applies here — a
miss is recoverable if the model is told what it missed.

`agent_invocations` already records `failure_kind`; add `anchor_miss` so the rate
is measurable the way `json_shape` is, and the decision to keep or drop the
whole-file fallback becomes evidence-based rather than a judgement call.

---

## 7. What this removes

| Removed | Why it existed | Why it goes |
|---|---|---|
| 7 Jinja templates | deterministic code production | codegen produces better and less constrained code; `SWA`, the only measured improvement, came from the codegen path |
| `technique/registry.py` | feed template gates | no gates left to feed |
| `applied` / `not_applicable` statuses | whether a gate could execute a recipe | every technique is expressible as a delta; the honest statuses are the ones the vocabulary store derives from evidence |
| `from labpilot… import compute_metric` in 6 templates | shared metric helper | templates gone; a generated artifact must run standalone, not import the tool that produced it |

`prompt_technique_fields` and the hypothesis triad stay: the model still needs to
know *what* to try. Only the mechanism that turned that into code changes.

---

## 8. Risks

**The one that would hurt.** A delta can still damage validation logic — anchored
edits make it *possible* to change anything, they do not prevent it. Mitigation is
detection, not prohibition: flag when a delta's anchor falls inside the validation
region and record it on the evidence card. A hypothesis about validation is
legitimate; one that changes validation while claiming to test a feature is a
false result. Prohibiting it outright would block the legitimate case.

**Drift over a long chain.** Twenty deltas deep, the code is far from the
baseline and no single review saw the whole thing. The experiment graph already
records the chain, so the mitigation is a periodic whole-file re-emission as a
readable checkpoint — not a correctness measure, a legibility one.

**Anchor brittleness on formatting.** If a formatter runs between generations,
every anchor misses. Do not format generated code between runs; the syntax check
is the only automatic pass over it.

---

## 9. Testing

The failure modes are all "it looked applied and was not", so:

1. **A missed anchor never half-applies.** Two edits, one anchor bogus: the file
   is byte-identical afterwards.
2. **An ambiguous anchor is a miss**, not a first-match win.
3. **Edits are order-independent** — applying `[a, b]` equals `[b, a]`.
4. **The validation region is preserved** across a feature-adding delta: render a
   baseline, apply a realistic edit set, assert `partition_suffix_holdout` and
   `_driver_columns` survive byte-identical.
5. **The fallback is recorded**, not silent — `generated_by` distinguishes it.
6. **A delta touching validation is flagged** on the card.

Test 4 is the one that matters; it is the property that lets templates go.

---

## 10. Rollout

Behind `codegen.strategy: whole_file | delta` in config, defaulting to
`whole_file`. Both paths coexist while the anchor-miss rate is measured on real
campaigns — the standard M14 phase 2b set for exactly this kind of decision, and
the reason 2b shipped default-off with a number attached rather than a guess.

1. Schema + apply + tests. Nothing calls it.
2. Skill teaches the edit format. Opt-in via config.
3. Measure `anchor_miss` over campaigns.
4. Flip the default when the rate justifies it; delete templates in the same
   change that makes delta the default, never before.

Templates are deleted **last**, and only once delta is the measured default —
the same discipline applied to the rule engines, where the precondition and the
deletion had to ship together for the deletion to be safe.
