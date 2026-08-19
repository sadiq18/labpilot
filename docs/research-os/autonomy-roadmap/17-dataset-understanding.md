# M22 — An inference without evidence is a guess

**Status:** not started · **Blocked by:** nothing · **Blocks:** [M23](18-baseline-correctness.md), [M25](20-eda-findings.md)

---

## Purpose

Re-profiling rogii with today's profiler infers the target column as **`EGFDU`**
— a horizon depth. The real target is `TVT`.

It has been wrong since the profiler learned to union test columns across
per-entity table kinds. `typewell.csv` carries its own `TVT` and ships in test,
so the horizontal well's `TVT` — the actual label, absent from horizontal test
files — stopped looking withheld, and inference fell through to the last
train-only column.

Nothing caught it for eleven days, because `prepare_workspace` reused the
`profile.json` written on 2026-08-02 and never re-derived it. The workspace was
one stale file away from training against a horizon depth, with no error
anywhere.

That is the failure this milestone is about, and the fix is not a better
heuristic. `tabular.py` records **five** rounds of PR #117 trying to make target
inference right, and names the pattern itself:

> *"position standing in for evidence"*

The rule will never be right. rogii proves a `typewell.csv` sharing the target's
name breaks a rule that works on titanic, spaceship-titanic and house-prices —
all three of which the current profiler gets **correct**, target and id.

**The goal is a heuristic honest about how much it knows**, so a weak answer is
visibly weak instead of silently wrong.

## What the profile cannot currently say

| It states | It cannot state |
|---|---|
| `target_column: "TVT"` | how sure, or on what evidence |
| `modality: "tabular"` | that a dataset is tabular **and** image |
| `warnings: [str]` | anything a decision can read — nothing parses them |
| — | that it does not know, and needs to ask |

`ModalityResult` has a `confidence` field. It is **never copied onto the
profile**, and `_llm_tiebreak` returns `"high"` on every path *including when
there is no LLM client at all*. `"ambiguous"` is unreachable downstream by
construction — a field whose value is fixed regardless of evidence is decoration,
not confidence.

## Four more defects of the same family

Each is a confident answer with nothing behind it:

1. **`row_count` is a lie on real data.** `playground-series-s6e7/profile.json`
   says `row_count: 100000, row_count_estimated: false`. The file has 690,088
   rows; `max_rows_sample` caps at 100,000.
2. **A metric is already mis-mapped on disk.** That same `competition.json`:
   `{"name": "balanced_accuracy_score", "key": "accuracy"}`. Balanced accuracy is
   not accuracy.
3. **`tabular.py:557` tells operators to "set `target_column` in the competition
   config".** No such field exists on `CompetitionSpec`. The profiler's only
   advertised escape from its worst failure mode is fiction.
4. **The zarr branch in modality detection is unreachable** — the
   `csv_count >= max(image_count, 1)` early return fires first, and every zarr
   competition ships a `sample_submission.csv`.

## The insight

Confidence is **not a probability**. It is a *coverage score over a fixed
evidence checklist* — how much of the evidence that would settle this question
actually fired. Two runs on the same bytes produce the same number. That is the
whole contract, and it is testable.

```
raw  = 1 - Π(1 - wᵢ)          # noisy-OR over fired signals
naming class contributes at most 0.20 in total
conf = min(raw, *hard_caps)
```

Worked, house-prices `target_column`: `named_in_submission_header` .80 +
`train_minus_test_unique` .70 + `non_null_in_train` .20 +
`target_dtype_matches_metric` .30 + `target_is_numeric` .15 → **0.97**, derived
from the catalogue rather than chosen.

The catalogue is what retires five rounds of PR #117. The positional branch at
`tabular.py:281` — `overlap[1]`, which works on aerial-cactus by convention and
would pick `id` if the header were reversed — gets weight **0.10** and a hard cap
of **0.50**. The fix is not a better rule; it is a cap that makes the branch
visibly weak so the system asks.

Likewise `iid_split` caps at 0.75: it is the *residual* hypothesis, what you
conclude when nothing else fired, not something proven.

## Structure: two planes, extend in place

`DatasetSchema` is the evolved `DatasetProfile` — same class, alias kept.

- **Value plane** — flat, current names. Every existing consumer is unchanged.
- **Evidence plane** — `inferences: dict[str, Inference]` keyed by field name.
  `schema.confidence_in("target_column")`. Nobody unwraps anything.

`Inference` carries **no** `value`. The schema holds the fact and its
justification in two places that cannot disagree, because only one of them holds
the fact. It reuses `ExperienceFacet`'s shape (`memory/models.py:22`) — *"hints
are not treated as ground truth"* — adding machine-readable `signals` so
calibration is checkable, and `rejected` so a vetoed claim is recorded rather
than dropped.

Not wrapped: a wrapper guarantees two names for one fact, which `report.py` and
`derived.py` exist to prevent.

## Modality is a list, not a scalar

`modalities: list[ModalityPresence]` with `role` in `{primary, auxiliary}`.
rogii becomes `[tabular(primary), image(auxiliary)]` instead of the PNGs being
noticed at `modality.py:110` and thrown away. `modality: str` survives as a
computed mirror, so six string-reading modules change nothing.

Per-modality extensions — `TabularSchema`, `VisionSchema`, `AudioSchema`,
`TextSchema`, `EnvironmentSchema` — hang off a modality-agnostic core answering
six questions that have an answer for a row, an image, a clip **and** an episode:
`prediction_unit`, `identity`, `target`, `split`, `metric`, `sources`.

Audio has no support anywhere today. RL has no representation at all; today
`profile_directory` raises `FileNotFoundError: No CSV files found` on a
ConnectX-shaped competition.

## Ask, or warn and block

**Schema questions must never route through `--yes`, `maybe_approve` or
`auto_approve`.** This is the single most important boundary in the milestone.

`maybe_approve` asks *"may I do the thing you asked for?"* — auto-allow is safe
by construction, because the default is the answer the operator would give. A
schema question asks *"which of these facts is true?"*, where **there is no safe
default**, and because `_profile_is_current` reuses `profile.json`, an
auto-answer *freezes* the guess into the workspace for every later campaign.
That is rogii at larger scale.

- interactive → **ask**, with candidates and the evidence for each
- unattended → **warn and block**; the campaign stops with the question pending

`SchemaQuestion.id = sha256(competition | field | sorted candidates)`, so a
question is never re-asked — and a changed candidate set is genuinely a different
question that *should* be. Answers live in `schema_answers.json`, **not**
`profile.json`, because the latter is rebuilt on every `PROFILE_SCHEMA_VERSION`
bump and an operator's answer must survive a profiler upgrade.

## The LLM proposes; the data vetoes

A `BaseMicroAgent` reading the competition description, **not** given the
deterministic inferences — withholding our answer keeps the proposal
independent, so agreement is evidence rather than echo.

Every claim faces a named structural verifier. Three outcomes, all recorded:

- **confirmed** → one signal worth 0.10. Ten points, never more.
- **contradicted** → discarded into `Inference.rejected`; never touches the value.
- **nominated_and_verified** → only where the deterministic path produced
  *nothing* and every verifier passes; capped **0.55**, below the ask threshold,
  so it always asks. Only fields with a real structural verifier may accept a
  nomination.

Off by default.

## Exit criteria

1. Every `Inference.confidence` equals `combine(its signals)` — no call site
   writes a float. Enforced by a self-consistency test, not by review.
2. The value plane is **byte-identical** whether the LLM proposer is absent or
   returns deliberately wrong claims for every field. This is what makes
   "propose-only" a mechanism rather than a comment.
3. rogii resolves `target_column: TVT`, lists `EGFDU` among the alternatives with
   its evidence, and records a confidence below `asserted`.
4. The three competitions the profiler already gets right — titanic,
   spaceship-titanic, house-prices — keep identical values and score ≥ 0.85.
5. A dataset that is genuinely ambiguous produces a `SchemaQuestion` and, under
   `--yes`, **blocks** rather than answering itself.
6. rogii's profile carries `[tabular(primary), image(auxiliary)]`, not one winner.

## Traps

- **Do not put `value` on `Inference`.** Two copies of one fact, and this repo
  has been bitten by that pattern before.
- **Do not delete `warnings`.** Nothing parses it; four things render it, one of
  them the codegen prompt. Make it a computed view over structured `notes`.
- **Do not try to fix `overlap[1]` with a better heuristic.** Five rounds of
  evidence say that route does not converge. Cap it and ask.
- **Watch the codegen prompt budget.** `_summarise_profile` exists because a
  14,437-token prompt ended a campaign, and `files` was 61% of the profile. The
  evidence plane could easily be larger. It must be stripped from the prompt,
  keeping only confidence and the evidence lines for fields below `asserted`.
- **Do not implement `action_space` inference for RL.** No fixture exists and the
  output is unfalsifiable. Detect that it *is* an environment competition, cap
  confidence at 0.50, and ask.
- **`_llm_tiebreak`'s `confidence="high"` is laundered.** Nothing downstream may
  read it as certainty until the source is fixed.
