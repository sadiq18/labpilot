# SOP — Winning a Competition with LabPilot

Standard operating procedure for taking a Kaggle competition from zero to a
trustworthy submission using the `research` CLI.

This is the **operator's** guide: what to run, **what to check before trusting
the output**, and what to do when a step is wrong. Flag reference lives in
[../research-pipeline/CLI.md](../research-pipeline/CLI.md).

> **The one rule.** A number you did not verify is not a result. Most of this
> document is about checking that the system understood the problem before you
> let it spend hours optimising the wrong thing.

---

## 0. Setup (once per machine)

```bash
uv sync --extra dev
brew install libomp        # macOS, for LightGBM
```

For a local model (no API cost, fully offline):

```bash
ollama serve
ollama pull qwen2.5-coder:14b
```

**Share the dataset cache across competitions.** Datasets are immutable and
competition-scoped, so there is no reason to re-download gigabytes per
workspace:

```bash
export LABPILOT_KAGGLE_CACHE_DIR=~/workspace/.labpilot-cache/kaggle
```

---

## 1. Create the workspace

Workspaces are **client-owned** and live outside the LabPilot checkout, so your
competition data and results are never entangled with the engine's source.

```bash
uv run research init <slug> --path ~/workspace --git
cd ~/workspace/<slug>
cp .env.example .env          # then fill in KAGGLE_API_TOKEN
```

Every later command runs from inside the workspace:

```bash
uv run --project /path/to/labpilot research <command>
```

`labpilot.yaml` marks the workspace root; the CLI discovers it from the CWD.
Experience memory (`experiences.db`) is deliberately shared one level up, so
lessons transfer between competitions.

### Verify before continuing

```bash
uv run --project /path/to/labpilot research doctor
```

All non-optional rows must be **OK**. In particular `LLM provider` must show
the provider *and* the model — if it says `not pulled` or `unreachable`, every
downstream intelligence step will silently degrade to template text instead of
failing loudly.

---

## 2. Understand the problem

```bash
research analyze                      # add --fetch-kaggle for kernels + discussions
```

This downloads the data (reusing the shared cache), profiles it, stores
artifacts in `knowledge.db`, generates hypotheses, and writes
`research_brief.md`.

Because this is a Kaggle context, prefer **kernels over papers**: public
notebooks encode the tricks that actually score. Papers are worth pulling only
when the competition is genuinely novel.

### ✅ Gate 1 — did it understand the data?

Read the `[dataset]` notes. This is the single highest-leverage check in the
whole SOP. Confirm:

| Check | Why it matters |
|---|---|
| Row/column counts are non-zero and plausible | `0 rows / 0 columns` means the profiler fell back to a filesystem inventory and **everything downstream is guessing** |
| `target` is the column you expect | A wrong target silently trains the wrong model |
| Partition facts, if reported | `rows are NOT iid` changes what a valid CV even is |
| **train-only columns** | These do not exist at inference. If one of them is your target's near-copy, using it as a feature is a leak |
| Scored-suffix note, if reported | Marks a predict-forward task; random splits are meaningless |

Analyzers are slow on a local model, and the papers/repositories analyzers can
run for tens of minutes on network fetches. Narrow them while iterating:

```bash
research analyze --include competition,dataset --skip-hypothesize --skip-brief
```

Progress lines with elapsed time go to stderr, so `--format json` stays
machine-parseable.

---

## 3. Plan

```bash
research plan create --baseline
research plan show P-001
```

### ✅ Gate 2 — is the validation protocol right?

Open `baseline_choice.json` and read the `validation` block **before running
anything**:

```json
"validation": {
  "scheme": "partition_suffix_holdout",
  "holdout_fraction": 0.732,
  "exclude_features": ["ANCC", "ASTNU", "..."],
  "rationale": "scored rows form a contiguous suffix of each test partition ..."
}
```

The scheme is derived from the data, not from the template:

| Scheme | When | What it protects against |
|---|---|---|
| `kfold` | iid rows | — |
| `group_kfold` | one file per entity | Near-duplicate rows spanning the train/val boundary |
| `partition_suffix_holdout` | scored rows are a tail | Validating on a random split when inference must predict forward |

**If the scheme looks wrong, stop.** A wrong protocol produces a CV number that
is uncorrelated with the leaderboard, and every subsequent decision — feature
choice, model, hyperparameters — is then optimising noise.

---

## 4. Run

```bash
research run --plan P-001                 # add --dry-run for a smoke pass first
```

### ✅ Gate 3 — is the score believable?

Check `metrics.json`:

- **Compare against the naive baseline the template reports.** The partitioned
  template emits `anchor_hold_mse` alongside the model's score. If the model
  does not beat the naive anchor, the features are not earning their keep — do
  not submit, and do not tune. Fix the representation.
- **Sanity-check the magnitude** against the public leaderboard. A CV score
  hundreds of times better than the winning score means leakage, not genius.
- **Confirm the validation scheme** recorded in `metrics.json` is the one you
  approved in Gate 2.

---

## 5. Reflect, then iterate

```bash
research reflect run --execution E-001
research journal
research experiments report
```

Reflection turns the run into evidence, updates beliefs, and proposes the next
hypothesis. Then loop: `plan create` → `run` → `reflect`.

---

## 6. Submit

Submissions are **rate-limited**, so treat them as a scarce resource.

```bash
research submit --execution E-001 --dry-run     # verify the file first
research submit --execution E-001 -m "baseline: partition suffix holdout"
```

Only submit when local validation is both **better than the previous best** and
**trustworthy** (Gate 3 passed). Record the public score — `submit` writes it
back onto the hypothesis so later decisions can use it.

---

## 7. Autonomous mode

```bash
research conduct "Beat <target score> on <slug>" --max-steps 12
research conduct status
```

The Conductor observes, chooses a tool, seeks approval, dispatches, and loops
until a budget or stop condition fires. Use the staged commands above when you
want to debug *why* it chose something; use `conduct` when the loop is trusted.

Approvals default to on before a new plan batch and before any leaderboard
submit. Keep them on until you trust the campaign.

### How it decides — collect once, test many

The orchestrator shape is: gather evidence → ingest → techniques and beliefs →
hypotheses → plan → run → reflect, **looping on the tail rather than the head**.

Tools are filtered by precondition before the policy ever sees them, so the
campaign cannot spend a step on impossible or wasteful work:

| Tool | Withheld when |
|---|---|
| `reflect`, `submit` | nothing has run yet |
| `run_plan`, `run_experiment` | no plan exists |
| `generate_plan` | a plan is still unrun |
| `analyze_competition`, `search_papers` | ≥3 untested hypotheses queued, **or** evidence gathered <6h ago |

The two brakes on evidence gathering exist for different reasons: a full
hypothesis queue means the bottleneck is *testing*, and a recent sweep means
another one mostly re-ingests the same kernels under new artifact ids —
growing the store without adding information. Both signals appear in the
observe bundle (`untested_hypotheses`, `hours_since_last_artifact`), and the
skip reason is logged:

```
Skipping evidence gathering: 12 untested hypotheses already queued
```

**Goal persistence.** A policy that has used each tool once tends to declare
victory. While a `--target-metric` is set and unmet, an advisory stop is
overridden (up to twice) and redirected to reflect-and-try-the-next-hypothesis.
Budgets, `--max-steps`, and the real stop conditions are unaffected.

### Reading a campaign

Progress lines carry elapsed time, so a slow step is distinguishable from a
hung one:

```
step 2/14: observing + deciding …
step 2/14: chose run_experiment (26.1s)
```

Early stops with the objective unmet are recorded as suggestions — check
`research conduct status` for them; they are the system telling you which
capability it lacked.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `metrics.json` shows `status: last_resort_scaffold` | Codegen produced nothing and no template matched | Now fails the run loudly instead of continuing. Check `research doctor`, and confirm the problem type is inferable (a competition contract fixes `unknown`) |
| `submission.csv` header is `id,prediction` | Emergency stub wrote a placeholder | Same cause as above — the run should no longer reach submission in this state |
| Plan will not re-run (`status=done`) | Plans are single-use | Create a new plan (`plan create` is idempotent for the baseline; use a hypothesis plan to iterate) |
| Profile shows `0 rows / 0 columns` | Profiler could not resolve train/test roles | Check the layout is `train/` + `test/`; add a competition contract under `configs/competitions/<slug>.yaml` |
| `llm_unavailable` in analyzer notes | Provider unreachable or model not pulled | `research doctor` now names the exact cause |
| Command appears hung | Long analyzer or model call with no output | Progress lines with elapsed time are on stderr; narrow with `--include` |
| CV wildly better than the leaderboard | Leakage — usually a shuffled split on non-iid rows, or a train-only feature | Re-read Gate 2; check `exclude_features` |
| Tests behave differently on two machines | Ambient `.env` leaking into the suite | Fixed: the suite forces the local-model probe closed; keep new tests hermetic |

---

## Known limitations (be honest about these)

- **A generic baseline is not a winning solution.** The templates give a
  trustworthy, leak-free starting point. Closing the gap to a top score still
  needs domain modelling — on `rogii-wellbore-geology-prediction`, beating the
  naive anchor required anchoring and partition-aware features, and beating the
  *leaderboard* additionally requires correlating the horizontal well's gamma-ray
  log against the reference type well, which no generic template infers.
- **Local models are slow, and codegen is the weakest link.** A 14B model spends
  minutes per analyzer, and on `rogii` it produced no usable training code at
  all. The run now falls back to the deterministic baseline template rather
  than a placeholder, so a weak local model degrades to "solid baseline"
  instead of "silent garbage" — but the LLM path is where a stronger model buys
  the most.
- **The Conductor's action mapping is keyword-based.** It routes intents to tool
  chains by matching words, so novel intents fall through as "no capability"
  (recorded as a suggestion rather than silently dropped).
- **Hypotheses come from other people's work, not the system's own.** Candidates
  are mined from kernels, papers and repositories. Cross-modality suggestions
  are rejected (no ViT on a tabular task), but there is no
  reflection → hypothesis edge yet: the system does not turn *its own* failed
  experiments into new ideas. That, plus running evidence gathering as a
  background producer alongside a testing consumer, is the next real step.
- **"Good enough" is currently a count, not a judgement.** A backlog of three
  weak hypotheses blocks gathering exactly as well as three strong ones. The
  honest fix needs reflection outcomes: if the last N tested hypotheses all came
  back inconclusive or rejected, the queue is bad regardless of its size.
- **`metric_name` may not match the competition metric exactly** (e.g. `rmse`
  reported where the leaderboard uses `mse`). They rank identically for
  selection, but read the number with that in mind.
