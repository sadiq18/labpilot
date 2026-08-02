# Evidence log — 2026-08-02

Everything found by driving `research conduct` against
`rogii-wellbore-geology-prediction` for a day. Recorded because the *rationale*
for each plan is here, and because several of these are the kind of defect that
is invisible until someone checks a file timestamp by hand.

**Branch:** `research-os-m7-competition-hardening` · **Result:** 677 tests
passing (from 608 with 3 failures and a 28-minute runtime), 20 commits.

---

## The competition, and why it was a good test

`rogii-wellbore-geology-prediction` is not an ordinary tabular task:

- 1,553 CSVs — one pair per well (`<well>__horizontal_well.csv`,
  `<well>__typewell.csv`), 773 train wells, 3 test wells
- Target `TVT`; `TVT_input` is present but **NaN in exactly the scored region**
- Scored rows are a contiguous **suffix** (~73%) of each test well — a
  predict-forward task, not an i.i.d. one
- Six train-only marker columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`,
  `BUDA`) that do not exist at inference

Every one of those properties broke a different assumption in the pipeline,
which is what made it valuable. A Titanic-shaped competition would have passed.

## Defects found

| # | Defect | How it presented | Fix |
|---|---|---|---|
| 1 | Profiler could not read partitioned datasets | `0 rows / 0 columns`, fell back to a filesystem inventory | Role detection by directory; target inferred from train/test schema diff ∩ submission header |
| 2 | Validation was leakage-blind | `cv_mse 3892` from a shuffled row split across 773 wells — adjacent rows are near-duplicates | `ValidationPlan` derived from the profile: `partition_suffix_holdout`, holdout 0.732 |
| 3 | Analysis was data-blind | Dataset profile only existed *after* a run, so planning never saw the data | `analyze` materialises and profiles `data/raw` |
| 4 | Codegen wrote fake results | `cv_accuracy: 0.0` on a regression task; submission header `id,prediction` instead of `id,tvt` — reported success | Template fallback before the stub; non-dry run fails loudly |
| 5 | LLM output silently discarded | `CompetitionPageAnalyzerAgent LLM path failed` → rule engine, every call | Robust extractor, then Ollama `format: "json"` |
| 6 | Model answered in prose | `'These rules outline the guidelines... Here are some key points:'` where JSON was required | Constrained decoding |
| 7 | Campaign could never run a 2nd experiment | `generate_plan` hardcoded `baseline=True`; baseline is idempotent | Resolve to top proposed hypothesis once a baseline exists |
| 8 | Campaign never gathered evidence | `analyze_competition` dispatched with `args={}` → `fetch_kaggle` defaulted False | Fetch on by default; volume 5 → 15, configurable |
| 9 | Wasteful re-analysis | Re-swept kernels/papers (~15 min) with 10 hypotheses idle | Gated on backlog ≥3 **and** 6h artifact cooldown |
| 10 | Plan queuing loop | `generate_plan` chosen on 5 consecutive steps, none executed | Blocked while a plan is unrun |
| 11 | Stale plan targeting | `plan P-008 status=done; need ready or in_progress`, 3 steps lost | `@latest` prefers runnable plans |
| 12 | Premature stop | Stopped at step 4/12 with target 39× away | Goal persistence (bounded override) |
| 13 | Nonsense hypotheses | *"Improve on H-BASELINE by applying **vit**"* — a Vision Transformer, on tabular regression | Cross-modality rejection |
| 14 | **Campaign runs were dry runs** | `run_experiment` finished in 0.9s, reported success, chosen 9 times | `dry_run=False` passed explicitly |
| 15 | Non-hermetic tests | 608 passing, 28m40s, silently making real ollama calls; machine-dependent | Liveness probe forced closed in `conftest` |
| 16 | Unbounded retries | Unauthenticated Semantic Scholar retried 429s 7× with backoff — most of a 45-min stall | 2 attempts when unauthenticated |
| 17 | Per-workspace dataset cache | New workspace re-downloaded 1.2 GB it already had | `LABPILOT_KAGGLE_CACHE_DIR` |
| 18 | LLM health undiagnosable | `llm_unavailable` in a profile with no way to tell why | `doctor` checks reachability + model pulled |

## Open issues found but not fixed

Recorded here because they were observed directly and would otherwise be lost.

- **Circular import in the reporting capability.** Importing
  `execution/capabilities/reporting/capability.py` standalone raises
  `cannot import name 'ExperienceExtractor' from partially initialized module
  labpilot.research_engine.memory.extractor`. It works in the normal path only
  because `engineer.py` imports in an order that resolves it — so it is latent
  and will bite the first test or entry point that imports it directly.
- **Metric-key inconsistency.** The selector defaults `tabular_regression` to
  `rmse` while the competition scores `mse`; `metrics.json` carried `cv_rmse`
  and `mse` in the same file. Harmless for ranking, fatal for a score *series*
  (see [M8](02-objective-loop.md)).
- **`record_suggestion` output is never read.** When an intent maps to no tool
  the Conductor records "Need capability/tool X" — the system naming the
  capability it lacks. Nothing surfaces it. Nearly free to expose in
  `conduct status`, and it turns the loop into a roadmap generator.
- **Two meanings of "capability".** The execution `Capability` classes (10 of
  them, covering all 18 task types) and the Conductor's tool catalog are
  different layers sharing one word. Guarantees confusion in design discussion.
- **`query_memory` is unverified.** The policy selected it during campaign 7; it
  was never confirmed to change any subsequent decision.
- **Hypothesis quality depends on the rule engine.** With the LLM path failing,
  `generate_candidates` produced `vit`, `cnn`, `Mixed` and `test` as techniques
  for a tabular regression. Cross-modality rejection now filters the worst, but
  the generator itself is only as good as the model behind it →
  [M14](09-llm-required.md).

## Two mistakes I made, and what they teach

**A test that encoded an assumption instead of a behaviour.** For defect 14 the
accompanying test asserted `"dry_run" not in _default_args(tool)` — the absence
of a key. It passed while the behaviour was wrong, because `run_experiment`
defaults `dry_run=True` in its own signature. The fix was reported as complete
and was not.

> Assert the effect, not the call. `assert args.get("dry_run") is False`.

**Reporting "dispatched" as "done".** Campaign progress was reported from task
status rather than from artifacts changing. `run_experiment` completing is not
`metrics.json` being rewritten. The final verification used a file-mtime watcher
and a before/after diff instead.

Both are the same disease the codebase has: *success reported without checking
the effect*.

## Approaches tried and rejected

Kept so they are not retried blindly.

- **Per-partition Ridge calibration of the anchor** (rogii template). Fit the
  observed head, project forward. Made things markedly worse: it calibrated on
  `TVT_input`, a column observed only in the head, and collapsed when it went
  missing. Even once availability-guarded it extrapolated badly across the
  regime change between the vertical head and horizontal tail. A comment marks
  the spot in `tabular_regression_partitioned/train.py.j2`.
- **Long-window linear extrapolation as the residual base.** Mean MSE ~1,335 vs
  ~255 for simply holding the last observed value. Slope error compounds over a
  long horizon. The slope survives as a *feature*; the flat value is the anchor.
- **Excluding the papers analyzer from campaigns for speed.** Reverted on
  instruction — the right fix was bounding the retry budget, not dropping an
  evidence source. Correct call: fix the cause, not the symptom.

## Modelling result

| | Score | Meaning |
|---|---|---|
| Original pipeline | `cv_mse 3892` | Shuffled split — measured nothing |
| Naive anchor (hold last value) | 226.3 | Honest reference the template reports |
| Partition-aware template | **194.8** | Honest, 154 held-out wells, beats its anchor |
| Competition winner | 4.668 | — |

194.8 is honest and ~20× better than the reported baseline, but not competitive.
The remaining gap is geosteering domain modelling — correlating the horizontal
well's gamma-ray log against the reference typewell to locate the bit
stratigraphically. No generic template infers that. The *mechanism* to discover
it now exists (hypotheses come from real Kaggle kernels); whether a small local
model proposes it is a separate bet.

## What the campaigns proved

By campaign 4 the loop genuinely closed: analyze → 12 hypotheses → `P-002→H-010`,
`P-003→H-013` → train on 773 partitions / 2.98M rows (`smoke: false`) → MSE
194.80.

By campaign 9 it was confirmed the **score never changes**, because the
technique never reaches the model. That single observation is
[M7](01-technique-to-model.md) and the reason this roadmap exists.
