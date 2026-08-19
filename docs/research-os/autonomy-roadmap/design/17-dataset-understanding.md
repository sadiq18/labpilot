# Design — M22: Dataset Schema with confidence and evidence

**Plan:** [../17-dataset-understanding.md](../17-dataset-understanding.md) ·
**Status:** design · **Blocks:** [M23](../18-baseline-correctness.md),
[M25](../20-eda-findings.md) · **Must survive:** [M12](../06-beyond-kaggle.md)

---

## 1. Background

Every campaign this OS runs optimises over a description of the dataset that one
component produced and nothing ever checked. That component is
`accessor/profiler/`, and its output — `profile.json` — decides the target
column, the ID, the usable features, the validation split and the metric. M23
(baseline correctness), M25 (EDA findings) and M26 (feature specs) all read it.
M12 takes the OS off Kaggle entirely, which is where most of the profiler's
current evidence disappears.

---

## 2. Problem

The profiler states answers it has no way to justify, and there is no mechanism
by which a weak answer can look weak. Read on `main` at `cd35485`, and cited by
symbol wherever this milestone's own step 1 moves the lines:

| Site | Defect |
|---|---|
| `tabular.py` — the single-table branch of `profile_dataset`, and `_try_profile_partitioned` | Two target-inference paths. Both end in position deciding: `overlap[1]`, or `sorted(candidates)[-1]` after a tie |
| `tabular.py` — the `ambiguous_target` warning in `_try_profile_partitioned` | It says *"set `target_column` in the competition config"*. `CompetitionSpec` has no such field — the only advertised escape is fiction |
| `modality.py:34`, `_llm_tiebreak` | `confidence` is `"high"` on every path, including the no-LLM path |
| `workspace/capability.py:458-466` | The profiler is called **without** an LLM client, so that no-LLM path is the one production always takes |
| `modality.py:104-117` | The zarr branch is unreachable: the CSV-preference return fires first, and every zarr competition ships a `sample_submission.csv` |
| `tabular.py` — `profile_file`, and the profile built in `profile_dataset` | `row_count` is the length of a sample capped at `max_rows_sample`, and `row_count_estimated` stays `False`. On disk: `playground-series-s6e7/profile.json` says 100,000 rows, unstamped; the file has 690,088 |
| `competition/metrics.py:38-49` | Substring mapping: `balanced_accuracy_score` → `accuracy`. On disk in that competition's `competition.json` |
| `workspace/capability.py:503-580` | A second modality decision. When the profiler raises, this writes a valid-looking profile with `target_column: null` and prose in `warnings` |
| whole module | The profiler `rglob`s CSVs directly. No seam between *where data lives* and *what is inferred from it* |

**Measured cost.** rogii trained against a horizon depth for eleven days because
`profile.json` was written 2026-08-02 and never re-derived; a six-experiment
campaign scored 91× worse than one line of code and reported it as progress
(`1789 → 1409 → 1380`, where carrying one column forward scores 15.1).
`tabular.py` records **five** rounds of PR #117 trying to make target inference
right by rule, and names the pattern: *position standing in for evidence*.

Two shapes repeat: **a value fixed regardless of evidence** (modality
confidence, `row_count_estimated`, the metric key), and **a declaration nothing
can reach** (the zarr branch, `TaskStatus "blocked"`, the config field the
warning names) — [M20](../15-gates-must-fail.md)'s shape in a layer M20 did not
sweep.

---

## 3. Requirements

### Functional

1. Answer five questions, each with an evidence-backed confidence:
   **target** (`target_column`, `target_type`, `target_distribution`),
   **split** (`train_table`, `test_table`, `train_test_relationship`),
   **features** (`feature_columns`, `excluded_columns`),
   **identity** (`id_columns`), **objective** (`metric`).
2. Confidence is a **pure function of the evidence that fired**. No call site
   may write a float.
3. A field below the acting threshold produces a **question**, not a value the
   system acts on. Interactive → ask; unattended → block.
4. The core is **deterministic**: same bytes → same schema, byte for byte, with
   no model in the loop.
5. An LLM or a metadata declaration may **propose and corroborate**; neither may
   write a value, and either can be vetoed by the data.
6. Nothing in the inference core may assume a Kaggle-shaped input. A dataset
   with no test set, no submission template and no declared metric must produce
   a usable schema.

### Non-functional

| Property | Target |
|---|---|
| Determinism | Two runs, different working directories → identical JSON |
| Profiling cost | No more than one extra single-column pass per table over today |
| Codegen prompt | Evidence plane adds **0 tokens** at `asserted`; ≤ 1 line per weaker field (the 14,437-token call that ended a campaign is the bound) |
| Consumer churn | 0 required changes in the 5 modules that read the profile |
| Question rate | ≤ 1 question per field per dataset, ever (answers are durable) |

---

## 4. Goals & success metrics

| # | Success criterion | Measured by |
|---|---|---|
| 1 | No confidence is asserted by hand | Every `Inference.confidence == combine(signals)`, checked over all fixtures |
| 2 | The LLM cannot change an answer | Value plane byte-identical with the proposer absent, correct, and adversarial |
| 3 | An ambiguous target is visibly ambiguous | The partitioned fixture without a template ties and asks, naming all candidates |
| 4 | No regression where the profiler is right | titanic / spaceship-titanic / house-prices: identical values, confidence ≥ 0.85 |
| 5 | Ambiguity never auto-resolves | Under `--yes`, a campaign stops with the question pending and runs no experiment |
| 6 | A dataset can be two modalities | `[tabular(primary), image(auxiliary)]` on the mixed fixture |
| 7 | It works with zero Kaggle inputs | No test table, no template, no declared metric → usable schema |

Goals 1–6 are the plan's exit criteria verbatim. Goal 7 is added here: without
it, every other goal can be met by a profiler that only works on Kaggle, and
[M12](../06-beyond-kaggle.md) would have to replace this layer rather than
extend it.

---

## 5. Scope

**In:** the source protocol and one adapter; `DatasetSchema` with measured /
inferred / derived fields; the signal catalogue and `combine`; the five answers;
modality as a list plus `prediction_unit`; the four honesty defects; questions
with ask-or-block; `research schema`; the LLM proposer behind a flag.

**In, easily missed:** the inventory path (`_write_inventory_profile`) also
writes `inferences` — every field 0.0, band `uncertain`, signal
`profiler_failed`. A schema written *because nothing could be inferred* is the
last place the evidence plane may be silent.

**Out, and why:**

| Excluded | Why |
|---|---|
| Adapters beyond `LocalFileSource` | A seam with one implementation is honest; three unused adapters are the "seventh provider adapter" trap |
| The baseline floor (M23) | This milestone makes the target and split it is computed against trustworthy — nothing more |
| Benchmark corpus (M24) | Separate milestone; this supplies the schema it scores |
| `action_space` inference for RL | No fixture, unfalsifiable output. Detect *that* it is an environment, cap at 0.50, ask |
| Audio / environment schema bodies | Presence, a capped confidence and a question only — a schema nothing can falsify is the defect being removed |

---

## 6. Architecture

```
DatasetSource (protocol)          ← the M12 seam: files today, warehouse/env later
   │  tables() columns() sample() exact_unit_count() declared()
   ▼
Inference core (deterministic)
   │  evidence gathering ─→ signals ─→ combine() ─→ Inference{confidence, band}
   ▼
DatasetSchema
   ├── value plane     target_column, id_columns, metric, …   (unchanged names)
   └── evidence plane  inferences[field] → signals, band, alternatives, rejected
   │
   ├─ band uncertain ─→ SchemaQuestion (derived) ─→ ask │ block
   └─ optional: LLM proposer → structural verifier → +0.10 or rejected
```

Two planes, one fact. The **value** lives in `schema.target_column` and nowhere
else; the **evidence** lives in `inferences["target_column"]` and carries no
value. A wrapper would guarantee two names for one fact, which is the defect
class `report.py` and `derived.py` exist to prevent.

### Components

| Component | Responsibility | Depends on |
|---|---|---|
| `profiler/source.py` | `DatasetSource` protocol, `TableRef`, `DeclaredFacts`; `LocalFileSource` | pandas only |
| `profiler/evidence.py` | `Signal`, `Inference`, `SignalSpec`, `CATALOGUE`, `combine`, bands | nothing |
| `profiler/infer/*.py` | One module per question: target, identity, features, split, metric | source + evidence |
| `profiler/tabular.py` | `DatasetSchema`, measured column facts, orchestration | the above |
| `profiler/questions.py` | Derive `SchemaQuestion`s; read/write `schema_answers.json` | schema |
| `cli/schema_cli.py` | `research schema show` / `answer` | questions |
| `profiler/proposer.py` | `SchemaProposalAgent` + structural verifiers (off by default) | micro-agents |

---

## 7. Implementation

### 7.1 The source seam

```python
class TableRef(BaseModel):
    id: str                          # "train.csv", "train/w001__horizontal_well.csv"
    uri: str                         # relative path, sql://…, s3://…, env://connectx

class DatasetSource(Protocol):
    def tables(self) -> list[TableRef]: ...
    def columns(self, table: TableRef) -> list[str]: ...      # header only
    def sample(self, table: TableRef, limit: int | None = None) -> pd.DataFrame: ...
    def exact_unit_count(self, table: TableRef, column: str) -> int: ...
    def declared(self) -> DeclaredFacts: ...
```

`TableRef` carries **no `role` and no `unit_count`** at step 1, though both
appear on the schema later. Calling a table "train" or "the prediction
template" is *inference* — it is half of how `train_test_relationship` is
answered — and it lands in step 3 with the evidence that justifies it; an
honest unit count lands in step 5. A field the profiler cannot yet fill would
be a declaration nothing reaches, which is the defect class being removed.

`DeclaredFacts` is whatever the environment states before anything is inferred —
a Kaggle evaluation metric, a column comment, a `labpilot.yaml` target, the
user's goal string. It is a source of *signals*, never of values: a declaration
the data contradicts is recorded in `rejected`.

A Protocol rather than a base class: a warehouse table, an object store and an
interactive environment share no structure with a directory of CSVs, and the
core needs only `sample`, `columns` and `declared`.

### 7.2 `DatasetSchema` — measured, inferred, derived

> **Which pile does a field belong to?** If two competent people with the same
> bytes could disagree, it is **inferred** and carries evidence. If they could
> not, it is **measured** and carries none. This is what stops a confidence
> field that means nothing, as `ModalityResult.confidence` does today.

```python
PROFILE_SCHEMA_VERSION = 3

class DatasetSchema(BaseModel):          # DatasetProfile, evolved in place
    schema_version: int = 0
    competition: str                                        # dataset id

    tables: list[TableRef] = []                             # measured
    train_table: TableRef | None = None
    test_table: TableRef | None = None
    columns: list[ColumnProfile] = []                       # the one column store

    target_column: str | None = None                        # inferred ↓
    target_type: TargetType = "unknown"
    target_distribution: TargetDistribution | None = None
    id_columns: list[str] = []
    train_test_relationship: SplitRelationship = "unknown"
    metric: MetricRef | None = None
    datetime_columns: list[str] = []
    text_columns: list[str] = []
    excluded_columns: dict[str, ExclusionReason] = {}

    inferences: dict[str, Inference] = {}                   # evidence plane
    notes: list[Note] = []
    modalities: list[ModalityPresence] = []

    @computed_field                                         # derived ↓
    @property
    def feature_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.name not in self.excluded_columns]

    @computed_field
    @property
    def missingness(self) -> dict[str, float]:
        return {c.name: c.null_pct for c in self.columns}

    # …numerical_columns, categorical_columns, cardinality, and the
    # compatibility mirrors: id_column, modality, train_file, row_count

    @computed_field
    @property
    def confidence(self) -> float:                          # weakest link
        return min((self.confidence_in(f) for f in REQUIRED_FIELDS), default=0.0)

REQUIRED_FIELDS = ("target_column", "id_columns", "train_test_relationship", "metric")
DatasetProfile = DatasetSchema           # one class, both names; nothing wrapped
```

`@computed_field` (pydantic v2) serializes, so `profile.json` still carries
`feature_columns`, `missingness` and every legacy field name for readers that
never construct the model — while each fact is stored exactly once. A stored
`categorical_columns` that disagrees with `columns` is the defect this repo
keeps paying for.

The scalar `confidence` is the **weakest link**, not an average: a schema
certain about four things and guessing at the target is a guessing schema.

Portability lives in two enums:

```python
TargetType = Literal["binary", "multiclass", "multilabel", "continuous", "count",
                     "ordinal", "sequence", "structured", "none", "unknown"]

SplitRelationship = Literal["disjoint_units", "temporal_split", "partition_suffix",
                            "same_entities_new_period", "test_unlabeled",
                            "no_test_provided", "environment", "unknown"]
```

`no_test_provided`, `environment` and `target_type: none` are **values**, not the
absence of one. That is what makes a warehouse table and an RL environment
describable by the same schema.

### 7.3 Evidence and `combine`

```python
class Signal(BaseModel):
    id: str                # a catalogue key — the weight lives in the catalogue,
    detail: str = ""       # so a stored schema cannot disagree with its rule

class Inference(BaseModel):
    signals: list[Signal] = []
    confidence: float      # == combine(signals), always
    band: Literal["asserted", "probable", "uncertain"]
    alternatives: list[Alternative] = []
    rejected: list[RejectedClaim] = []

def combine(signals: Sequence[Signal]) -> float:
    specs = [CATALOGUE[s.id] for s in signals]
    naming_mass = min(0.20, 1 - prod(1 - k.weight for k in specs if k.naming))
    raw = 1 - prod(1 - k.weight for k in specs if not k.naming) * (1 - naming_mass)
    caps = [k.cap for k in specs if k.cap is not None]
    return round(min(raw, *caps) if len(caps) == len(specs) else raw, 4)
```

Confidence is **not a probability**. It is coverage over a fixed checklist — how
much of the evidence that would settle this question actually fired. A cap binds
only when *every* signal that fired is capped: a weak rule stops being the
ceiling as soon as real evidence arrives beside it.

Bands: `asserted` ≥ 0.85 · `probable` 0.60–0.85 · `uncertain` < 0.60 → ask.

### 7.4 Weights belong to families, not to datasets

| Family | Ceiling | Rationale | Examples |
|---|---|---|---|
| Stated | 1.00 | A human or authoritative system said so | `operator_answer` 1.00, `declared_by_source` .90 |
| Structural | 0.80 | Would have to be a coincidence | `named_in_prediction_template` .80, `absent_from_scoring_input` .70 |
| Distributional | 0.40 | Consistent, but other columns satisfy it too | `present_across_train_units` .40, `dtype_matches_metric` .30, `non_null_in_train` .20 |
| Naming | 0.20 **total** | Conventions differ by domain and language | `name_matches_metric_family`, `name_is_id_like` |
| Positional | 0.10, **cap 0.50** | Order standing in for evidence | `positional_template_overlap`, `last_withheld_column` |

A signal names **what was observed**, never which file it came from. A dataset
that cannot supply one simply does not fire it, lands lower, and asks.

The naming ceiling is what keeps a target called `y`, `цель` or `outcome_2019`
reachable on structure alone. The positional cap is the plan's third trap made
structural: `overlap[1]` is not improved, it is capped, so a rule that would
pick `id` from a reversed header can never decide alone.

**Uniqueness is not a signal.** "Only one column is withheld" adds no weight —
being the sole candidate is already expressed by an empty `alternatives` list.
Paying for it twice would make a one-candidate dataset look better-evidenced
than a two-candidate one firing identical evidence, and the two-candidate case
is exactly the one that must ask.

Residual conclusions cap at **0.75**: `disjoint_units` ("IID") is what is
concluded when nothing else fired, and an independence assumption that is wrong
turns a CV score into fiction. Actionable, never assertable.

The catalogue itself is data. `target_column`'s entries, which the worked
examples below spend:

| Signal | Family | Weight | Cap |
|---|---|---|---|
| `operator_answer` | stated | 1.00 | — |
| `declared_by_source` | stated | 0.90 | — |
| `named_in_prediction_template` | structural | 0.80 | — |
| `absent_from_scoring_input` | structural | 0.70 | — |
| `present_across_train_units` | distributional | 0.40 | — |
| `dtype_matches_metric` | distributional | 0.30 | — |
| `non_null_in_train` | distributional | 0.20 | — |
| `name_matches_metric_family`, `name_in_goal_text` | naming | 0.20 together | — |
| `is_numeric` | distributional | 0.15 | — |
| `llm_proposal_confirmed` | — | 0.10 | — |
| `positional_template_overlap`, `last_withheld_column` | positional | 0.10 | **0.50** |

The other four questions follow the same families: `id_columns` from
`named_in_prediction_template` (.80), `present_in_train_and_scoring` (.50),
`unique_per_unit` (.40, jointly for a composite key) and `name_is_id_like`;
`train_test_relationship` from `no_test_provided` (.90 — a fact, not a guess),
`environment` (.90), `scored_is_partition_suffix` (.80), `temporal_split` (.75)
and the capped residual above; `metric` from `exact_alias_match` (.90),
`direction_declared` (.30) and `substring_match` (.15, **capped 0.55**).

`exact_alias_match` reads `competition/metric_vocabulary.py`, now on `main` with
one consumer. The substring map it replaces is **still live** in
`competition/metrics.py:38-49` — the path that writes `competition.json` — so a
metric can still reach a workspace by substring, land at 0.55, and ask. That is
the honest reading of `{"name": "balanced_accuracy_score", "key": "accuracy"}`
until the parser path is rewired too.

### 7.5 Features — the safety-critical answer

`feature_columns` is derived as *every column minus `excluded_columns`*, each
exclusion carrying its reason and its own evidence:

| Reason | Test |
|---|---|
| `is_target`, `is_id` | resolved above |
| `unavailable_at_scoring` | the predicate `train_only` already uses |
| `equals_target` | equal to the target wherever both are present — rogii's `TVT_input`, and the classic post-outcome leak |
| `post_outcome` | timestamp later than the label's, or declared |
| `constant` | one distinct value in the sample |
| `operator_excluded` | answered |

This is the one field where a wrong answer is worse than no answer: leakage
makes a score look **better**, so nothing downstream detects it. Exclusions are
therefore conservative, always named in `notes`, and always overturnable by an
answer.

### 7.6 Worked examples — three deliberately unlike shapes

| Case | Signals | Confidence |
|---|---|---|
| **A. house-prices** (all strong signals present) | template .80 + absent-from-scoring .70 + dtype↔RMSE .30 + non-null .20 + numeric .15 | **0.9714** `asserted` |
| **B. 1,546-table partitioned set**, template present | template .80 + across-units .40 + non-null .20 + numeric .15 | **0.9184** `asserted` |
| **B′. same set, no template** | across-units .40 + non-null .20 + numeric .15 | **0.592** `uncertain` |
| **C. warehouse table**: no test set, no template, no declared metric | declared .90 + non-null .20 + numeric .15 | **0.932** `asserted` |

A is the number in the milestone brief, derived from the catalogue rather than
chosen. B is today's behaviour, which is *correct* — and nothing in today's
profile distinguishes being right for that reason from being right by accident.
B′ is the same dataset one file poorer: a second withheld column with the same
dtype and nullity scores **0.592 too**, an exact tie currently broken by a count
and a sort. C is the M12 case: every structural signal is unavailable by
construction, a declaration carries it, and the structural checks that *can* run
validate rather than produce the answer. With nothing declared, C lands
`uncertain` on three fields and asks three questions once.

### 7.7 Questions: derived, asked or blocking

```python
def pending_schema_questions(schema, answers) -> list[SchemaQuestion]:
    return [question_for(schema, f) for f in REQUIRED_FIELDS
            if schema.inferences[f].band == "uncertain" and f not in answers]
```

There is no `schema_questions.json`. A stored question list is derived state
that outlives its cause — repair the schema and a stale file keeps the campaign
blocked, which is what `apply_card_to_beliefs` taught this repo (AGENTS.md
rule 2). One durable file, `schema_answers.json`, holding only what a human
said, keyed by `sha256(dataset | field | sorted candidates)` so a question is
never re-asked and a changed candidate set is genuinely a new question.

It lives beside `profile.json`, **not inside it**: `profile.json` is rebuilt on
every `PROFILE_SCHEMA_VERSION` bump, and an operator's answer must survive a
profiler upgrade. An answer contributes `operator_answer` (1.00) and closes the
question.

`run_until_stop` already takes `approval_prompt` and `offline_fallback_prompt`
as CLI-injected callables (`cli/conduct.py:337,341`), each `None` under `--yes`.
`schema_prompt` threads the same way with one deliberate asymmetry: **there is
no `auto_answer` counterpart to `auto_approve`.** Absent a prompt an approval
falls back to auto-approve; a schema question has nothing to fall back to, so it
blocks. The asymmetry is the design — an option that could answer a schema
question unattended must not exist, because `--yes` would eventually reach it.

```python
# conductor/loop.py, beside the existing `session.status == "paused"` check
pending = pending_schema_questions(load_schema(root), load_answers(root))
if pending and schema_prompt is None:
    ...   # stop=True, rationale=f"stop:schema_question — {pending[0].field}"
```

`StopReason` gains `"schema_question"` — distinct, not folded into
`policy_stop`. `evaluate_stops` is **not** extended: it takes config and state,
and giving it a workspace would make a pure function read the disk.
`prepare_workspace` writes `TaskStatus "blocked"`, declared at `models.py:14`
and written by nothing today.

### 7.8 The proposer

`SchemaProposalAgent(BaseMicroAgent)` reads the dataset description and the
user's goal — **not** the deterministic inferences, so agreement is evidence
rather than echo. Every claim faces a structural verifier (`column_exists`,
`withheld_at_scoring`, `unique_per_unit`, `dtype_matches_metric`):

- **confirmed** → one signal worth 0.10, however many claims agree.
- **contradicted** → `Inference.rejected`, never near the value.
- **nominated_and_verified** → only where the deterministic path produced
  nothing and every verifier passes; capped **0.55**, below the ask threshold,
  so a nomination always ends in a question.

Off by default (`profiler.llm_proposals: false`).

---

## 8. Design choices & tradeoffs

| Choice | Rejected | Chosen | Tradeoff |
|---|---|---|---|
| Evidence shape | Wrap each value in `{value, confidence}` | Flat value plane + `inferences[field]` | Consumers unchanged and one copy of each fact; slightly less obvious that a field has evidence |
| Weak rules | Fix `overlap[1]` with a better heuristic | Weight 0.10 + hard cap 0.50 | Five rounds of PR #117 say the rule never converges; capping means more questions on odd datasets |
| Weight assignment | Per dataset, tuned | Per **family** with ceilings | Portability over per-dataset accuracy — deliberately, since M12 has none of these datasets |
| Questions | Persist a question list | Derive from band + answers | Cannot go stale; recomputed on every read (cheap: a dict lookup per required field) |
| Unattended ambiguity | Auto-answer with the best guess | Block, no `auto_answer` option | A campaign can stop for a human; the alternative freezes a coin flip into every later run |
| Confidence scalar | Mean over fields | Weakest link | One number that cannot hide a guessed target; a single weak field drags the schema score down |
| Source access | Base class with file assumptions | `Protocol`, one adapter | M12 becomes an adapter; one indirection today with no second implementation to validate it |
| Metric mapping | Fix the substring map here | Record *how* it was matched; consume `metric_vocabulary.py` | Keeps this milestone's surface honest; a substring match caps at 0.55 and asks until the parser path is rewired to the vocabulary |
| Modality | Keep the scalar winner | List + computed mirror | Auxiliary modalities stop being discarded; two spellings of one fact, prevented by making one computed |

---

## 9. Observability

An operator asks *"does the system know what it is doing?"* and gets an answer
without reading source:

| Surface | Shows |
|---|---|
| `research schema show` | Per-field value, confidence, band, the signals that fired, and alternatives with their evidence |
| `research schema answer <field> <value>` | Closes a question durably |
| `profile.json` → `confidence` | One weakest-link number a gate can read |
| `notes` (and its `warnings` view) | Structured, machine-readable reasons — `equals_target`, `rows_not_iid` — replacing prose nothing parses |
| Campaign transcript | `stop:schema_question — target_column` instead of a silent guess or a generic `policy_stop` |
| Codegen prompt | One line per field below `asserted`, zero for fields at it |

Alert-worthy: schema `confidence` below 0.60 at campaign start; any `rejected`
claim (declaration or LLM contradicted by the data); `profiler_failed` in the
evidence plane.

---

## 10. Testing

Each check is written to fail before its step lands (AGENTS.md rule 4).

| # | Check | Proves |
|---|---|---|
| 1 | `confidence == combine(signals)` and `band == band_of(confidence)` for every inference in every fixture, including inside `alternatives`; asserted non-empty first | Goal 1 — structurally, since no call site can write a float |
| 2 | Profile every fixture twice in different working directories; compare full JSON | Determinism (requirement 4) — the failure mode a previous test here missed by comparing renders across directories |
| 3 | Build each fixture with no LLM, a correct stub, and a stub wrong on **every** field; compare `model_dump(exclude={"inferences"})` | Goal 2 — makes "propose-only" a mechanism, not a comment |
| 4 | Case C: no test table, no template, no declared metric | Goal 7 — fails today at `FileNotFoundError` |
| 5 | Case B′: exact tie, no asserted value, one question naming every candidate with evidence | Goal 3 |
| 6 | titanic (`conftest.py:118`) + spaceship-titanic + house-prices fixtures | Goal 4 |
| 7 | Tie fixture under `--yes`: campaign stops `stop:schema_question`, zero experiments; with `schema_prompt`: answered, closed, `asserted`. Mutation: delete the block → campaign runs → test fails | Goal 5 |
| 8 | Mixed CSV+image fixture: `[tabular(primary), image(auxiliary)]`, `modality` mirror still `"tabular"` | Goal 6 |

**Real-data validation** uses a sandbox copy, never the live workspace
(AGENTS.md rule 1). Fixtures replicate *shapes*; real datasets check the shapes
were replicated honestly. `scripts/hostile-test.sh` applies — `research schema
show` is Rich output, truncated at 40 columns.

---

## 11. Rollout

| Step | Content | Ships when |
|---|---|---|
| 0 | Fixtures for cases A, B, B′, C, a >`max_rows_sample` table, a CSV-less environment layout | They reproduce the defects |
| 1 | `source.py` + `LocalFileSource`; the profiler reads through it | Schemas byte-identical |
| 2 | `evidence.py`, catalogue, `combine`; `inferences` populated from today's decisions; `notes`/`warnings` view | Check 1 passes; **no values change** |
| 3 | The five answers rewritten as scoring; `id_columns`, `excluded_columns`, `train_test_relationship`, `metric` | Checks 4, 5, 6 |
| 4 | Questions, `schema_answers.json`, `schema_prompt`, block path, `research schema` | Check 7 |
| 5 | Modality list, `prediction_unit`, zarr, tie-break confidence, `row_count`, metric recording; the fiction deleted | Check 8 |
| 6 | Proposer + verifiers, off by default | Check 3 |

**Migration.** `PROFILE_SCHEMA_VERSION` 2 → 3, so every existing profile is
stale and `_ensure_profile` re-derives it. **The catalogue is part of the schema
version**: changing a weight changes what every stored confidence means, so it
bumps the version — without that rule, check 1 degrades into "nobody edited the
catalogue lately". `schema_answers.json` is never touched by a bump.

**Rollback.** Revert the version constant and profiles re-derive on the old
code; answers survive independently. Steps 1–2 are value-neutral and safe to
land alone; step 3 is the first that can change an answer, and step 4 is the
first that can stop a campaign.

**What could go wrong.** Off-Kaggle datasets asking more than expected — bounded
by durable answers and by declarations in the workspace config. A blocked
campaign where a guess would have been right — accepted deliberately, and only
below 0.60. `loop.py` and `budgets.py` also being edited by
[M17](../12-run-until-done.md): ~8 lines in a different function, stated here so
it is a rebase rather than a surprise.
