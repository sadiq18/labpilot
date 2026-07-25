# Plan 4 — Reflection Engine upgrade

Back to [Milestone 2](README.md).

**Status:** Shipped. **Depends on:** Plan 1 (`Experiment`), Plan 2 (`Hypothesis`), Plan 3
(`ExperimentComparison`). **Unlocks:** Plan 5 (optional richer signal).

---

## Goal

This is the **only place in the milestone that uses an LLM** (per the brief's Task 4 and the
"where LLMs belong" principle). Turn `reflection.md` from freeform text into a structured
object that:

1. Answers the four questions from the brief: *why did performance change, what evidence
   supports that, what likely caused it, what should we try next, how confident are we.*
2. Closes the loop on the hypothesis under test (Plan 2), if any.
3. Optionally proposes *new* hypotheses for the ranking backlog (Plan 6).

## Current state

`reflection/generator.py:ReflectionGenerator.generate()` prompts the LLM (via
`llm/client.py:complete_with_fallback`) with profile + baseline + metrics + submission + brief
text, and returns **freeform markdown**. There is no JSON contract, no access to the parent
run, no access to `ExperimentComparison`, and no side effect beyond writing `reflection.md`.
`improvement/planner.py` has its own ad-hoc JSON-parsing helper (`_parse_json_object`, regex +
fence stripping) for its LLM calls — this plan is the second consumer of "ask the LLM for
JSON," so it's the right time to extract that helper into a shared module.

## Design

### 1. `StructuredReflection`

```python
class HypothesisUpdate(BaseModel):
    hypothesis_id: str
    new_status: HypothesisStatus
    note: str

class HypothesisDraft(BaseModel):
    observation: str
    reason: str
    prediction: str
    confidence: float
    tags: list[str] = []

class StructuredReflection(BaseModel):
    run_id: str
    observation: str
    evidence: list[str]
    likely_cause: str
    confidence: float
    suggested_next: list[str]
    hypothesis_updates: list[HypothesisUpdate] = []
    new_hypotheses: list[HypothesisDraft] = []
    generated_by: Literal["llm", "template_fallback"]
```

This mirrors the brief's Task 4 I/O contract almost exactly (`previous`, `current`, `metrics`,
`changes` in → observation/evidence/cause/confidence/next out), with `changes` supplied by
Plan 3's `ExperimentComparison` instead of a raw diff, and `previous`/`current` supplied by
Plan 1's `Experiment`s instead of ad-hoc file reads.

**Extends `Experiment` (Plan 1) with the reflection itself, not just a path.** Plan 1 only
gives `Experiment.reflection_path: str | None` — a pointer to `reflection.md`. That means
nothing downstream (Plan 6's ranking, a script, a test) can read *what the reflection actually
concluded* without a second file read and a manual `json.loads`. This plan adds:

```python
class Experiment(BaseModel):
    ...
    reflection: StructuredReflection | None   # NEW — runs/<id>/reflection.json if present, else None
```

Computed the same way as Plan 1's `config_snapshot`: read at assembly time in `graph.py`, not
stored redundantly — `StructuredReflection.model_validate_json((run_dir / "reflection.json").read_text())`
if the file exists, else `None`. `reflection_path` is kept as-is alongside it (still useful for
linking to the human-readable markdown from the dashboard/CLI); `reflection` is for anything
that wants the structured fields (`observation`, `likely_cause`, `confidence`,
`suggested_next`, ...) directly off the `Experiment` object instead of re-parsing a file by
path. Runs created before this plan ships, or where reflection generation failed, simply get
`reflection=None` — no crash, matching every other optional-field pattern in Plan 1.

### 2. Generation flow

```python
def generate_structured(
    self,
    experiment: Experiment,               # Plan 1
    parent_experiment: Experiment | None,
    comparison: ExperimentComparison | None,   # Plan 3, None for root runs
    hypothesis: Hypothesis | None,             # Plan 2, None if not tagged
    ...existing profile/baseline/metrics/submission args...,
) -> StructuredReflection:
```

Prompt construction extends the existing `reflection_user.j2` template (adds a
`## What changed since the parent run` block rendered from `comparison`, and a
`## Hypothesis under test` block rendered from `hypothesis`, both empty/omitted for root runs
— matches the existing pattern where `brief_text` is conditionally included). System prompt
gains the structured-JSON-output instruction, following the same style already used in
`improvement/planner.py`'s `_PLANNER_SYSTEM` constant.

**Fallback path (no LLM configured) still returns a `StructuredReflection`**, not a different
shape — `generated_by="template_fallback"`, `confidence=0.0`, `suggested_next` populated from
the same static checklist `_fallback_reflection()` uses today. This keeps every downstream
consumer (Plan 5's knowledge base) able to assume one schema always exists, matching the
existing "fails soft, not different" fallback pattern in `brief/` and `reflection/`.

### 3. Shared JSON-parsing helper (small refactor)

Extract `improvement/planner.py:_parse_json_object` into `llm/json_utils.py:parse_json_object`
(strip code fences, find outermost `{...}`, `json.loads`). Both `planner.py` and the new
`reflection/generator.py:generate_structured` import it. Low-risk, removes duplication that
would otherwise be copy-pasted a second time.

### 4. Side effects on success

After generating the structured reflection:

- Write `runs/<run_id>/reflection.json` (new artifact) **and** keep writing
  `runs/<run_id>/reflection.md` (rendered from the structured fields, replacing today's direct
  LLM markdown — the markdown becomes a *view* of the JSON, not a second independent LLM
  output). This preserves the existing `report/generator.py` behavior, which reads
  `reflection.md` as markdown and only needs a template tweak, not a rewrite.
- For each `HypothesisUpdate`: call `HypothesisStore.update_status(...)` (Plan 2), appending
  `run_id` to `evidence_for`/`evidence_against` based on whether the update is confirming or
  rejecting.
- For each `HypothesisDraft`: call `HypothesisStore.create(..., source="reflection")` — these
  land in `status=proposed` and become candidates for Plan 6's ranking, exactly like a
  manually-authored hypothesis would.

This is orchestrated from `orchestrator/pipeline.py`'s `write_reflection` stage, which already
has access to the run dir, parent lookup (if any), and now the Plan 3 `comparison.json` it just
wrote in the prior stage.

### 5. New/changed files

| File | Change |
|---|---|
| `src/labpilot/experiments/models.py` | + `StructuredReflection`, `HypothesisUpdate`, `HypothesisDraft`; `Experiment` (Plan 1) + `reflection: StructuredReflection \| None` |
| `src/labpilot/experiments/graph.py` | `build_graph()` loads `reflection.json` into `Experiment.reflection` when present (mirrors Plan 1's `config_snapshot` loading) |
| `src/labpilot/llm/json_utils.py` | new — extracted `parse_json_object` |
| `src/labpilot/improvement/planner.py` | use `llm/json_utils.py` instead of local `_parse_json_object` |
| `src/labpilot/reflection/generator.py` | + `generate_structured()`; existing `generate()` kept for compatibility or reimplemented as a thin wrapper that renders the structured result to markdown |
| `src/labpilot/reflection/prompts/reflection_user.j2` | + comparison/hypothesis context blocks |
| `src/labpilot/reflection/prompts/reflection_system.md` | + structured JSON output contract |
| `src/labpilot/orchestrator/pipeline.py` | `write_reflection` stage wires comparison + hypothesis in, applies side effects |

## Non-goals

- Reflection does **not** decide to run anything. `new_hypotheses` land as `proposed`; nothing
  auto-executes them (that's explicitly deferred to the Milestone-3 planner).
- No structured-output enforcement via provider-specific JSON mode / function calling in v1 —
  reuse the existing fence-stripping parser for provider portability (matches current
  `llm/client.py` being provider-agnostic over OpenAI/Gemini). Revisit if parsing failures are
  observed in practice.
- `reflection.md`'s human-facing prose quality is not expected to regress, but exact wording
  will change since it's now rendered from structured fields — this is acceptable and worth
  calling out to reviewers, not a silent behavior change to hide.

## Open questions (resolved)

1. Should `hypothesis_updates`/`new_hypotheses` side effects happen even when
   `generated_by == "template_fallback"`? → No; the static fallback never fabricates a
   hypothesis update or draft (empty lists), since it has no real signal to offer beyond the
   existing generic checklist.
2. Cap on `new_hypotheses` per reflection to avoid backlog spam? → Yes, cap at 3,
   configurable (`experiments.reflection.max_new_hypotheses`).
3. Comparison timing vs reflection? → Comparison is best-effort context, never a gate.
   `write_reflection` always produces structured artifacts even if comparison fails; in that
   degraded path leave `new_hypotheses` empty. Hypothesis side effects only when
   `generated_by == "llm"` and (`parent_id is None` or `comparison is not None`).
4. Which hypotheses may the LLM update? → Only the run's tagged `hypothesis_id`. Mismatched
   IDs are ignored. Untagged runs ignore `hypothesis_updates`. Drafts land via
   `HypothesisStore.create(..., source="reflection")`.

## Acceptance criteria

- `generate_structured()` returns a valid `StructuredReflection` for both the LLM-configured
  and no-LLM-configured code paths (existing `test_reflection_generator.py` pattern extended,
  not replaced).
- `runs/<run_id>/reflection.json` and `runs/<run_id>/reflection.md` are both written and the
  markdown is derived from the JSON (no independent LLM call producing markdown directly).
- Running `research improve --run-id <parent> --hypothesis H-001 --strategy tune` against a
  fixture where the LLM (mocked) returns a `confirmed` update for H-001 results in
  `knowledge/<slug>/hypotheses/H-001.json` having `status: confirmed` and the child run id in
  `evidence_for` after the run completes.
- `llm/json_utils.py:parse_json_object` has unit test coverage ported from whatever existing
  `_parse_json_object` tests cover today.
- After a run's `write_reflection` stage completes, `build_graph(...).nodes[run_id].reflection`
  is a populated `StructuredReflection` matching the contents of `reflection.json` on disk —
  not just `reflection_path` pointing at the markdown.
- `Experiment.reflection` is `None` (not a crash) for a fixture run with no `reflection.json`
  (e.g. a run created before this plan shipped, or one where `write_reflection` never ran).
