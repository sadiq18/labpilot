# Backlog — the `implement` tool's technique path

**Status:** Backlog — orphaned by M7's close.
[M15](../autonomy-roadmap/10-capability-audit.md)'s re-audit routed these two
defects to [M7](../autonomy-roadmap/01-technique-to-model.md), but M7 shipped
2026-08-07 scoped to the **plan-execution** path and never covered the
standalone `implement` tool. Nothing owns them now.

## Why M7 being done does not close these

M7's exit criteria are campaign-level and plan-path only: *three hypotheses
with different techniques produce three different `cv_*` values*, *each
experiment record names its technique*, *an inapplicable recipe is rejected at
plan time*. Its differ-table validated
`run_plan`/`run_experiment` → `CodeEngineeringCapability`, which M15 re-confirmed
as `real`. It never mentions `ImplementationSpecialist`, `prefer_patch`, or the
`implement` tool — `grep` finds none of them in the plan doc.

So M7 is honestly done for what it scoped. These are a different path.

## The two defects

Both were found by building M15's contract fixtures, and both are pinned by
tests that fail if the behaviour changes in **either** direction.

**1. `prefer_patch` is a silent no-op.**
`ImplementationSpecialist.execute()` short-circuits before
`CodeEngineeringCapability` whenever the workspace already has code:
`meta.setdefault("prefer_patch", True)`, and `ensure_separable_layout` then
returns an `ArtifactRef` pointing at the *unmodified* `pipeline/train.py`.
Because `refs` is non-empty, `implement()`'s own
`ImplementProducedNothingError` guard never fires and the tool reports
success. Only a fresh workspace or an explicit `force_rewrite=True` reaches
the real path, and nothing sets that by default for this tool.
Pinned: `test_implement_without_force_rewrite_is_a_silent_noop`.

**2. `technique` never reaches the codegen prompt on this path.**
`build_v1_task_context` puts the agent task's metadata on the synthetic
`ResearchTask`, while `CodeEngineeringCapability._write` reads
`plan.metadata` — written to one object, read from another. The prompt
therefore always renders `Technique: —` however the caller sets it, and only
`description` (via `goal`) conditions the output. This is why
`catalog.py` declares `varies_by=["description"]` rather than `["technique"]`,
and why `implement` is `capability_status="partial"`.

## Impact

The Conductor can select `implement`, be told it succeeded, and have nothing
change — the failure shape M15 exists to surface. It is labelled honestly
today (`partial`), so the control plane no longer *overstates* it; closing
these would let it be promoted to `real`.

## Done when

- A second `implement` call with a different `description` changes
  `pipeline/train.py` without needing `force_rewrite=True`, **or** the
  short-circuit is removed for the `implement` capability specifically.
- `technique` reaches the codegen prompt on this path, or the tool stops
  accepting a parameter it cannot honour.
- `catalog.py` can declare `implement` `real`, and
  `test_declared_statuses_match_the_re_audit` is updated deliberately.
