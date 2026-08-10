# M15 — The capability layer audit

**Status:** not started · **Purpose:** stop the control plane outrunning the
tools again  
**Design:** [../design/12-capability-audit.md](../design/12-capability-audit.md)

> The table immediately below is the **2026-08-02** snapshot, kept for
> history. It is superseded by **[the 2026-08-11 re-audit](#the-re-audit-2026-08-11)**
> further down, which found a live instance of the same failure class in
> `implement` — not the one M19 already fixed, a different one.

---

## Purpose

> They built an excellent orchestration and control plane, but the actual
> capability layer is hollow. The Conductor can decide to try a CNN, but nothing
> can implement one. The tools are named correctly but render a fixed template
> instead of doing real work.

This is the diagnosis in one sentence, and it is a **structural** problem, not a
list of bugs. Every future milestone that adds a decision-making capability will
recreate it unless there is a standing answer to: *"which tools can actually
change the world, and by how much?"*

A named tool implies a capability. `implement()` in a catalog reads as "this
system can implement things". It routes to a specialist, which routes to
`CodeEngineerAgent`, which — when the LLM produces nothing — renders the same
fixed template every time.

## Goal

A maintained, honest inventory of what each tool can actually do, with hollow
tools either filled or renamed.

## The audit (2026-08-02)

| Tool | Reality | Notes |
|---|---|---|
| `analyze_competition` | **real** | Fetches, profiles, ingests, hypothesises. The strongest tool in the catalog. |
| `generate_plan` | **real** | Compiles a genuine task DAG |
| `run_plan` / `run_experiment` | **real** | Trains on 2.98M rows; verified `smoke: false` |
| `submit` | **real** | Packages (never uploads — `submit_learn` does) |
| `reflect` | **real but inert** | Writes assessments, beliefs, lessons; none feed a later decision → [M8](02-objective-loop.md) |
| `implement` | **hollow** | Routes to `CodeEngineerAgent`; on failure renders the fixed template. Ignores `technique`. → [M7](01-technique-to-model.md) |
| `search_papers` | **partial** | Real when authenticated; unauthenticated free tier hits standing 429s |
| `query_memory` | **unverified** | Chosen by the policy in campaign 7; never confirmed to change a decision |

**One real capability drives the score** (`run_plan`), and it has exactly **one
reachable configuration**.

## The re-audit (2026-08-11)

Re-verified against current `main`, tool by tool, code read directly rather
than carried forward from the stale table above.

| Tool | `capability_status` | `varies_by` | Notes |
|---|---|---|---|
| `analyze_competition` | **real** | `only` | Unchanged from 2026-08-02. Different analyzer selection (`build_default_registry()`'s `"competition"`/`"experiments"`/`"dataset"`/`"papers"`/`"repositories"`) genuinely produces different report content. |
| `generate_plan` | **real** | `hypothesis_id` | Unchanged. `planner/templates.py` (deterministic) or `ResearchPlannerAgent` (LLM) select genuinely different task graphs per hypothesis/technique — the M14-promoted deterministic path, not a silent stand-in. |
| `run_plan` | **real** | `plan_id` | Unchanged — confirmed independent of the `implement`-tool finding below: `run_plan`'s `WRITE_CODE` task routes directly to `CodeEngineeringCapability` via `default_capability_registry()` (`execution/engineer.py:546`), never through `ImplementationSpecialist`. |
| `run_experiment` | **real** | `plan_id` | Independent handler from `run_plan` (`tools/handlers/specialists.py`), same verdict, same reasoning — jointly-scored in the 2026-08-02 table, now confirmed separately. |
| `reflect` | **real, but inert** | `execution_id` | Unchanged — produces genuinely different beliefs/evidence per execution; still nothing downstream reads them ([M8](02-objective-loop.md)). |
| `submit_learn` | **real** | `execution_id` | Unchanged. Verified `dry_run=True` still returns real per-execution metrics via `build_execution_outcome`/`load_execution_outcome`, not a canned stub. |
| `query_memory` | **real** (was "unverified") | `query` | First confirmed verdict. `build_research_context` genuinely retrieves different content for different queries when the knowledge DB has data; an empty DB returns an empty context regardless of query, but that's a fixture-vacuity concern for the contract test (§6.2.2 of the design), not evidence the tool is unreal in a populated workspace. |
| `search_papers` | **partial** | — | Unchanged. Real via Semantic Scholar when online; honestly degrades to an empty hit list (`source="offline"` or `source="error:<Type>"`) under `offline=True` or any network failure — the degradation is visible in its own output, not disguised as success. |
| `submit` | **fixed** (was "real") | — | **Verdict changed.** `package_execution_submission` copies `workspace.root/submission.csv` verbatim; `execution_id` only relabels the destination filename (`execution/outcome.py:159-187`). The packaged *content* never depends on the input. This is an honest `fixed` step, not a regression — `submit` doesn't read as a capability verb (`implement`/`optimise`/`tune`) the way exit criterion 3 warns about, so no rename is needed. The 2026-08-02 table's "real" verdict here was never actually checked against the code; it was carried over. |
| `implement` | **partial** (was "hollow", M19 made it look "real") | `description` — **not** `technique` | **The significant findings this re-audit exists to make.** Two separate defects, both below: the M19 fix is real but conditional and the condition fails in the common case; and `technique` never reaches codegen on this path at all. |

### `implement`: a second hollow path, one layer up from the one M19 fixed

M19 (2026-08-09) genuinely fixed the codegen layer: `CodeEngineeringCapability._write`
now tries delta → whole-file LLM → last-resort scaffold, and a non-dry run with
no usable code fails the step instead of faking success
([code_engineering/capability.py:558-615](../../../src/labpilot/research_engine/execution/capabilities/code_engineering/capability.py#L558-L615)).
That fix is real, and `run_plan`/`run_experiment` reach it directly.

The Conductor's standalone `implement` **tool** does not reach it the same
way. Its call chain is `implement()` (tools/handlers/specialists.py) →
`ImplementationSpecialist.execute()` (agents/implementation.py) →
`V1CodeEngineeringCodingTool` → `CodeEngineeringCapability` — and
`ImplementationSpecialist.execute()` has a short-circuit **before** that last
step:

```python
# agents/implementation.py
meta.setdefault("prefer_patch", True)   # set whenever code already exists
prefer_patch = bool(meta.get("prefer_patch")) and not bool(meta.get("force_rewrite"))
if existing and prefer_patch and agent_task.capability in {"implement", "write_code", "write"}:
    # Preserve existing train/src; only ensure separable inference layout.
    refs = ensure_separable_layout(workspace, task_id=agent_task.id)
```

`ensure_separable_layout` (agents/coding.py) does not regenerate
`pipeline/train.py` — if it already exists, it returns an `ArtifactRef`
pointing at the **unmodified file** and only writes `infer.py` if that one is
missing. `CodeEngineeringCapability` — the M19-fixed path — is never called.

Two things make this the same failure class as the one M19 fixed, not a
smaller cousin of it:

1. **It reports success.** `refs` is non-empty (it contains the untouched
   `train.py` ref), so `tools/handlers/specialists.py::implement()`'s own
   guard — `if not refs: raise ImplementProducedNothingError(...)` — never
   fires. The tool returns `ToolResult(refs=[...], data={"paths": [...]})`
   with `train.py`'s path in it, indistinguishable at the `ToolResult` level
   from a call that actually regenerated the file.
2. **It's the default, not an edge case.** `prefer_patch` defaults to `True`
   whenever code already exists, and nothing sets `force_rewrite=True` for
   this path by default: the tool's own signature defaults it to `False`
   (`tools/handlers/specialists.py:42`), and `conductor/actions.py`'s
   `_default_args` for `"implement"` returns only
   `{"description": "update workspace code"}` — no `force_rewrite`
   (confirmed by grep: `force_rewrite` is set exactly once in `src/`, in
   `planner/planner.py:98`, and that write lands on `run_plan`'s *plan*
   metadata, which `CodeEngineeringCapability._write` never reads — grepped,
   zero occurrences — so it reaches neither path). **Every `implement` tool
   call after the first, on a workspace that already has code, silently
   no-ops on the actual training code while reporting success**, regardless
   of what `technique` was requested.

This is why `implement` is `"partial"` rather than `"real"`: it only reaches
the M19-fixed real path on a fresh workspace or when a caller explicitly
passes `force_rewrite=True` — and nothing in the reviewed code does that by
default for this specific tool. Closing this — either making
`ImplementationSpecialist` thread `technique`/force a rewrite when the
requested technique differs from what's on disk, or removing the
`prefer_patch` shortcut for the `implement` capability specifically — is
**out of scope for M15** per its own rule (§4 of the design: this milestone
finds and labels gaps, [M7](01-technique-to-model.md) closes them). Flagging
it here is what the audit is for.

### `implement` does not vary by `technique` at all — it varies by `description`

Found while *building the contract fixture*, not while reading code — which
is the mechanism working as designed: writing the test disproved a
`varies_by` claim this same re-audit had written into `catalog.py` an hour
earlier.

Even with `force_rewrite=True` bypassing the `prefer_patch` shortcut above,
the `technique` kwarg never reaches the codegen prompt. Captured the rendered
prompt from a real `implement` invocation carrying `technique="mixup"`:

```text
Technique: —
Goal:
apply mixup Prefer separable layout: pipeline/train.py for training …
```

The cause is one object written and a different one read:

* `implement()` puts its `**extra` (including `technique`) into
  `AgentTask.metadata`;
* `build_v1_task_context` (agents/coding.py) copies that onto the synthetic
  **`ResearchTask`** — `ResearchTask(..., metadata=dict(agent_task.metadata))`
  — and constructs the enclosing `ResearchPlan` with **no metadata at all**;
* `CodeEngineeringCapability._write` reads `plan_meta = dict(context.plan.metadata or {})`
  — the **plan**, never the task — so `resolve_technique` sees an empty dict
  and returns `status="none"`, and the prompt renders `Technique: —`.

This is [AGENTS.md](../../../AGENTS.md)'s rule 3 again — *"the guard exists
and its input is wrong"* — for the tenth time in this codebase, and the first
one caught by M15's own mechanism rather than by a failed campaign.

`catalog.py` now declares `varies_by=["description"]` for `implement`, which
is what the tool can actually vary by: `description` reaches codegen as
`goal`/`task_description` and genuinely conditions the output (verified —
two descriptions produce two different `train.py` files). Declaring
`technique` would have been precisely the unverified capability claim this
milestone exists to catch, shipped by the milestone itself.

Both findings are pinned as tests in
`tests/unit/test_tool_contract_fixtures.py` — including
`test_implement_without_force_rewrite_is_a_silent_noop`, which documents the
`prefer_patch` no-op rather than asserting it is correct, so it fails loudly
if the behaviour changes in either direction.

## Approach

**1. Every tool declares what it can vary.** A tool that cannot produce a
different outcome given different inputs is not a capability, it is a fixed step.
Make this declarable and assertable:

```python
ToolDescriptor(
    name="implement",
    varies_by=["technique", "hypothesis_id"],   # must change output
    output_artifacts=["code"],
)
```

**2. A contract test per tool: different input → different artifact.** For
`implement`, two techniques must produce two different `train.py` digests. This
is the generalised form of [M7](01-technique-to-model.md)'s exit criterion, and
it is the check that would have caught the hollow layer on day one.

**3. Rename what stays fixed.** If a step genuinely cannot vary, its name must
say so (`render_baseline_template`, not `implement`). Honest names prevent the
Conductor — and the reader — from assuming a capability that is not there.

**4. Surface the inventory.** `research tools` listing each tool as
real / partial / fixed, with its `varies_by`. An operator should be able to see
the hollow layer without reading source.

## Exit criteria

1. Every catalog tool has a contract test proving different inputs yield
   different artifacts, or is renamed to admit it is fixed.
2. `research tools` prints the inventory with capability status.
3. No tool named as an action (`implement`, `optimise`, `tune`) is a fixed step.

## Traps

- **The naming collision is real and worth fixing.** "Capability" means two
  different things: the execution `Capability` classes (`workspace`, `training`,
  `submission` — 10 of them, covering all 18 task types) and the Conductor's
  `record_suggestion("Need capability/tool X")`, which is about the *tool*
  catalog. Two vocabularies for two layers guarantees confusion in design
  discussions.
- **Do not add tools to close the gap.** The catalog has 10 tools and one of
  them can vary an outcome. An eleventh named tool makes the control plane look
  richer and changes nothing.
- **`record_suggestion` is already the right instinct.** When an intent maps to
  no tool, the Conductor records "Need capability/tool X". That is the system
  telling you what to build — it is currently written and never read. Surfacing
  those suggestions in `conduct status` is nearly free and turns the loop into a
  roadmap generator.

## Related code

- `src/labpilot/research_engine/tools/catalog.py` — the 10 tool descriptors
- `src/labpilot/research_engine/tools/handlers/specialists.py` — `implement`, `run_experiment`
- `src/labpilot/research_engine/execution/registry.py` — the *other* capability concept
- `src/labpilot/research_engine/conductor/metrics.py` — `record_suggestion`
