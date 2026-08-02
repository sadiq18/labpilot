# M15 — The capability layer audit

**Status:** not started · **Purpose:** stop the control plane outrunning the
tools again

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
