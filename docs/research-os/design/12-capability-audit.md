# Design — Capability audit

Back to [../README.md](../README.md) · Plan:
[../autonomy-roadmap/10-capability-audit.md](../autonomy-roadmap/10-capability-audit.md) ·
Sibling (related, not the same mechanism — see §4):
[11-capability-registration.md](11-capability-registration.md) ·
Tools: [02-tools.md](02-tools.md).

**The actual naming collision** (per the plan's Traps section) is one level
up: between `execution/registry.py`'s `CapabilityRegistry` (`TaskType` →
executor) and the Conductor's tool-catalog notion of "capability" — which
covers *both* this doc and 11, not a collision between the two of them.

**Status:** Not started (design)  
**Depends on:** `ToolRegistry` / `ToolDescriptor` (M1, shipped)  
**Standing practice, not a one-off milestone:** every tool added after this
ships with a contract test, enforced in CI — see
[autonomy-roadmap/README.md §Ordering notes](../autonomy-roadmap/README.md).

---

## 1. Background

[The plan](../autonomy-roadmap/10-capability-audit.md) diagnosed the catalog on
2026-08-02: eight tools, one (`run_plan`) actually able to vary its output, and
`implement` rendering a fixed Jinja template whenever the LLM produced nothing —
reported as `completed` regardless.

**That specific finding is now stale.** M19 (merged 2026-08-09, PRs #110–#118)
deleted the Jinja pack in the same commit that made `codegen.strategy=delta`
default (see
[code_engineering/capability.py:10-15](../../../src/labpilot/research_engine/execution/capabilities/code_engineering/capability.py#L10-L15)).
`implement`'s handler now:

- tries a delta proposal (aider) when a parent exists and a gateway is configured,
- falls back to whole-file LLM codegen,
- falls back further to a last-resort scaffold **only for dry runs** — a
  non-dry run with no usable code now **fails the step** instead of reporting
  success (`code_engineering/capability.py:600-615`).

So `implement` moved from *hollow* toward *partial-to-real*, but nobody has
re-run the audit against current code, and the other seven rows (`reflect`,
`query_memory`, `search_papers`, …) are equally unverified against today's
`main`. The 2026-08-02 table also only has 8 rows against today's 10
descriptors — `submit_learn` and the now-independent `run_experiment` were
never audited at all, so the re-audit is a **first baseline** for those two,
not a correction. **The first deliverable of this design is re-running the
audit, not trusting the 2026-08-02 table.**

---

## 2. Problem statement

A named tool implies a capability, and nothing today checks that claim.
Concretely:

1. `ToolDescriptor` has no field for "what does changing my input change" —
   `varies_by` does not exist (`grep -rn varies_by src/` — zero hits).
2. No test asserts *different input → different artifact* for any of the 10
   catalog tools. `test_tool_registry.py` checks names, wiring, and import
   boundaries — never output content.
3. `research tools list` prints name + description only
   ([tools_cli.py:69-78](../../../src/labpilot/cli/tools_cli.py#L69-L78)) — an
   operator cannot see which tools are real without reading source, exactly
   the gap the plan calls out.
4. A verb-named tool that cannot vary (`implement`, `optimise`, `tune`) reads
   as more capable than it is to both the Conductor's policy and the human
   reading `conduct status`.

---

## 3. Requirements

**Functional**

- `ToolDescriptor` gains two required fields: `varies_by: list[str]` (input
  keys the tool claims change its output) and `capability_status:
  Literal["real", "partial", "fixed"]` (no default — see §6.1).
- A contract-test harness, branching on `capability_status` (§6.2): `real` —
  ≥2 fixture inputs differing only in a declared `varies_by` key must produce
  different artifact digests; `fixed` — the tool's name must not read as an
  action verb (`implement`, `optimise`, `tune`, `generate`, …); `partial` —
  the degraded (offline/unauthenticated) path must fail gracefully rather
  than raise or fake a real result.
- `research tools list` prints `capability_status` and `varies_by` per tool
  (extend the existing table; **do not** add a new subcommand — see §9).
- Every entry in `default_tool_descriptors()` re-audited against current
  `main` and its status corrected where §1 already shows drift (`implement`
  at minimum).

**Non-functional**

- Contract tests run under the existing unit marker (`pytest -m "not llm and
  not image and not deep"`) — no live LLM calls; `implement`'s test uses
  `FakeCodegenLLM` extended to vary by input (§6.2), not a live provider.
- Omitting `capability_status` on a new descriptor is a construction-time
  `ValidationError` (§6.1), not a lint rule to remember — this is what stops
  the audit going stale again the way the 2026-08-02 table did. A meta-test
  asserts every registered descriptor's `capability_status` is one of the
  three declared values, as a regression guard on that property.

---

## 4. Scope

**In scope**

- `ToolDescriptor` field additions (`varies_by`, `capability_status`).
- Re-running the audit table against current code; updating
  [10-capability-audit.md](../autonomy-roadmap/10-capability-audit.md)'s table.
- One contract test per catalog tool (10 tools).
- Extending `research tools list` with the two new columns.
- Renaming any tool the re-audit finds is *structurally* fixed (exit
  criterion 3) — decided per-tool by the re-audit, not pre-decided here.

**Out of scope**

- **Closing** any gap the audit finds. `implement` producing better code is
  [M7](../autonomy-roadmap/01-technique-to-model.md)'s job; `reflect` feeding a
  later decision is [M8](../autonomy-roadmap/02-objective-loop.md)'s. This
  design only makes the gap visible and keeps it visible.
- Adding an eleventh tool. The plan is explicit: *"An eleventh named tool
  makes the control plane look richer and changes nothing."*
- The gap-ledger / promote workflow
  ([11-capability-registration.md](11-capability-registration.md)) — that is
  about *missing* tools the Conductor asked for; this is about *existing*
  tools that lie about what they do. Related, not the same mechanism.
- `execution/registry.py`'s `CapabilityRegistry` (`TaskType` → executor) — the
  plan's own **Traps** section calls this out as a different vocabulary for a
  different layer; this design touches `tools/catalog.py` only.

---

## 5. Goals & success metrics

1. Every one of the 10 catalog tools has either a passing contract test
   proving `varies_by` inputs change the output digest, or a name change that
   admits it doesn't.
2. `research tools list` output alone (no source reading) tells an operator
   which tools are real.
3. `git grep varies_by src/labpilot/research_engine/tools/catalog.py` returns
   10 matches, one per descriptor — nothing shipped undeclared.
4. Re-audited table in the plan doc reflects current `main`, with the
   `implement` row corrected per §1.

---

## 6. Design

### 6.1 Descriptor extension

```python
class ToolDescriptor(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_artifacts: list[str] = Field(default_factory=list)
    required_workspace_fields: list[str] = Field(...)
    handler: ToolHandler
    # New:
    varies_by: list[str] = Field(default_factory=list)
    capability_status: Literal["real", "partial", "fixed"]  # required, no default
```

**No default, deliberately.** An earlier draft defaulted this to `"fixed"`
("guilty until proven otherwise") — but Pydantic gives every field its
default whether or not the caller set it, so nothing would ever be
*missing*, and the "CI fails if a descriptor omits `capability_status`"
requirement in §3/§9 would have nothing to catch: an 11th tool added without
setting it would silently inherit `"fixed"` and pass. Making the field
**required** turns that into a construction-time `ValidationError` — a
`ToolDescriptor` without a declared status doesn't build, so it can't reach
`catalog.py`, let alone `main`. The meta-test in §9 becomes a regression
guard confirming this stays true, not the sole enforcement mechanism.

### 6.2 Contract test shape

One parametrized test, not ten hand-written ones — the fixtures differ, the
assertion doesn't:

```python
@pytest.mark.parametrize("name", sorted(t.name for t in default_tool_descriptors()))
def test_tool_contract(name, tmp_path):
    tool = build_default_tool_registry().require(name)
    if tool.capability_status == "fixed":
        # Exit criterion 3: a fixed step's name must not read as an action.
        assert not _reads_as_action_verb(tool.name), (
            f"{tool.name} declares varies_by=[] but is named like a capability"
        )
        return
    if tool.capability_status == "partial":
        _assert_degrades_gracefully(tool, tmp_path)  # see §6.2 discussion
        return
    ws = _fixture_workspace(tmp_path, tool.name)
    inputs_a, inputs_b = _fixture_inputs(tool.name)  # differ only in varies_by keys
    result_a = tool.handler(ws, **inputs_a)
    result_b = tool.handler(ws, **inputs_b)
    assert _digest(result_a) != _digest(result_b), (
        f"{tool.name}: varying {tool.varies_by} produced identical output"
    )
```

`_digest` reuses `file_digest` from
[`execution/capabilities/_helpers.py`](../../../src/labpilot/research_engine/execution/capabilities/_helpers.py)
rather than a new hashing helper — same function M7's differ-table already
validated for `implement`'s output (evidence-log-2026-08-07.md).

`_fixture_inputs` is the part that costs real time: each tool's two fixture
input sets have to be genuine (a real second technique, a real second query),
not `{"x": 1}` vs `{"x": 2}` — a vacuous contract test that always passes is
exactly the failure mode [AGENTS.md](../../../AGENTS.md) calls out ("prove
your test fails without your fix"). Three categories need different fixture
handling than the happy-path `implement` example above:

- **`implement` itself needs `FakeCodegenLLM` to actually vary.** Checked
  against the code: today's `FakeCodegenLLM`
  (`tests/helpers/fake_codegen.py`) ignores its `system`/`user` arguments and
  always returns the same two-file proposal — it currently proves a proposal
  can be *produced*, not that it *varies*. `_fixture_inputs("implement")`
  needs a small extension to the fake (return different `train.py` content
  keyed by the `technique` field in the prompt data) before this test can be
  written; budget it as part of this work, not assumed reuse.
- **Network/auth-dependent tools use their existing `dry_run` escape hatch.**
  `submit_learn` already takes `dry_run` (`tools/handlers/submit.py`) —
  the contract test varies `execution_id` under `dry_run=True` and never
  touches Kaggle. `analyze_competition` needs a fake/offline
  `AnalyzeOrchestrator` the way other analyzer tests already stub it;
  reuse whatever fixture those tests use rather than inventing a new one.
- **`search_papers` is `"partial"`, and neither branch of §6.2 fits it as
  written.** It's real when authenticated, and
  (`tools/handlers/papers.py`) writes an empty hit list under
  `offline=True` or on a 429. Under the no-network CI constraint (§3), the
  contract test for a `"partial"` tool doesn't assert digests differ — it
  asserts the *offline* path degrades gracefully (empty hit list, no
  exception) rather than raising or silently claiming a real result. That is
  a different assertion from the `real`/`fixed` branches, so `capability_status
  == "partial"` gets its own (small) branch in the harness, not a forced fit
  into the other two.

### 6.3 Inventory surface

Extend the existing table in `tools_list()`
([tools_cli.py:69](../../../src/labpilot/cli/tools_cli.py#L69)) with two
columns — no new command:

```python
table.add_column("status")
table.add_column("varies_by")
...
table.add_row(desc.name, desc.description[:80], desc.capability_status,
              ", ".join(desc.varies_by) or "—")
```

### 6.4 If a rename actually happens

Exit criterion 3 floats renaming a structurally-fixed tool. A rename is not
just `catalog.py` — the Conductor hardcodes tool names outside the registry:
`conductor/actions.py` keys its intent→tool templates and a status check on
literal strings (`"implement"` appears in both), and `conductor/policy.py`
keys a dict the same way. [11-capability-registration.md §4.4](11-capability-registration.md)
makes this exact point for the opposite direction (registering a tool doesn't
stop `no_capability` until the intent map is updated too); the same is true
in reverse for a rename. If the re-audit renames a tool, updating those two
files is part of the same PR, not a follow-up.

---

## 7. Components

| Component | Change |
|---|---|
| `tools/descriptors.py` | add `varies_by`, required `capability_status` fields |
| `tools/catalog.py` | populate both fields on all 10 descriptors, per re-audit |
| `tests/helpers/fake_codegen.py` | extend `FakeCodegenLLM` to vary output by the `technique` prompt field (needed for `implement`'s contract test — see §6.2) |
| `tests/unit/test_tool_contracts.py` (new) | parametrized contract test + meta-test for missing fields |
| `cli/tools_cli.py` | `tools_list()` — two new columns |
| `conductor/actions.py`, `conductor/policy.py` | update if the re-audit renames a tool — both hardcode tool names (e.g. `"implement"` in the intent-template list and a policy dict); see §6.4 |
| `autonomy-roadmap/10-capability-audit.md` | re-audited table, replacing the 2026-08-02 snapshot |

---

## 8. Design choices & tradeoffs

| Choice | Option considered | Chosen | Why |
|---|---|---|---|
| Status source | Runtime-inferred (diff two live invocations at startup) | Static field, test-enforced | Simpler; avoids a new inference engine the plan's own trap warns against building |
| Inventory surface | New `research tools audit` subcommand | Extend `tools list` | An 11th surface for a 10-tool catalog is the same "looks richer, changes nothing" trap applied to CLI instead of tools |
| `capability_status` field | Default to `"fixed"` (assume it doesn't work) | **Required, no default** | A default — any value — means the field is never truly missing under Pydantic, so the "CI catches an unaudited new tool" requirement (§3, §9) has nothing to catch. Required makes omission a construction-time error instead of a silent `"fixed"` |
| Fixed-tool renames | Decide names now | Decide per-tool after the re-audit runs | The plan's exit criterion 3 is about *structurally* fixed steps; §1 already shows one row's status changed since the table was written — deciding names before re-auditing would repeat the mistake |

---

## 9. Testing strategy

- Reuse `Workspace.from_competition(...).ensure_roots()`, used in both
  `test_tool_registry.py` and `test_specialists.py`; `scaffold_workspace` is
  `test_tool_registry.py`'s pattern specifically (`submit`-shaped tools that
  need a real competition workspace on disk) — pick per tool, don't assume
  both are interchangeable.
- `implement`'s contract test needs the extended `FakeCodegenLLM` from §6.2
  (varies its response by `technique`) fed two distinct techniques — not a
  live LLM call, and not today's fake unmodified.
- Failure scenario the meta-test exists to catch: a future PR adds tool #11
  without `capability_status` — CI fails on the missing-field check before
  the tool reaches `main`, rather than the catalog silently regaining an
  unverified row.

---

## Open questions

1. Exact verb-list / heuristic for `_reads_as_action_verb` in §6.2 — a fixed
   allowlist (`implement`, `optimise`, `tune`, `generate`, `train`) is
   probably enough; no need for NLP.
2. Where the re-audited table lives long-term — keep it in
   [10-capability-audit.md](../autonomy-roadmap/10-capability-audit.md) (the
   plan) or move it here. Recommend: stays in the plan doc, since that's
   where the original diagnosis and the "not started" status live; this
   design doc stays mechanism-only.
