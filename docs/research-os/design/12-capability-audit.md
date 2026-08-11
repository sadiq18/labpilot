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
`main`. The 2026-08-02 table also has only 8 rows against today's 10
descriptors: `run_experiment` **is** covered there — jointly with `run_plan`,
verdict `real` — but the two now have independent handlers (`run_plan` in
`tools/handlers/run.py`, `run_experiment` in `tools/handlers/specialists.py`),
so that joint verdict needs re-splitting, not just re-confirming;
`submit_learn` has zero prior mention and is a
genuine **first baseline**. **The first deliverable of this design is
re-running the audit, not trusting the 2026-08-02 table.**

**M14 already ran this exact discipline one layer down.** M14 Phase 3
("delete or demote", complete 2026-08-07, PR #104) retired all 20
micro-agent `_run_rule_engine` fallbacks — LLM stand-ins were deleted
outright; genuine deterministic logic was **promoted to a named, explicit
step instead of disguised as a failed LLM call**
([09-llm-required.md](../autonomy-roadmap/09-llm-required.md)). Verified
against current code: `_render_template_fallback` no longer exists anywhere
in the tree (`grep -rn _render_template_fallback src/` — zero hits, beyond
the M19 Jinja deletion §1 already covers), and `planner/templates.py` — the
deterministic path behind `generate_plan` — is exactly the "promoted"
pattern M14's Phase 3 describes: an explicit, named, opt-in fallback under
`llm_client=None`, not a silent stand-in. That is **why** `generate_plan`
was already `real` in the 2026-08-02 table and stays that way here.

This narrows what M15 actually needs to do. M14 covers the **micro-agent**
layer (`intelligence/micro_agents/`, `reflection/`, `planner/planning_engine`,
`execution/code_engineer`) and is out of scope here (§4) — already audited,
already renamed where it needed to be. M15 is the same audit one layer up,
at the **Conductor tool catalog** (`tools/catalog.py`'s 10 descriptors), which
M14 never touched. Combined with M19's `implement` fix, the *known* hollow
count in the catalog going into the re-audit is plausibly zero — but that is
the re-audit's finding to make, not an assumption this design bakes in.

### 1.1 Two questions this design does *not* answer

Worth stating explicitly, because both are natural to ask and neither is what
the contract test measures:

**"What tools are we missing?"** Not this milestone. Discovering *absent*
capability is [11-capability-registration.md](11-capability-registration.md)'s
job — the Conductor already writes `record_suggestion("Need capability/tool
X")` whenever it wants something the catalog doesn't have, surfaced today via
`research tools gaps`. The plan's own Traps section is explicit that adding a
tool to close a gap is the wrong move for *this* design: *"An eleventh named
tool makes the control plane look richer and changes nothing."* M15 only
audits the 10 tools that already exist.

**"How well is an existing tool performing?"** Also not this milestone. The
contract test proves one narrow thing — *does a declared `varies_by` input
actually change the output* — which is a floor, not a quality score. It
cannot tell you `implement` writes *good* code, only that it writes
*different* code for a different technique. Whether a tool's output is any
good is split across other milestones already: [M9](../autonomy-roadmap/03-verification-first.md)
(can the result be trusted), [M20](../autonomy-roadmap/15-gates-must-fail.md)
(do the pass/fail gates actually gate), [M7](../autonomy-roadmap/01-technique-to-model.md)/[M8](../autonomy-roadmap/02-objective-loop.md)
(does the technique/reflection move the score). `reflect` is the clearest
example already in the catalog: it's `real` — it produces different beliefs
for different inputs — and simultaneously `inert`, because nothing downstream
reads them. `varies_by` cannot see that; M8 owns it.

### 1.2 Does moving to LLM-only tooling remove the need for this?

No — and the premise needs a correction first. Deterministic paths were not
removed catalog-wide; M14 Phase 3's own rule was to *keep and promote*
genuine deterministic logic rather than delete it. `planner/templates.py`
(795 lines) is still `generate_plan`'s live, explicitly-named deterministic
path today. `execution/baseline/selector.py` is deterministic by design —
problem-type/metric selection from a data profile, no LLM call, and correctly
so. `submit`/`submit_learn` were never LLM tools at all; they package and
upload a CSV. What M14 actually removed was the *silent, automatic* rule-engine
fallback — not deterministic code itself.

Even in a hypothetically all-LLM catalog, the risk this design targets does
not shrink — it moves. M14's failure mode was *the wrong path ran silently*
(a rule engine stood in for a broken LLM call and nobody could tell). The
failure mode M15 targets is different and, if anything, more likely once
every path is an LLM call: *the right path ran, and still produced output
that does not actually depend on the input.* An LLM can fail silently in ways
a rule engine can't — truncation, a rate limit, a model that ignores the
`technique` field in its own prompt and returns boilerplate anyway — and two
LLM outputs can differ syntactically (variable names, comments) without ever
implementing the different technique that was asked for. That is precisely
what `implement` did pre-M19, and it is what the contract test's digest
check — driven by a genuinely varying fixture, not `{"x": 1}` vs `{"x": 2}`
— exists to catch regardless of which layer (rule engine or LLM) produced the
non-varying output.

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

- `ToolDescriptor` gains two fields: `capability_status: Literal["real",
  "partial", "fixed"]` (required, no default — see §6.1) and
  `varies_by: list[str]` (input keys the tool claims change its output;
  defaults to `[]`, which is the *correct* value for a `fixed` tool — not a
  missing-value placeholder). A validator enforces `capability_status ==
  "real"` implies non-empty `varies_by`, so the thing a default would hide
  (a real tool nobody bothered to declare `varies_by` for) is still caught,
  without forcing every `fixed` tool to write out `varies_by=[]` by hand.
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
- Every non-catalog `ToolDescriptor(...)` construction, updated to set the
  new required field. Making `capability_status` required (§6.1) is a
  breaking change everywhere the type is constructed, not just in
  `catalog.py` — found 16 more call sites across 8 files: `cli/conduct.py`
  (3), `tests/helpers/campaign_harness.py` (1), and test doubles in
  `test_conductor.py`, `test_context_m4_capstone.py`,
  `test_tool_registry.py` (the `"echo"` fixture tool), `test_gap_ledger.py`,
  `test_campaigns.py`, `test_conductor_single_tool_mode.py`. Test-double
  descriptors get `capability_status="fixed"` (they're not catalog tools;
  their status is irrelevant to the audit) — this is a one-line addition per
  site, not new design, but it must land in the same PR as §6.1 or those 8
  files fail to construct.

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
- The micro-agent rule-engine layer (`intelligence/micro_agents/`,
  `reflection/`, `planner/planning_engine`, `execution/code_engineer`) — M14
  Phase 3 already ran this design's discipline there (delete pure LLM
  stand-ins, promote genuine deterministic logic to a named step) and shipped
  it 2026-08-07 (PR #104, §1). Re-auditing already-audited, already-renamed
  code would duplicate M14, not extend it.

---

## 5. Goals & success metrics

1. Every one of the 10 catalog tools has either a passing contract test
   proving `varies_by` inputs change the output digest, or a name change that
   admits it doesn't.
2. `research tools list` output alone (no source reading) tells an operator
   which tools are real.
3. `git grep varies_by src/labpilot/research_engine/tools/catalog.py` returns
   10 matches — every descriptor states `varies_by` explicitly (even `[]` for
   `fixed` tools) as a documentation convention. The actual enforcement for
   `real` tools is the §6.1 validator, not this grep — an explicit `[]` is
   readable-by-convention, not machine-required.
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

    @model_validator(mode="after")
    def _real_tools_declare_variance(self) -> "ToolDescriptor":
        if self.capability_status == "real" and not self.varies_by:
            raise ValueError(
                f"{self.name}: capability_status='real' but varies_by=[] — "
                "declare what input changes the output, or downgrade the status"
            )
        return self
```

**`capability_status` has no default, deliberately.** An earlier draft
defaulted it to `"fixed"` ("guilty until proven otherwise") — but Pydantic
gives every field its default whether or not the caller set it, so nothing
would ever be *missing*, and the "CI fails if a descriptor omits
`capability_status`" requirement in §3/§9 would have nothing to catch: an
11th tool added without setting it would silently inherit `"fixed"` and pass.
Making the field **required** turns that into a construction-time
`ValidationError` — a `ToolDescriptor` without a declared status doesn't
build, so it can't reach `catalog.py`, let alone `main`. The meta-test in §9
becomes a regression guard confirming this stays true, not the sole
enforcement mechanism.

**`varies_by` keeps its `[]` default**, unlike `capability_status` — `[]` is
the *correct* value for a `fixed` tool, not a stand-in for "not filled in
yet," so making it unconditionally required would force every genuinely
fixed tool to write `varies_by=[]` for no benefit. The gap that mattered —
a `"real"` tool that forgot to declare what it varies by — is closed by the
validator above instead: `capability_status="real"` with an empty
`varies_by` fails construction, the same way a missing `capability_status`
does.

### 6.2 Contract test shape

> **Shipped 2026-08-11** as `tests/unit/test_tool_contracts.py`, with
> fixtures in `tests/unit/tool_contract_fixtures.py`. Three corrections the
> implementation forced on the sketch below, each recorded where it applies:
> the branch keys off `varies_by` rather than `capability_status` (`implement`
> is `partial` *and* varies); `_digest` became a per-tool `observe()` because
> one payload digest cannot serve ten artifact shapes (§6.2.1); and each call
> must be **observed before the next one runs**, since tools whose artifact is
> a file at a fixed path otherwise get read twice after the second write — a
> false *negative* that failed on first run.
>
> **A fourth, found in review rather than by the tests:** an `observe()` must
> never read back a field the handler *echoes from its own input*.
> `run_experiment`'s first version returned `data["plan_id"]`, which the
> handler copies straight from its argument — so the contract compared the
> two fixture inputs to each other and passed regardless of what the tool
> did. Demonstrated with two plans carrying identical task graphs under
> different ids: the old observation differed (wrongly green), the
> evidence-set one collapsed (correctly red). When choosing what to observe,
> prefer state the tool *wrote* over anything it returns.

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
        ws = _fixture_workspace(tmp_path, tool.name)
        degraded_kwargs = _degraded_inputs(tool.name)  # e.g. {"offline": True}
        result = tool.handler(ws, **degraded_kwargs)
        _assert_degraded(tool.name, result)  # per-tool: no exception raised,
        # AND the payload honestly says "degraded" rather than looking real —
        # see the note below on why this can't be one generic assertion
        return
    ws = _fixture_workspace(tmp_path, tool.name)
    inputs_a, inputs_b = _fixture_inputs(tool.name)  # differ only in varies_by keys
    result_a = tool.handler(ws, **inputs_a)
    result_b = tool.handler(ws, **inputs_b)
    assert _digest(result_a) != _digest(result_b), (
        f"{tool.name}: varying {tool.varies_by} produced identical output"
    )
```

For `implement` — a file artifact — `_digest` reuses `file_digest` from
[`execution/capabilities/_helpers.py`](../../../src/labpilot/research_engine/execution/capabilities/_helpers.py)
directly, the same function M7's differ-table already validated
(evidence-log-2026-08-07.md). Tools whose artifact is a metadata record
rather than a file need a normalized variant instead — see §6.2.1.

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
- **`search_papers` is `"partial"`, and neither of the other two branches of
  §6.2 fits it.** It's real when authenticated, and `tools/handlers/papers.py`
  writes an empty hit list under `offline=True`, and on any exception from
  the network call (not specifically a 429 — the handler's `except Exception`
  is broader than the plan doc's original wording suggested). Under the
  no-network CI constraint (§3), the contract test for a `"partial"` tool
  doesn't assert digests differ — it asserts the degraded path returns
  cleanly rather than raising or silently claiming a real result. That needs
  a third fixture function alongside `_fixture_inputs`:
  `_degraded_inputs(name) -> dict` returns the kwargs that force the
  degraded path per tool — `{"offline": True}` for `search_papers` today,
  and the same lookup a future second `"partial"` tool would extend rather
  than special-case. `capability_status == "partial"` gets its own branch in
  the harness (shown above) built from the same `_fixture_workspace` helper
  as the `real` branch, not a new workspace-construction path.

  **`_assert_degraded` cannot be one generic assertion.** An earlier draft
  checked `result.data.get("error")` — but `search_papers`'s
  `ToolResult.data` never has an `"error"` key; its shape is `{"query",
  "source", "count", "papers"}`, and the offline path sets
  `source="offline", count=0, papers=[]` with no exception at all. A generic
  `error`-key check would be **vacuously true** on that exact payload — it
  would pass even if `count`/`papers` silently carried fabricated results,
  which is the specific failure this branch exists to catch (§3). So
  `_assert_degraded` is per-tool, keyed the same way `_degraded_inputs` and
  `_fixture_inputs` are: for `search_papers`, assert `data["source"] ==
  "offline"` and `data["papers"] == []`; a second `"partial"` tool declares
  its own honest-degradation check rather than reusing this one by accident.

### 6.2.1 A second failure direction: the digest can lie the other way

Everything above guards against a **false pass** — a tool that doesn't vary
looking like it does (`FakeCodegenLLM`'s flat response). There is a mirror
risk the harness must also guard against: a **false real-verdict** from
hashing the wrong thing. Several tools' primary artifact is a JSON record
carrying an ID assigned fresh on every call — `generate_plan`'s `plan.id`
(`P-001`, `P-002`, …), `run_plan`/`run_experiment`'s `execution.id`, and
`reflect`'s `evidence_id` / `belief_update_id`. If `_digest` hashes the raw artifact
including that ID, **two calls with the identical `varies_by` input** still
hash differently, because the ID incremented — the contract test would pass
for a tool that ignores its input entirely, which is a worse failure than a
vacuous fixture: it reports a hollow tool as audited-real.

`file_digest` on `train.py` (`implement`'s case) doesn't have this problem —
the file *is* the content, no wrapper ID. Tools whose artifact is a metadata
record need a **normalized digest**: strip the auto-generated id/timestamp
fields before hashing, so what's compared is the content that should
actually trace back to the varying input (task types/descriptions for a
plan, metric values and written-file paths for an execution, evidence/claim
content for a reflection). This is per-artifact-shape, not generic — budget
it alongside `_fixture_inputs`, not as a detail inside `_digest` itself.

### 6.2.2 Per-tool fixture scope

Concrete enough to start from, not a full spec — each row is what the
contract test needs to *not* be vacuous in either direction (§6.2, §6.2.1):

**Confirmed by the 2026-08-11 re-audit** ([autonomy-roadmap/10-capability-audit.md](../autonomy-roadmap/10-capability-audit.md#the-re-audit-2026-08-11)) — the "re-audit to confirm" placeholders below are resolved.

| Tool | `capability_status` (confirmed) | `varies_by` (proposed) | Fixture strategy | Known risk |
|---|---|---|---|---|
| `analyze_competition` | real | `only` | Offline (`llm_client=None`, no `kaggle_config`) — reuse the analyzer-stub fixture already in `test_research_intelligence.py` / `test_competition_analyzer.py`; vary `only="competition"` vs `only="dataset"` (`build_default_registry()`'s actual registered names — `"competition"`, `"experiments"`, `"dataset"`, `"papers"`, `"repositories"`; an earlier draft of this row cited `"overview"`/`"data"`, which don't exist and would raise `UnknownAnalyzerError`) | **Normalized digest required, not just likely**: `AnalysisReport.generated_at` (`intelligence/models.py:129`) defaults to `datetime.now(UTC).isoformat()` — it always differs, every call, unconditionally |
| `search_papers` | partial | — | `_degraded_inputs`/`_assert_degraded`, §6.2 | Already scoped |
| `generate_plan` | real | `hypothesis_id` (or `baseline`) | Seed two `Hypothesis` rows with genuinely different `technique`/`observation` fields via `HypothesisStore`; compile a plan from each | **Normalized digest required** (§6.2.1) — `plan.id` differs on every call regardless of input |
| `implement` | **partial** — confirmed, not "real": the M19-fixed codegen path exists but `ImplementationSpecialist`'s `prefer_patch` shortcut skips it by default whenever the workspace already has code, silently no-opping on `train.py` while still reporting success (autonomy-roadmap §"implement: a second hollow path") | `technique` — **only reachable with `force_rewrite=True` or a fresh workspace**; the fixture must set `force_rewrite=True` explicitly or the contract test will pass against the `prefer_patch` no-op path, not the real one | Extended `FakeCodegenLLM` (§6.2) **plus `force_rewrite=True` in the fixture inputs** — updated per the re-audit finding | Without `force_rewrite=True`, this contract test would be vacuous in the false-pass direction on any fixture workspace that already has `pipeline/train.py` |
| `run_plan` | real | `plan_id` | Two plan fixtures with **different task graphs** (not just different ids), `dry_run=False`, wired to one of `tests/conftest.py`'s synthetic dataset fixtures (`titanic_data_dir`, `generic_regression_data_dir`, `multiclass_data_dir`) | **This is new work, not reuse — corrected from an earlier draft.** `test_engineer_capabilities.py` never actually uses those `conftest.py` fixtures (checked: zero references); every test there runs `dry_run=True`/`train_stub`/stub scaffolding, no real CSV touched. `dry_run=True` is very likely **not enough** to prove variance on its own: a stub/smoke run tends to short-circuit to the same wiring-only artifact regardless of plan content, the same failure shape M19 removed for `implement`. This tool needs the slower real-but-tiny path — the datasets exist, the wiring to a real `run_plan` call under `dry_run=False` does not, and should be budgeted as such. |
| `run_experiment` | **real** — confirmed independently of `run_plan` (was jointly-scored in the 2026-08-02 table); routes to `CodeEngineeringCapability` directly, not through `ImplementationSpecialist`, so the `implement`-row finding does not apply here | `plan_id` | Same as `run_plan` — same dataset fixture, independent handler | Same dry-run risk as `run_plan` |
| `reflect` | real but inert | `execution_id` | Two prior executions with different metrics/evidence already on disk (via `ExecutionArtifacts` fixtures) | Normalized digest (§6.2.1) — but be precise about *which* id: `evidence_id` (`ReflectionStore.new_evidence_id()`) and `belief_update_id` are genuinely fresh every call and must be stripped; `belief_id` itself is **not** — it's a stable upsert key, `f"belief:{competition}:{slug(technique_name)}"` (`reflection/beliefs/updater.py:41`), and stays the same across calls for the same technique. Stripping the wrong field leaves the digest still contaminated by the fields that actually change. |
| `submit` | **fixed** — confirmed, verdict changed from the 2026-08-02 table's unchecked `real` | — | Contract test uses the `fixed` branch of §6.2 (name-doesn't-read-as-verb check) | `package_execution_submission` copies `workspace.root/submission.csv` verbatim and only relabels it by `execution_id` — content never depends on input. Legitimate `fixed` step; `submit` doesn't read as a capability verb, so no rename needed. |
| `submit_learn` | real | `execution_id`, `dry_run=True` | Two prior executions with different stored outcomes/metrics (`load_execution_outcome`) | **Verified, not vacuous**: read `execution/submit_learn.py:440-465` — under `dry_run=True` the handler still calls `build_execution_outcome`/`load_execution_outcome` and returns real per-execution metrics/plan tags, not a canned dry-run stub. The `dry_run` escape hatch genuinely proves variance without touching Kaggle. **Also needs the §6.2.1 normalized digest** — an earlier draft of this row omitted it: `ExecutionOutcomeSummary` and `SubmissionRecord` (`execution/outcome.py`, `artifacts/submission.py`) both carry fresh `created_at`/`updated_at` timestamp fields, the same pattern flagged for every other `real` row in this table. Raw-digesting this artifact has the identical false-real-verdict risk §6.2.1 exists to catch. |
| `query_memory` | **real** — first confirmed verdict, was "unverified" | `query` | **Must seed the knowledge DB with ≥2 distinct evidence/claim rows first** | An empty knowledge DB returns an empty context regardless of query — the same vacuous-fixture risk as `FakeCodegenLLM`, just with data instead of a fake LLM |

The two hardest rows are `run_plan`/`run_experiment` (dry-run may not prove
anything, and the honest fixture is slower) and `submit` (the re-audit may
find `real` was never the right verdict). Both are flagged rather than
resolved here — resolving them is the re-audit's job (§1), and this table
exists so the re-audit isn't rediscovering the same fixture problems this
design already hit while scoping `implement`.

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
just `catalog.py` — the Conductor hardcodes tool names outside the registry,
in at least three places, all keyed on the literal string `"implement"`:

- `conductor/actions.py`'s intent-keyword tuple (an *intent-matching* string,
  not a tool reference — a rename must update it too, or the keyword still
  fires and maps to a tool name that no longer exists);
- `conductor/actions.py`'s `_default_args`, which selects default arguments
  by tool name (a default-argument selector, not a status/precondition
  check — corrected from an earlier draft of this doc, which mislabeled it);
- `conductor/policy.py`'s `requires` dict in `available_tools`, which is the
  actual gating/precondition check (`"implement": has_runnable`).

[11-capability-registration.md §4.4](11-capability-registration.md) makes
this exact point for the opposite direction (registering a tool doesn't stop
`no_capability` until the intent map is updated too); the same is true in
reverse for a rename. If the re-audit renames a tool, updating all three
sites is part of the same PR, not a follow-up.

---

## 7. Components

| Component | Change |
|---|---|
| `tools/descriptors.py` | add `varies_by`, required `capability_status` fields |
| `tools/catalog.py` | populate both fields on all 10 descriptors, per re-audit |
| `tests/helpers/fake_codegen.py` | extend `FakeCodegenLLM` to vary output by the `technique` prompt field (needed for `implement`'s contract test — see §6.2) |
| `tests/unit/test_tool_contracts.py` (new) | parametrized contract test + meta-test for missing fields |
| `tests/unit/tool_contract_fixtures.py` (new) | `_fixture_workspace`, `_fixture_inputs`, `_degraded_inputs`, `_digest`/normalized-digest, one entry per tool — the per-tool detail §6.2.2 scopes but doesn't write |
| `cli/tools_cli.py` | `tools_list()` — two new columns |
| `conductor/actions.py`, `conductor/policy.py` | update if the re-audit renames a tool — three hardcoded-string sites, see §6.4 |
| `cli/conduct.py`, `tests/helpers/campaign_harness.py`, and 6 test files (§4) | add `capability_status="fixed"` to every non-catalog `ToolDescriptor(...)` construction — required by §6.1, else these fail to construct |
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
- Two failure scenarios the meta-test / validator combination exists to
  catch: (1) a future PR adds tool #11 without `capability_status` — CI fails
  on construction before it reaches `main`; (2) a tool is marked `"real"`
  with `varies_by=[]` — the §6.1 validator fails construction the same way,
  instead of shipping an unaudited "real" claim the way `varies_by`'s own
  default would otherwise let through silently.

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
