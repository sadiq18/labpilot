# Design — Capability registration

Back to [../README.md](../README.md) · Backlog:
[../backlog/capability-registration.md](../backlog/capability-registration.md) ·
Tools: [02-tools.md](02-tools.md) · Campaigns: [06-campaigns.md](06-campaigns.md).

**Status:** Design  
**Depends on:** M3 campaign loop + `ToolRegistry` (shipped)  
**Does not replace:** Research Engineer `BaseCapability` map (plan-task handlers) —
Conductor gaps are **OS tools**, not Engineer `TaskType`s (bridge only via a
`ToolDescriptor` wrapper when needed).

---

## 1. Goal

Grow the Conductor tool catalog from **observed gaps** (`no_capability` /
suggestions), not speculative “add every tool.” New capabilities become
first-class `ToolDescriptor`s so policy can select them without code forks of
the campaign loop.

```text
Campaign step
    │
    ├─ maps to allowlisted tool → Scheduler.dispatch → ToolRegistry.invoke
    │
    └─ unmapped → record_suggestion + no_capability++
                        │
                        ▼
              Gap ledger (aggregate)
                        │
           volume / theme justifies add?
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
     Create capability        Defer / reject
     (descriptor + handler)
            │
            ▼
     Register + refresh allowlist
            │
            ▼
     Next steps can map / invoke
```

---

## 2. Current state (what already exists)

| Piece | Today |
|-------|--------|
| Catalog | `build_default_tool_registry()` / `default_tool_descriptors()` |
| Register API | `ToolRegistry.register(ToolDescriptor)` (in-process, replace-by-name) |
| Gap emit | `map_research_action` → `ActionPlan(unmapped=True)` → `record_suggestion` |
| Counter | `CampaignMetrics.no_capability` on `os_campaign_metrics` |
| Detail | `os_suggestions` rows (`kind`, `message`, `context_json`) |
| Surface | `research conduct` status prints metrics + recent suggestions |
| Allowlist | Frozen once at start of `run_until_stop` (bug for live registration) |

**Gap:** No durable **cross-session** aggregation, no promotion workflow, no
plugin/config path, no versioning/cost hints, no approval gate for enabling new
tools mid-campaign.

---

## 3. When should we add a capability?

Add only when **gap evidence** + **actionability** both hold. Prefer not adding
when the intent is wrong, a synonym of an existing tool, or a one-off.

### 3.1 Hard signals (from metrics)

| Signal | Suggests |
|--------|----------|
| Same `context.intent` or parsed tool name appears **N≥3** times in `os_suggestions` within a rolling window (e.g. 7 days / last K sessions) | Recurring missing action |
| `no_capability` rate **> R%** of campaign steps for a competition or goal class | Catalog too thin for that workflow |
| Operator repeatedly intervenes with the same manual step (comment themes on approvals) | Human is the capability |
| Policy `suggested_tools` names missing from `registry.names()` | Explicit tool request unmet |

Defaults (tunable later): **N=3**, **R=15%**, window = last **20** sessions or **14** days.

### 3.2 Soft signals (human / product)

- Roadmap milestone needs a named tool (e.g. coding adapter) — can short-circuit
  volume gates with explicit design approval.
- Seed/inspect or warm-start workflows blocked by a missing retrieve/act tool.

### 3.3 Do **not** add when

- Intent maps to an existing tool under another name → **alias** / improve
  `map_research_action` templates instead.
- Gap is “run arbitrary Python forever” without sandbox → refuse or route to
  controlled `execute_python` adapter with budgets.
- One-off curiosity from LLM policy with no repeat → leave as suggestion only.
- Request is high-risk (`submit*`, remote git, spend money) without approval path.

### 3.4 Decision record

Every promotion (or reject) writes a **CapabilityDecision**:

```text
gap_key | evidence_count | decision(add|alias|defer|reject)
reason | owner | created_at | linked_tool_name?
```

So evolution is auditable, not chat folklore.

---

## 4. How are new capabilities created?

Three creation paths. All end as a `ToolDescriptor` registered on the same
`ToolRegistry` Conductor uses.

### Path A — Wrap existing library (preferred)

```text
Intent gap → identify existing function/agent
          → ToolDescriptor(name, schemas, handler=fn)
          → register
```

Examples: wrap an Engineer capability, memory CLI, or analyzer as an OS tool.

**Owner:** Platform / Research OS.  
**Risk:** Low if handler already tested.

### Path B — Adapter to external system (reuse data plane)

```text
Intent gap → CodingTool / browser / sandbox adapter
          → ToolDescriptor behind stable name
          → register (may need secrets + approval)
```

Aligns with [02-tools.md](02-tools.md) build-vs-reuse table and backlog
[coding-tool-adapters](../backlog/coding-tool-adapters.md).

**Owner:** Adapters package.  
**Risk:** Medium–high; gate with autonomy / `maybe_approve`.

### Path C — New handler (build)

```text
Intent gap → design I/O + artifacts → implement handler + unit tests
          → ToolDescriptor → register → update map templates / policy allowlist docs
```

Use when no library or adapter exists. Requires tests under `tests/unit/` before
enablement in default catalog.

### 4.1 Capability package shape (proposed)

```text
src/labpilot/research_engine/tools/
  catalog.py                 # default builtins
  registry.py                # register / invoke
  plugins/                   # optional discovered modules
    __init__.py
  gap_ledger.py              # NEW: aggregate suggestions → GapStats
  registration.py            # NEW: promote / register / enable helpers
```

Optional config (later):

```yaml
# labpilot.yaml (illustrative)
tools:
  plugins:
    - labpilot_plugins.eda:register
  enable:
    - run_eda
  require_approval:
    - submit
    - submit_learn
```

**v1 ships without YAML plugins** if code registration + CLI promote is enough.

### 4.2 Descriptor extensions (incremental)

Keep current `ToolDescriptor`; add optional fields when needed:

| Field | Purpose |
|-------|---------|
| `version` | Semver string for plugins |
| `cost_hint` / `duration_hint` | Policy budgeting |
| `risk` | `low` \| `high` — drives approval |
| `gap_keys` | Intents / suggestion patterns this tool closes |
| `enabled` | Soft disable without unregistering |

### 4.3 Live allowlist refresh

`run_until_stop` must **rebuild** `allowlist = set(registry.names())` each
iteration (or after registration), so newly registered tools are usable in the
same session. `decide_next` already re-reads names.

### 4.4 Mapping after register

After add:

1. Register descriptor.
2. Update `_TEMPLATES` / intent → tool map **or** register `gap_keys` so
   `map_research_action` resolves the intent.
3. Optionally re-dispatch the pending unmapped intent once (same step).

Without (2), registration alone does not stop `no_capability`.

---

## 5. How we track `no_capability` to evolve capabilities

### 5.1 Emit (already)

Keep `record_suggestion` as the write path. Enrich `context` (small change):

```json
{
  "intent": "eda_overview",
  "suggested_tools": ["run_eda"],
  "missing_tools": ["run_eda"],
  "competition": "birdclef-2026",
  "session_id": "…",
  "goal": "…"
}
```

Parse structured fields from suggestion messages where possible
(`Need capability/tool 'X'`) into `missing_tools`.

### 5.2 Gap ledger (new)

Durable aggregation **across sessions** (same competition DB or shared
experiences/parent research root — prefer Conductor store extension):

```text
os_capability_gaps
  gap_key          -- stable: intent | missing_tool | normalized message hash
  kind             -- no_capability | alias_candidate | …
  count            -- occurrences
  last_seen_at
  first_seen_at
  sample_contexts  -- JSON array (capped)
  status           -- open | watching | promoted | deferred | rejected
  promoted_tool    -- nullable
```

Update on every `record_suggestion` (increment `count`, refresh `last_seen_at`).

### 5.3 Operator surfaces

| Surface | Behavior |
|---------|----------|
| `research conduct status` | Show top open gaps (key, count, last_seen) |
| `research tools gaps` (new) | List / filter gaps; `--promote`, `--defer`, `--reject` |
| `research tools list` (new or extend) | Registered names + enabled/risk |
| Metrics export (later) | [telemetry-suggestions-export](../backlog/telemetry-suggestions-export.md) |

### 5.4 Evolution loop

```text
1. Collect   — suggestions + no_capability counter (automatic)
2. Aggregate — gap ledger by gap_key
3. Review    — human or scheduled report: top gaps by count × recency
4. Decide    — add | alias | defer | reject (CapabilityDecision)
5. Create    — Path A/B/C + tests
6. Register  — ToolRegistry + map templates + optional approval
7. Verify    — next campaigns: gap count for that key stops rising;
               success = mapped invocations > 0 for new tool
8. Retire    — disable tools with zero use + high failure (later)
```

### 5.5 Success metrics for a new capability

After promotion, watch for **14 days** (or N sessions):

- Gap key `count` growth ≈ 0
- Tool invoke count > 0
- Task failure rate for that tool not worse than catalog median
- No spike in `human_interventions` caused by the new tool

If gaps continue under a new name → improve mapping (alias), not add a twin tool.

---

## 6. Roles and responsibilities

| Role | Owns | Does not own |
|------|------|--------------|
| **GapEmitter** (`map_research_action` + loop) | Emit structured suggestions | Creating tools |
| **GapLedger** | Aggregate / status transitions | Handler implementation |
| **CapabilityAuthor** | Descriptor + handler + unit tests | Conductor policy changes |
| **Registrar** | `ToolRegistry.register`, enable flags, allowlist refresh | Training / Kaggle submit policy |
| **Approver** (operator / autonomy ladder) | Enable high-risk tools | Writing handlers |
| **Conductor policy** | Select among **registered + enabled** names | Inventing tools not in registry |

---

## 7. Impacted files (implementation sketch)

| Area | Files |
|------|--------|
| Emit enrichment | `conductor/actions.py`, `conductor/loop.py`, `conductor/metrics.py` |
| Ledger + schema | `accessor/sqlite/schema.sql`, `migrate.py`, `conductor/store.py` |
| Registration helpers | **new** `tools/registration.py`, `tools/gap_ledger.py` |
| Allowlist refresh | `conductor/loop.py` |
| CLI | `cli/conduct.py`, **new** `cli/tools_cli.py` (list / gaps / promote) |
| Catalog | `tools/catalog.py`, `tools/descriptors.py` (optional fields) |
| Docs | this design; backlog checklist; [02-tools.md](02-tools.md) pointer |
| Tests | `tests/unit/test_campaigns.py` extend; **new** `test_gap_ledger.py`, `test_tool_registration.py` |

---

## 8. Phased delivery

| Phase | Deliverable | Exit |
|-------|-------------|------|
| **P0** | Enrich suggestion `context`; refresh allowlist each loop iteration | Structured gaps; live register works in-session |
| **P1** | `os_capability_gaps` ledger + `research tools gaps` CLI | Cross-session top gaps visible |
| **P2** | `promote` / `defer` / `reject` + CapabilityDecision audit | Evolution loop operable by human |
| **P3** | Plugin entry points + descriptor `risk` / approval | High-risk tools gated; no core fork to add adapters |
| **P4** | Telemetry export of gaps | Feeds [telemetry-suggestions-export](../backlog/telemetry-suggestions-export.md) |

P0–P2 are enough to “keep evolving capability” from real `no_capability` data.
P3–P4 scale the factory.

---

## 9. Non-goals

- Auto-generating tool **implementations** with LLM without tests/approval
- Merging Engineer `BaseCapability` registry into `ToolRegistry` (bridge only)
- Silent enablement of `submit*` / spend / remote-git tools
- Replacing Conductor policy with a free-form agent that invents tools
- Boiling the catalog (“add OpenHands”) without a gap key or milestone ask

---

## 10. Relation to other backlog

| Item | Relation |
|------|----------|
| [coding-tool-adapters](../backlog/coding-tool-adapters.md) | Path B consumer once gaps say “implement better” |
| [future-specialists](../backlog/future-specialists.md) | May expose specialist actions as tools after registration |
| [telemetry-suggestions-export](../backlog/telemetry-suggestions-export.md) | Export gap ledger + `no_capability` |
| [async-conductor](../backlog/async-conductor.md) | Needs stable registry; registration stays sync/in-process first |

---

## 11. Open questions (resolve at impl start)

1. Ledger DB: per-competition Conductor DB vs parent research root shared file?
2. Auto-promote thresholds vs always-human promote for v1? (**Recommend:** human promote in P2; thresholds only open the review queue.)
3. Should `alias` be a first-class ledger status that only edits templates?

**Recommended defaults:** shared parent research root for gap ledger; **human
promote**; aliases as `status=alias` with `promoted_tool` pointing at existing name.
