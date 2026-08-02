# Design — Capability registration

Back to [../README.md](../README.md) · Backlog:
[../backlog/capability-registration.md](../backlog/capability-registration.md) ·
Tools: [02-tools.md](02-tools.md) · Campaigns: [06-campaigns.md](06-campaigns.md).

**Status:** Parked (design accepted; impl blocked)  
**Depends on:** M3 campaign loop + `ToolRegistry` (shipped); **impl blocked on**
[telemetry-suggestions-export](../backlog/telemetry-suggestions-export.md)
(client log / gap collection)  
**Does not replace:** Research Engineer `BaseCapability` map (plan-task handlers) —
Conductor gaps are **OS tools**, not Engineer `TaskType`s (bridge only via a
`ToolDescriptor` wrapper when needed).

**Pickup rule:** Do not start P0–P4 until telemetry can collect redacted gaps
from client installs into a maintainer-readable feed.

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

### 5.2 Two stores: local (user) vs product (maintainer)

`os_suggestions` / `os_campaign_metrics` live in the **user’s competition
knowledge DB**. LabPilot maintainers cannot read that DB remotely. So the
design splits storage by audience:

| Store | Where | Audience | Purpose |
|-------|-------|----------|---------|
| **Local emit** | User competition DB (`os_suggestions`, `no_capability`) | End user / same machine | Debug “why did my campaign stall?” |
| **Local rollup** (optional) | Same DB or research-root `os_capability_gaps` | Same machine | Cross-session view on that laptop |
| **Product gap feed** | Opt-in export / telemetry (not the user’s DB) | **LabPilot maintainers only** | Decide what to add to the shared catalog |

Without an export bridge, local suggestions never reach the people who merge
tools into LabPilot. That bridge is required for product evolution — not optional
polish. See [telemetry-suggestions-export](../backlog/telemetry-suggestions-export.md).

```text
User machine                         Maintainer / LabPilot product
─────────────                        ────────────────────────────
record_suggestion
no_capability++
     │
     ▼
os_suggestions (local SQLite)
     │
     ├─ research conduct status     ← user-facing (read-only summary)
     │
     └─ opt-in export / telemetry ──► product gap feed (aggregate)
              (P1 for product use)         │
                                           ▼
                                    maintainer review queue
                                           │
                                    promote / defer / reject
                                           │
                                    PR into tools/catalog.py
                                           │
                                           ▼
                                    all users on next release
```

**Export contents (minimal, privacy-aware):** gap_key, kind, count, last_seen,
hashed/redacted sample contexts — **not** full goals, artifacts, or secrets.
Default **off**; user (or lab deploy) opts in.

### 5.3 Local gap rollup (optional helper)

On the user machine, aggregate across sessions for local debugging:

```text
os_capability_gaps   (local only)
  gap_key | kind | count | last_seen_at | first_seen_at
  sample_contexts (capped) | status (open|watching) 
```

Update on every `record_suggestion`. Local status does **not** mean “will ship
in LabPilot.” Product promote/reject lives only on the maintainer feed.

### 5.4 Surfaces by audience

| Surface | Who runs it | Behavior |
|---------|-------------|----------|
| `research conduct status` | **User** | Show `no_capability` + recent suggestions / top local gaps (read-only) |
| `research tools list` | **User** | Registered tool names (what Conductor can call) |
| `research tools export-gaps` | **User** (opt-in) | Write redacted gap aggregate JSON (or push telemetry) — no promote |
| `labpilot-maint tools gaps` (or gated `research tools gaps`) | **Maintainer only** | Read **product gap feed**; `--promote` / `--defer` / `--reject` |
| Dashboards (Phoenix / Langfuse / …) | **Maintainer** | Same feed, visual |

**Rule:** End users never promote into the shared LabPilot catalog. Promote is a
maintainer action on exported/aggregated data, followed by a normal code PR.

Gating options for maintainer CLI (pick at impl):

1. Separate entrypoint / package extra (`labpilot[maint]`)
2. Env flag `LABPILOT_MAINTAINER=1` required for `gaps --promote`
3. Offline tool that only reads a downloaded export file (no user DB access)

Recommend **(3) for v1** (simplest, no privilege on user installs): maintainer
runs review against an export artifact, not against live user SQLite.

### 5.5 Evolution loop (product)

```text
1. Collect   — user machines: suggestions + no_capability (automatic, local)
2. Export    — opt-in redacted aggregates → product gap feed
3. Aggregate — maintainer store: gap_key × count × recency across users
4. Review    — maintainer only: top gaps
5. Decide    — add | alias | defer | reject (CapabilityDecision in product tracker)
6. Create    — Path A/B/C + tests in LabPilot repo
7. Register  — merge ToolDescriptor + map templates
8. Verify    — post-release: exported gap growth for that key drops;
               tool invoke count > 0 where telemetry exists
9. Retire    — disable unused / high-failure tools (later)
```

Local-only loop (same laptop, no export): useful for **local plugins** later;
does not ship capabilities to all users.

### 5.6 Success metrics for a new capability

After a catalog tool ships, watch for **14 days** (or N exported sessions):

- Gap key `count` growth in product feed ≈ 0
- Tool invoke count > 0 (where telemetry exists)
- Task failure rate for that tool not worse than catalog median
- No spike in `human_interventions` caused by the new tool

If gaps continue under a new name → improve mapping (alias), not add a twin tool.

---

## 6. Roles and responsibilities

| Role | Owns | Does not own |
|------|------|--------------|
| **End user** | Run campaigns; see local status; optionally export gaps | Promote into LabPilot; edit shared catalog |
| **GapEmitter** (`map_research_action` + loop) | Emit structured suggestions locally | Creating tools |
| **Local rollup** | Aggregate on user DB for status UX | Product decisions |
| **Export / telemetry** | Ship redacted aggregates off-machine | Handler implementation |
| **Maintainer (you)** | Review product gap feed; decide; implement PR | Reading raw user DBs by default |
| **CapabilityAuthor** | Descriptor + handler + unit tests in repo | Conductor inventing tools |
| **Registrar** | Merge register path + allowlist refresh in product | Training / Kaggle submit policy |
| **Approver** (autonomy ladder) | Enable high-risk tools at runtime | Writing handlers |
| **Conductor policy** | Select among **registered + enabled** names | Inventing tools not in registry |

---

## 7. Impacted files (implementation sketch)

| Area | Files |
|------|--------|
| Emit enrichment | `conductor/actions.py`, `conductor/loop.py`, `conductor/metrics.py` |
| Local rollup | `accessor/sqlite/schema.sql`, `migrate.py`, `conductor/store.py` |
| Export | **new** gap export serializer; ties to [telemetry-suggestions-export](../backlog/telemetry-suggestions-export.md) |
| Maintainer review | Offline tool or gated CLI over **export files / feed**, not user DB |
| Registration helpers | **new** `tools/registration.py`, `tools/gap_ledger.py` |
| Allowlist refresh | `conductor/loop.py` |
| User CLI | `cli/conduct.py` status; `research tools list`; `research tools export-gaps` |
| Maintainer CLI | gaps list / promote / defer / reject (not default user path) |
| Catalog | `tools/catalog.py`, `tools/descriptors.py` (optional fields) |
| Docs | this design; backlog checklist; [02-tools.md](02-tools.md) pointer |
| Tests | `tests/unit/test_campaigns.py` extend; **new** `test_gap_ledger.py`, `test_gap_export.py` |

---

## 8. Phased delivery

| Phase | Deliverable | Exit |
|-------|-------------|------|
| **P0** | Enrich suggestion `context`; refresh allowlist each loop iteration | Structured local gaps; live register works in-session |
| **P1** | **Export bridge** — `export-gaps` (file) and/or telemetry of redacted aggregates | Maintainer can read gaps **without** user SQLite |
| **P2** | Maintainer review over export/feed — promote / defer / reject + CapabilityDecision | Product evolution loop operable by maintainer only |
| **P3** | Optional local `os_capability_gaps` rollup for `conduct status` UX | Users see top local gaps read-only |
| **P4** | Plugin entry points + descriptor `risk` / approval | Local/private tools without forking core; high-risk gated |

P0 + **P1–P2** are the minimum to evolve the **shared** catalog from real user
gaps. Local rollup (P3) helps the user; it does not replace export.

---

## 9. Non-goals

- Auto-generating tool **implementations** with LLM without tests/approval
- Merging Engineer `BaseCapability` registry into `ToolRegistry` (bridge only)
- Silent enablement of `submit*` / spend / remote-git tools
- Replacing Conductor policy with a free-form agent that invents tools
- Boiling the catalog (“add OpenHands”) without a gap key or milestone ask
- Giving end users a promote path into the shared LabPilot catalog
- Maintainers reading raw user competition DBs as the default workflow
- Exporting goals, artifacts, API keys, or full suggestion text by default

---

## 10. Relation to other backlog

| Item | Relation |
|------|----------|
| [coding-tool-adapters](../backlog/coding-tool-adapters.md) | Path B consumer once gaps say “implement better” |
| [future-specialists](../backlog/future-specialists.md) | May expose specialist actions as tools after registration |
| [telemetry-suggestions-export](../backlog/telemetry-suggestions-export.md) | **Required bridge** for product gap feed (not a nice-to-have) |
| [shared-multi-tenant-store](../backlog/shared-multi-tenant-store.md) | Later home for multi-user product gap feed |
| [async-conductor](../backlog/async-conductor.md) | Needs stable registry; registration stays sync/in-process first |

---

## 11. Open questions (resolve at impl start)

1. Export transport for v1: local JSON file the user sends, vs opt-in OTel/HTTP?
   (**Recommend:** file export first; telemetry when ops pain justifies.)
2. Product gap feed storage for maintainers: repo `gaps/` inbox, private issue
   tracker, or Langfuse/Phoenix project?
3. Auto-promote thresholds vs always-human promote for v1?
   (**Recommend:** human promote only; thresholds open the review queue.)
4. Should `alias` be a first-class decision that only edits map templates?
5. Local rollup DB: per-competition vs research-root file? (UX only; not product SoR.)

**Recommended defaults:** file-based redacted export → maintainer offline review;
**human promote**; aliases as `status=alias` with `promoted_tool` = existing name;
local rollup optional and never authoritative for the shared catalog.
