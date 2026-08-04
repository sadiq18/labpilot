# Design — M10: LLM tiering and role-based routing

**Plan:** [../04-llm-tiering.md](../04-llm-tiering.md) ·
**Status:** design · **Owner:** unassigned · **Build phase:** 1

---

## 1. Background

M14 made the system honest about *whether* an LLM produced a result. M10 is
about *which* LLM, and whether it was capable of the work.

The plan states the principle this milestone exists to enforce:

> Model capability is a product tier, not an architectural constraint.

A paying customer supplies a frontier model. The free tier is a development
mode with known limits. Neither should be the substrate the architecture is
built around — and today the architecture is built around whatever
`resolve_llm_client` happens to return first.

The decision layer for tiering already exists (`6d4e930`): `llm/catalog.py`
(providers, plans, entitlement), `llm/budget.py` (persistent rate ledger),
`llm/router.py::select_route` (entitlement ∩ data policy ∩ budget), and 15
tests. It has **zero call sites under `src/`**. The review that caught this
described it accurately: tested, unwired, and described as done.

M10 is the wiring.

---

## 2. Problem statement

### 2.1 Three routing mechanisms coexist; none of them tiers

| Mechanism | Selects by | Production call sites | Knows about limits? |
|---|---|---|---|
| `resolve_llm_client` (`client.py:166`) | provider priority order | **15** | no |
| `resolve_route` (`router.py:187`) | `task` profile from YAML | 1 (`LLM.generate`) | no |
| `select_route` (`router.py:114`) | **role + tier + budget** | **0** | yes |

The one that implements the design is the one nothing calls. The one that runs
everything picks the first provider whose key happens to be set, with no notion
of whether that model can do the work in front of it.

M10 must **collapse these to one**. Adding role routing beside the other two
would make it the fourth mechanism, and the next reviewer would be right to ask
which of the four is real.

### 2.2 The plan's own priority item is wrong

The plan says, of `prior_train[:120_000]`:

> At ~4 chars/token that is ~30k tokens per codegen call; ten calls eats a daily
> allowance. **Do this first** — it is ten minutes and it is the difference
> between the free tier lasting a day or an hour.

Measured 2026-08-04 by rendering `CodeEngineerAgent.user_prompt` against the
real rogii workspace — actual `pipeline/train.py` as `prior_train_py`, actual
`profile.json` as `profile_summary`, actual `baseline_choice.json`:

```
prior train.py on disk :   12,146 chars
profile.json on disk   :   13,821 chars
system prompt          :    2,641 chars
user prompt (rendered) :   23,036 chars
TOTAL                  :   25,677 chars  ~= 6,419 tokens
```

**~6.4k tokens, not ~30k.** The 120,000-char slice never binds, because
`code_engineer/agent.py:65` already caps the same field at 20,000 chars before
rendering — and nothing this system has produced approaches even that. The trim
is a no-op on every input observed so far.

This matters beyond one line item: the plan's stated *first* action was chosen
from an estimate that was 5× off, and it would have been done, felt productive,
and changed nothing.

### 2.3 The real cost driver is that nothing is cached or metered

`.cache/llm.sqlite` holds **one row**, created 2026-07-27. Nine rogii campaigns
have run since.

The cause: micro agents receive an `OllamaClient` / `OpenAIClient` and call
`.complete()` on it directly. `PromptCache` lives inside `LLM.generate`, which
micro agents never reach. So the cache covers a path almost nothing uses, and
every agent call in every campaign is uncached — and unmetered, since
`BudgetLedger.record` has no caller either.

A free tier with a daily token cap is spent by uncached repeats, not by one
oversized field.

### 2.4 Two latent breaks in the unwired layer

**Credentials are invisible.** `ProviderSpec.has_credentials()` reads
`os.environ[api_key_env]`. `Settings` loads the workspace `.env` through
pydantic-settings, which populates the settings object and **does not export to
the environment** (`accessor/kaggle/client.py:72` exports its keys explicitly,
precisely because of this). A key that lives only in `.env` is invisible to the
catalog, `eligible_providers` returns `[]`, and the campaign reports "no
eligible provider" while the key sits in the file the user just edited.

**`json_mode` is dropped.** `_BoundClient.complete` (`client.py:360`) calls
`LLM.generate(task="default", ...)` and takes no `json_mode` argument. Routing
micro agents through the `LLM` facade as-is would silently undo the constrained
JSON decoding fix — measured on rogii as 3/3 campaigns falling back before the
fix and 0/2 after. This is the single highest-value behaviour in the LLM layer
and the obvious wiring would discard it.

---

## 3. Goal

Every LLM call in the system is made through **one** resolution path that knows
what class of work it is serving, what the user's plan permits, what the
provider's published limits are, and what has already been spent — and records
which model actually served it.

---

## 4. Requirements

### Functional

- **F1** — A call site declares a **role** (`codegen` | `reasoning` |
  `summarize`), never a provider name or model.
- **F2** — Role resolution goes through `select_route`, honouring plan
  entitlement, `allow_training_on_inputs`, and the budget ledger.
- **F3** — An OpenAI-compatible adapter serves any provider given `base_url` +
  `api_key_env` (covers Groq, GitHub Models, OpenRouter, Mistral, Cerebras).
- **F4** — Every call is metered to the ledger and consults the prompt cache,
  including micro-agent calls.
- **F5** — The served provider and model are recorded on the experiment record,
  alongside M14's `generated_by`.
- **F6** — `research doctor` reports the resolved provider per role and fails
  loudly when a role has no capable provider.
- **F7** — `codegen` and `reasoning` **wait** on exhaustion; `summarize`
  degrades, and every degradation is stamped.
- **F8** — Credentials resolve from `Settings` as well as the process
  environment.

### Non-functional

- **NF1** — `json_mode` survives the wiring. A regression test asserts the
  constrained-decoding request reaches the provider.
- **NF2** — The 15 existing `resolve_llm_client` call sites and the 95
  `llm_client=` passes keep working during migration. No big-bang edit.
- **NF3** — Cache keys include the model (already true in `cache_key`) so a
  stored result is attributable to what produced it.
- **NF4** — Waiting is bounded and observable. An unbounded wait is
  indistinguishable from a hang in an unattended campaign.
- **NF5** — No multi-account key rotation. Rotating *across* providers within
  published limits only.

---

## 5. Scope

### In scope

- OpenAI-compatible adapter; provider → client factory
- One resolution entry point; `resolve_llm_client` reduced to a shim over it
- Role declaration on micro agents
- Metering and caching moved into the client wrapper
- Routing config in YAML; `research doctor` per-role report
- Retiring or mapping `resolve_route`'s task profiles

### Out of scope

- **Choosing providers or creating accounts.** M10 needs at least one free-tier
  API key to be verifiable end to end, and account creation is the user's to
  do — see §13.
- Streaming, tool-calling, or multi-turn. The contract stays
  `complete(system, user) -> str`.
- Cost accounting in currency. The ledger counts requests and tokens.
- Anything M7 owns (what the generated code contains).

---

## 6. High-level design

```
call site
   │  declares role: "codegen"
   ▼
LLMGateway.for_role("codegen")
   │
   ├─► select_route(routing, role, ledger)      ← entitlement ∩ policy ∩ budget
   │        │
   │        ├─ provider ────────────────► client_for(spec)   ← adapter factory
   │        └─ wait_seconds ────────────► RoleUnavailable(wait)
   ▼
RoleBoundClient.complete(system, user, json_mode=True)
   │
   ├─ cache.get(key incl. model) ─── hit ──► return
   ├─ provider.complete(...)
   ├─ ledger.record(provider, tokens)
   └─ stamp: served_provider, served_model, degraded
```

Three things move *into* the wrapper, deliberately: cache lookup, ledger
recording, and the served-model stamp. Every one of them is currently the
caller's job, and §2.3 is what that produced.

---

## 7. Components and responsibility boundaries

| Component | Owns | Must not |
|---|---|---|
| `llm/catalog.py` | provider data, plan entitlement | know about budget or clients |
| `llm/budget.py` | spend against published limits | know about roles or providers' identity beyond a name |
| `llm/router.py::select_route` | the decision | construct clients or perform I/O |
| `llm/adapters.py` **(new)** | HTTP shapes: `openai_compat`, `gemini`, `ollama` | decide *which* provider |
| `llm/gateway.py` **(new)** | role → client; cache, meter, stamp | contain routing rules |
| `BaseMicroAgent` | declares `llm_role` | resolve a provider |
| call sites | declare the role of the work | name a model |

The boundary that matters: **`select_route` decides, `gateway` executes.** A
router that constructs clients cannot be unit-tested without network, which is
how routing logic ends up untested everywhere else.

---

## 8. Design choices

### 8.1 Collapse to one router; do not add a third

`select_route` becomes the only resolver. `resolve_llm_client(config)` survives
as a shim that resolves role `default`, so the 15 call sites and 95
`llm_client=` passes keep working while they migrate. `resolve_route` and the
`llm.tasks` YAML block are mapped to roles (`planning`→`reasoning`,
`coding`→`codegen`, `summary`→`summarize`) and then deleted.

*Rejected:* keeping task profiles as a user-facing override. Two names for the
same axis is what produced three routers.

### 8.2 The role belongs to the agent, not to 95 call sites

`BaseMicroAgent` gains `llm_role: str = "reasoning"`; `CodeEngineerAgent` sets
`"codegen"`; summarisers set `"summarize"`. Construction stays
`Agent(llm_client=...)`, and when what is passed is an `LLMGateway`, the base
class binds `gateway.for_role(self.llm_role)` once in `__init__`.

This is why the role lives on the agent: it makes 95 sites correct without
editing them, and it puts the requirement next to the prompt that creates it —
whoever writes a prompt knows what class of model it needs.

*Rejected:* inferring the role from the agent's class name. Silent, and wrong
the first time an agent is renamed.

### 8.3 Meter and cache in the wrapper, not at call sites

Both are cross-cutting and both are currently opt-in. The result is one cache
row across nine campaigns. If a call goes through `RoleBoundClient`, it is
metered; there is no second path.

*Consequence to accept:* the ledger records requests it cannot always attribute
tokens to (Ollama returns counts, some OpenAI-compatible endpoints do not).
Record `0` tokens rather than estimating — an invented number in a quota ledger
causes a wait that cannot be explained.

### 8.4 Credentials come from `Settings`, then the environment

`has_credentials()` gains an optional resolver so `.env`-only keys are visible.
The alternative — exporting every key into `os.environ` at startup — leaks
credentials into every subprocess the runtime spawns, including generated
training code.

### 8.5 `codegen` waits; `summarize` degrades — but waiting must be bounded

Preserved from the plan, and the reasoning is worth restating because it is the
whole justification for the milestone: a weak model that implements a technique
badly causes the system to record *"technique X did not help"* — a false
negative indistinguishable from a real one, which then poisons hypothesis
generation permanently.

New constraint: `wait_seconds` is bounded by `max_wait_seconds` per role
(default 900s). Beyond it, raise `RoleUnavailable`. An unbounded wait in an
unattended campaign presents exactly like a hang, and the operator's only
recourse is to kill a run whose state they cannot inspect.

### 8.6 `json_mode` is a routing requirement, not an optimisation

The adapter contract includes `json_mode`, every adapter implements it or
declares it unsupported, and `ProviderSpec` records `supports_json_mode`. A
provider that cannot constrain output is not eligible for roles whose agents
parse JSON — which is all of them.

This is a direct consequence of §2.4: the natural wiring drops the flag, and
the measured effect of dropping it is a 3/3 fallback rate.

### 8.7 M10 does not ship on unit tests

The exit criterion is a real codegen call producing a working `train.py` — §11.
The plan already says this; it is repeated here because `select_route` was
tested, unwired, and described as done, and the same pressure applies again.

---

## 9. Low-level design

### 9.1 Adapter contract

```python
# llm/adapters.py
class ProviderAdapter(Protocol):
    supports_json_mode: bool
    def complete(self, system: str, user: str, *, model: str,
                 temperature: float, json_mode: bool = False) -> Completion: ...

@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0      # 0 when the endpoint does not report usage
    completion_tokens: int = 0

class OpenAICompatAdapter:
    """Any /v1/chat/completions endpoint: Groq, GitHub Models, OpenRouter, …"""
    def __init__(self, base_url: str, api_key: str) -> None: ...
```

`json_mode` maps to `response_format={"type": "json_object"}` for
OpenAI-compatible endpoints and `format: "json"` for Ollama (already
implemented, `ollama.py:57`).

### 9.2 Gateway

```python
# llm/gateway.py
class LLMGateway:
    def __init__(self, routing: RoutingConfig, ledger: BudgetLedger,
                 cache: PromptCache, settings: Settings) -> None: ...

    def for_role(self, role: str) -> RoleBoundClient:
        """Resolve now; the returned client re-resolves per call so a provider
        exhausted mid-campaign is not held for the campaign's lifetime."""

class RoleBoundClient:
    role: str
    last_served: ServedBy | None      # provider, model, tier, degraded

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        decision = select_route(self._routing, self.role, self._ledger)
        if decision.provider is None:
            if decision.wait_seconds > self._max_wait:
                raise RoleUnavailable(self.role, decision.reason)
            sleep(decision.wait_seconds); ...retry once
        ...cache lookup → adapter call → ledger.record → stamp
```

Re-resolving per call rather than per agent is deliberate: campaigns run for
hours and a provider's daily cap is reached mid-run. A client bound once holds a
dead provider until the process exits.

### 9.3 Agent binding

```python
# accessor/common/micro_agents.py
class BaseMicroAgent:
    llm_role: str = "reasoning"

    def __init__(self, llm_client: LLMClient | LLMGateway | None = None) -> None:
        if isinstance(llm_client, LLMGateway):
            llm_client = llm_client.for_role(self.llm_role)
        self.llm_client = llm_client
        ...
```

Plain stubs still work unchanged, so no test edits are forced — the same
property that let M14 phase 1 land without touching 95 sites.

### 9.4 Provenance extension (F5)

M14 records *what kind of thing* produced a result. M10 adds *which model*:

```python
last_generated_by: GeneratedBy    # existing: llm | rule_engine | …
last_served: ServedBy | None      # new: provider, model, tier, degraded
```

Both are written in `BaseMicroAgent.run`, and the experiment record carries
both. Without this, a failed hypothesis cannot be attributed to the idea rather
than the writer — the distinction the whole milestone exists to protect.

### 9.5 Config

```yaml
llm:
  plan: free                    # free | pro | enterprise
  allow_training_on_inputs: true
  providers:
    - {name: ollama, kind: ollama, tier: local, strong: false,
       models: {default: qwen2.5-coder:14b}}
    # add free-tier entries here once keys exist — see §13
  roles:
    codegen:   {requires_strong: true,  on_exhaustion: wait,    max_wait_seconds: 900}
    reasoning: {requires_strong: true,  on_exhaustion: wait,    max_wait_seconds: 900}
    summarize: {requires_strong: false, on_exhaustion: degrade}
    default:   {requires_strong: false, on_exhaustion: degrade}
```

The shipped default has one local provider and no keys, so a fresh checkout
behaves exactly as it does today.

### 9.6 Not doing

`prior_train[:120_000]` → `[:20_000]` is a one-word change that aligns the two
caps and removes a misleading number. Do it for tidiness, **not** as a
performance item, and do not report it as one (§2.2).

---

## 10. Testing strategy

| Level | Covers | Network |
|---|---|---|
| Unit | adapter request shape, gateway cache/meter/stamp, credential resolution | no |
| Contract | `json_mode` reaches the adapter for every role | no |
| Regression | `resolve_llm_client` shim returns a working client with no routing config | no |
| Integration | one real call per configured provider, marked `llm` | yes, opt-in |
| End-to-end | §11 — a real codegen call produces a working `train.py` | yes |

Specific tests worth naming, because each maps to a break found in §2:

1. **`test_json_mode_survives_gateway`** — a fake adapter asserts
   `json_mode=True` arrives. Guards the §2.4 regression.
2. **`test_every_call_is_metered`** — N gateway calls ⇒ N ledger rows. Guards
   §2.3 (one cache row / nine campaigns).
3. **`test_cache_hit_does_not_spend_quota`** — repeat call ⇒ 1 ledger row.
4. **`test_dotenv_only_key_is_visible`** — key in `.env`, absent from
   `os.environ`, provider still eligible. Guards §2.4.
5. **`test_codegen_waits_and_summarize_degrades`** — exhausted strong provider;
   assert wait for `codegen`, degraded stamp for `summarize`.
6. **`test_wait_is_bounded`** — `wait_seconds > max_wait` raises rather than
   sleeping.
7. **Source guard**, in the style of `test_agent_provenance.py`: no module under
   `src/` calls `resolve_route` or names a provider string for an LLM call once
   migration completes.

Default CI slice stays `uv run pytest -m "not llm and not image and not deep"`;
every networked test carries `llm`.

---

## 11. Evaluation

### 11.1 What is already measured

| Claim | Status | Evidence |
|---|---|---|
| Codegen prompt ≈ 30k tokens | **refuted** | 6,419 tokens measured, rogii |
| `prior_train[:120_000]` is the cost driver | **refuted** | agent caps at 20k; real input is 12k |
| Prompt caching reduces spend | **refuted as shipped** | 1 cache row across 9 campaigns |
| `select_route` is on the live path | **refuted** | 0 call sites under `src/` |
| Weak model breaks JSON parsing | **confirmed** | 3/3 fallback pre-fix, 0/2 post-fix |

### 11.2 Exit criteria

From the plan, unchanged in substance:

1. `select_route` is on the live path — no call site resolves a provider by name
   for `codegen`, `reasoning` or `summarize`. *Verified by source guard.*
2. `research doctor` reports the resolved provider per role and fails loudly
   when a role has none. *Verified by output inspection.*
3. **A real codegen call produces a working `train.py`** for one technique on
   rogii: it runs, writes metrics, and the technique is visible in the generated
   source. *This is the forcing function.*
4. The served model is recorded on the experiment record.
5. Budget exhaustion on `codegen` produces a wait, not a downgrade — **observed
   in a campaign**, not only unit-tested.

Criterion 3 is the one that matters. Everything else can pass while the system
still cannot write code.

### 11.3 Rogii protocol

Same shape as M14's eval, and for the same reason — a campaign that never
reaches the code path proves nothing about it:

1. Record the baseline: rogii's honest validation MSE is **194.8** on 154
   held-out wells, against a naive anchor of 226.3.
2. Run `research conduct` with routing live and one strong provider configured.
3. Accept only if: ≥1 codegen call served by the strong provider, the generated
   `train.py` runs to completion, and `metrics.json` is written.
4. Report the served model per experiment.

**Exclusion rule, carried over from M14's eval:** campaigns where the backlog
gate suppressed all agent calls must be excluded, not counted as clean. Counting
them once turned "0 of 3 fell back" into a false "6 of 9 clean".

### 11.4 What would falsify the milestone

If a strong provider is configured and codegen still cannot produce a running
`train.py`, M10 is not the blocker and M7's recipe registry is the floor that
matters. That outcome should be recorded, not worked around — it reorders the
roadmap.

---

## 12. Observability

Per call, at INFO: `role`, `provider`, `model`, `tier`, `cache_hit`,
`degraded`, `wait_seconds`, `tokens`.

Per campaign summary: calls and tokens by role and provider, degradation count,
total wait. This is the table that says whether a free tier can sustain a
campaign — and there is no way to know today, because nothing counts.

`research doctor` gains a routing section:

```
LLM routing (plan: free)
  codegen    ✗  no capable provider — set GROQ_API_KEY or add a paid provider
  reasoning  ✓  ollama · qwen2.5-coder:14b (local, weak — codegen will wait)
  summarize  ✓  ollama · qwen2.5-coder:14b (local)
```

A role with no capable provider is a **failure**, not a warning. That is the
M14 posture applied one level up: a missing capable model means the reasoning
will be wrong, and wrongness propagates into beliefs and claims where it is
expensive to remove.

---

## 13. Production readiness

**Prerequisite the user must supply.** M10 cannot be verified end to end without
at least one strong-provider API key. Obtaining one means creating an account,
which is not something I will do on the user's behalf. Before build phase 4,
the user provides a key for one free-tier provider (Groq, GitHub Models,
OpenRouter, Cerebras, or Mistral all fit the `openai_compat` adapter) or a paid
key. Phases 1–3 are buildable and testable without it.

**Phasing**, mirroring M14's, so each phase is independently verifiable:

| Phase | Work | Verifiable by |
|---|---|---|
| 1 | adapters + gateway + config, nothing wired | unit + contract tests |
| 2 | cache + meter + stamp inside the wrapper; `doctor` routing section | ledger rows in a local ollama campaign |
| 3 | `llm_role` on agents; `resolve_llm_client` becomes a shim; retire `resolve_route` | full suite + source guard |
| 4 | **thin M7 slice** — one technique, real codegen, working `train.py` | rogii, §11.3 |

**Rollback.** Phases 1–3 are inert with the shipped config (one local provider,
no keys) and behave as today. Phase 4 is where behaviour changes.

**Risk — the wait becomes the hang.** A campaign that waits on `codegen` for 15
minutes per call, unattended, looks broken. Mitigation: bounded wait (§8.5),
the wait logged at INFO with its reason, and the campaign summary reporting
total wait.

**Risk — free-tier data policy.** `allow_training_on_inputs: false` must be
honoured before any competition-sensitive content is sent. It is enforced in
`eligible_providers` today and needs a test that a workspace setting it false
gets no free-tier provider at all.

**Trap — do not rotate multiple accounts per provider.** Rotating *across*
providers within published limits is legitimate. Multi-account key rotation to
evade limits is ToS abuse and will kill the accounts at the worst moment.

**Trap — config-driven model names, always.** Free tiers deprecate models
constantly; anything hardcoded breaks unattended.
