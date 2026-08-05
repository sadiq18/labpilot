# Design — a budget-aware, outcome-learning LLM router

**Working name:** `fitroute` (placeholder — see §13.6) ·
**Status:** design · **License intent:** Apache-2.0 ·
**First consumer:** labpilot ([M10](../research-os/autonomy-roadmap/04-llm-tiering.md))

---

## 1. Background

This started as labpilot's internal M10 — route free-tier LLM calls so a
pre-revenue project could keep running. That framing was too small. The problem
is general, the constraint is not really "free", and the piece that would make
it valuable to other people is the piece labpilot needs most anyway.

The reframing, in the author's words: a router for **all** types of user, that
works like a load balancer across providers based on **budget available**, and
makes sure the caller gets the **best model that fits the work**.

Two measurements from labpilot motivate the design, and both generalise:

- A weak model answered a JSON-only prompt in English prose. The call returned
  **HTTP 200**. Every layer above treated it as success, fell back to a
  deterministic stub, and recorded a research finding that never happened.
  Fallback rate went 3/3 → 0/2 once constrained decoding was requested.
- The prompt cache holds **one row across nine campaigns**, because caching was
  the caller's job and the callers didn't. Nothing counted tokens at all.

Neither is a labpilot bug. Both are what happens when model choice, capability
checking, spend accounting and result validation are each somebody else's job.

---

## 2. Problem statement

### 2.1 What existing routers do and don't do

| | OpenRouter | LiteLLM | This |
|---|---|---|---|
| Many providers, one interface | ✅ | ✅ | ✅ |
| Picks the model for you | `auto`, on generic quality | no | **per work type** |
| Knows if a model *can* do the task | ❌ | flags exist, unused for routing | **hard precondition** |
| Treats bad output as failure | ❌ HTTP-level only | ❌ | **schema/verdict-level** |
| Learns from your results | ❌ | ❌ | **core mechanism** |
| Your local models in the pool | ❌ hosted only | ✅ | ✅ **first-class** |
| Spend cap per work type | one balance | ❌ | ✅ |
| Runs in your process, no middleman | ❌ | ✅ | ✅ |

OpenRouter is a **marketplace and gateway**. This is a **policy engine**. They
are not the same category, and the goal is not to compete on model breadth —
OpenRouter can be one of this router's providers.

### 2.2 The seven pain points, stated plainly

1. **"Which model should I use for this?"** Every framework makes you answer
   this per call site, forever, and re-answer it whenever a model is deprecated.
2. **"It worked on GPT-4o and broke on Llama."** No one checks that the chosen
   model supports structured output before sending a prompt that requires it.
3. **"I don't know what my free limits actually are."** No registry publishes
   them; they vary by account and change without notice.
4. **A 200 that's garbage counts as success.** Failover triggers on HTTP status.
   Schema-validation failure — the most common real failure — triggers nothing.
5. **You can't cap spend per kind of work.** Summarisation quietly eats the
   budget that codegen needed.
6. **Sensitive data has nowhere to go.** Free tiers commonly reserve the right
   to train on inputs, and there is no switch that says "not this workload".
7. **Runs aren't reproducible.** You cannot replay yesterday's agent run against
   what the models actually returned.

### 2.3 What can and cannot be synced (researched 2026-08-05)

| Source | Provides | Missing |
|---|---|---|
| [models.dev/api.json](https://models.dev/api.json) | per-provider `env` var + `api` base URL; per-model `structured_output`, `tool_call`, `context`, `output`, `modalities`, `cost{input,output,cache_read,cache_write}`, `open_weights`, `release_date` | rate limits, free-tier flag |
| [OpenRouter `/api/v1/models`](https://openrouter.ai/api/v1/models) | pricing (`0` for `:free` variants), `supported_parameters`, `benchmarks`, `context_length`, `expiration_date` | per-day quotas |
| [LiteLLM prices JSON](https://github.com/BerriAI/litellm) | `supports_response_schema`, `supports_function_calling`, costs, token limits | rpm/tpm, free tier |
| [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) | actual free-tier numbers | markdown, hand-maintained, explicitly volatile |

**Conclusion that shapes the design:** model *facts* are syncable from three
independent APIs. Free-tier *budgets* are not machine-readable anywhere, and the
lists that have them warn that they change constantly.

So budgets are **learned from the account's own traffic** — `Retry-After`,
`X-RateLimit-*`, and observed 429 ceilings — rather than trusted from docs. This
is not a workaround. Limits differ per account, per tier and per model, so the
observed number is the only correct one.

---

## 3. Goal

A caller declares **what kind of work** it is doing and **what it needs**. The
router returns the best model that can actually do it, within entitlement, data
policy and budget — and gets better at that answer every time the caller says
whether the result worked.

---

## 4. Requirements

### Functional

- **F1** — Callers declare a **role** and its **requirements**, never a model
  name (unless overriding, F7).
- **F2** — **Capability preflight**: a model that lacks a required capability
  (structured output, tool calling, context length, modality) is not a
  candidate. Hard filter, before any ranking.
- **F3** — **Budget-aware selection** across two independent limits: request
  rate (rpm/rpd/tpm) and currency spend, each per role and globally.
- **F4** — **Outcome memory**: the caller reports a verdict per call; the router
  maintains a per-(role, model) posterior and prefers what has been working.
- **F5** — **Verdict-level failover**: a schema-invalid or refused response
  fails over to the next candidate, exactly as a 5xx does.
- **F6** — **Registry sync**: model facts refreshed asynchronously from multiple
  sources, merged with precedence, overridable locally, usable offline.
- **F7** — **Override**: pin a specific provider/model at call, scope, env and
  config level. Pinning is recorded, never silent.
- **F8** — **Caching**: exact response cache keyed on model+system+prompt+params;
  cache-aware routing that prefers providers whose own prompt cache would hit.
- **F9** — **Conversation memory**: an optional thread object that fits history
  to the *chosen* model's context window.
- **F10** — **Data-policy routing**: refuse providers that may train on inputs,
  or that sit outside an allowed region, per workload.
- **F11** — **Record & replay**: every request/response recorded; a run replays
  offline against the recording.

### Non-functional

- **NF1** — The routing **decision is a pure function**. No I/O, no clock, no
  network. Testable exhaustively; portable to another language later (§8.1).
- **NF2** — Never block a call on registry sync. A stale registry serves; a
  missing one falls back to a vendored snapshot.
- **NF3** — Selection overhead < 1 ms. This sits in the hot path.
- **NF4** — Zero required configuration for the common case: an API key in the
  environment and a role name should work.
- **NF5** — No multi-account key rotation, ever. Rotating *across* providers
  within published limits is legitimate; rotating accounts to evade limits is
  ToS abuse and gets users banned.
- **NF6** — Credentials never leave the process, never enter logs, never enter
  the recording used for replay.

---

## 5. Scope

### In scope (v1)

Roles + requirements · capability preflight · rate and spend budgets · outcome
memory and bandit selection · verdict failover · registry sync · override ·
exact cache · OpenAI-compatible + Anthropic + Gemini + Ollama adapters ·
record/replay · labpilot adapter.

### Deferred (v2+)

OpenAI-compatible **proxy server** (§8.1) · semantic cache · streaming ·
multi-turn tool loops · conversation memory beyond context-fitting · a hosted
dashboard.

### Out of scope

- Being a marketplace. No billing relationship, no key brokering.
- Prompt engineering, evaluation harnesses, agent frameworks.
- Guaranteeing a model is *correct* — only that it was capable, affordable, and
  has been working.

---

## 6. High-level design

```
        caller: role="codegen", requires={structured_output, ctx>=64k}
                                   │
   ┌───────────────────────────────┴──────────────────────────────┐
   │                    ROUTING (pure function)                    │
   │                                                               │
   │  registry ──► capability filter ──┐                           │
   │  policy   ──► entitlement filter ─┼──► candidates             │
   │  ledger   ──► budget filter ──────┘         │                 │
   │                                             ▼                 │
   │                     rank: outcome posterior × cost × latency  │
   │                     (Thompson sampling; ε explores)           │
   └───────────────────────────────┬──────────────────────────────┘
                                   ▼
                cache lookup ──hit──► return (no spend)
                                   │miss
                                   ▼
                adapter.complete() ──► verdict from caller
                                   │
        ┌──────────────────────────┼───────────────────────┐
        ▼                          ▼                       ▼
   spend ledger            outcome posterior         limit learner
   (tokens, cost)          (role, model) → Beta      (429 / headers)
```

Everything below the pure function is I/O and is mockable. Everything inside it
is a decision that can be unit-tested without a network.

---

## 7. Components and responsibility boundaries

| Component | Owns | Must not |
|---|---|---|
| `registry` | model facts, sync, merge precedence, vendored snapshot | know about the caller or spend |
| `policy` | roles, requirements, entitlement, data policy | know about specific models |
| `ledger` | rate windows, spend, learned limits | make decisions |
| `memory` | per-(role, model) verdict posteriors | perform I/O during selection |
| `select()` | **the decision** — pure | construct clients, sleep, or call out |
| `adapters` | HTTP shapes per provider family | choose a provider |
| `gateway` | executes: cache, call, meter, record, verdict plumbing | contain routing rules |

The boundary that matters most: **`select()` decides, `gateway` executes.** A
router that constructs clients cannot be tested without a network, which is why
routing logic is untested nearly everywhere it exists.

---

## 8. Design choices

### 8.1 Python for v1, with the policy core kept portable

The language question, honestly compared:

| | Python | TypeScript | Rust |
|---|---|---|---|
| **Reach for this problem** | agent/ML frameworks, notebooks, research | AI *product* layer — Vercel AI SDK, LangChain.js, Mastra; edge deploy | infra teams; a proxy binary anyone can run |
| **Crowding** | high — LiteLLM owns Python routing mindshare | **low** for routing specifically | low |
| **Time to v1** | ~1× | ~1.2× | ~3–5× |
| **Contributor pool for OSS** | largest | large | smallest; async Rust deters drive-by PRs |
| **Fits a hot-path proxy** | poor (uvicorn + workers) | fine (edge/Workers) | **excellent** (cf. Pingora, linkerd2-proxy) |
| **Can labpilot consume it in-process** | **yes** | no — needs the proxy | yes, via PyO3 |
| **Single-binary distribution** | no | no | **yes** |

**Recommendation: Python.** Three reasons, in order of weight:

1. **The differentiator needs a workload.** Outcome memory is worthless without
   real verdicts, and labpilot is the only thing producing them in month one.
   A TypeScript or Rust v1 leaves labpilot's M7 blocked and the routing policy
   unvalidated — which would make this OpenRouter with more steps, which is
   precisely what should not be built.
2. **The policy is the product; the transport is not.** Porting a proven pure
   function is a weekend. Inventing the policy in Rust is a quarter.
3. Python is where the frameworks that would adopt it live.

**But** if maximum adoption were the only goal, TypeScript is the better bet —
the people who feel "which model do I use" hardest are product builders without
an ML team, and they are on npm. And Rust is the right answer for the *proxy*.

So NF1 exists to keep both doors open: `select()` is pure, has no dependencies
beyond dataclasses, and is specified precisely enough (§9.2) to port. The v2
proxy should be Rust; a thin TS client can call it.

### 8.2 Outcome memory is a constrained bandit, not a leaderboard

Selection among capable, affordable candidates is a **contextual bandit**:
context = role, arms = models, reward = verdict, constraint = cost and budget.
Naming it that has consequences worth taking:

- Per (role, model), keep a Beta posterior over success. A model with 2
  successes must not outrank one with 200 — the prior handles it.
- **Sample, don't argmax** (Thompson sampling). Pure exploitation pins you to
  whatever won first and never notices that a cheaper model got better.
- A model's prior is seeded from registry capability tier and benchmarks, so a
  newly released model starts plausible rather than at zero.
- A **new release or version bump resets that model's posterior** — the ID is
  the same, the weights are not.

*Rejected:* a global quality leaderboard. It answers "which model is best in
general", which is the question the caller already can't act on. The whole point
is "best for **this** work, on **my** prompts".

### 8.3 A 200 is not a success

The verdict, not the status code, decides. The caller reports one of:

```
OK · SCHEMA_INVALID · REFUSED · TRUNCATED · TOO_SLOW · DOWNSTREAM_FAILED
```

`SCHEMA_INVALID` and `REFUSED` trigger failover to the next candidate and
penalise the posterior. `DOWNSTREAM_FAILED` — "the code it wrote didn't run" —
penalises without failover, because the call itself was fine.

This is the labpilot lesson generalised: the observed failure was HTTP 200 with
English prose, and no system anywhere treats that as a routing signal.

*Consequence to accept:* the caller must report. An unreported call is recorded
as `UNKNOWN` and contributes nothing to the posterior — never silently as
success. A helper (`with router.call(...) as c:`) makes the common path report
`OK` on clean exit and the failure verdict on a raised parse error.

### 8.4 Budgets are learned, not configured

Two ledgers, kept separate because they exhaust differently:

- **Rate** — rpm/rpd/tpm windows, per (provider, account). Seeded from config,
  then **corrected by observation**: `Retry-After`, `X-RateLimit-Remaining`,
  `X-RateLimit-Reset`, and 429 ceilings. Observed always beats configured.
- **Spend** — currency, per role and global, per day/month. Computed from the
  registry's per-token costs and the response's usage. Free tier is not a
  special case: it is `cost == 0`.

Free-tier support therefore falls out of the general mechanism rather than being
the mechanism, which is the correction that started this redesign.

*Consequence to accept:* some endpoints do not report usage. Record `0` tokens
rather than estimating. An invented number in a quota ledger produces a wait no
one can explain.

### 8.5 Waiting must be bounded

Per role, `on_exhaustion: wait | degrade | fail` with `max_wait_seconds`.

Waiting is right when degrading is worse than being slow — a weak model that
implements a technique badly makes the system record *"technique X did not
help"*, a false negative indistinguishable from a real one. But an unbounded
wait in an unattended process is indistinguishable from a hang, and the
operator's only recourse is killing a run whose state they can't inspect.

### 8.6 Override is four-level and always recorded

```python
router.complete(role="codegen", model="groq/llama-3.3-70b")   # per call
with router.pinned("openai/gpt-5"): ...                        # scope
FITROUTE_FORCE_MODEL=ollama/qwen2.5-coder:14b                  # env
roles: {codegen: {pin: anthropic/claude-opus-5}}               # config
```

Precedence is call > scope > env > config. Pinned calls are **excluded from
exploration but still recorded**, flagged `pinned=true`. Silently folding pinned
results into the posterior would corrupt every comparison the router later makes
— you would be learning from a sample you chose, not one it chose.

### 8.7 Registry sync is background, merged, and offline-safe

Sources merge with precedence `local override > models.dev > openrouter >
litellm`, because models.dev alone carries the provider `env`/`api` fields that
make a provider *routable* rather than merely known.

Rules: sync never blocks a call; a vendored snapshot ships in the package so the
first run works with no network; each record carries `source` and `synced_at`;
staleness beyond a threshold warns rather than fails; a model past
`expiration_date` is dropped from candidates with a log line naming it.

*Rejected:* scraping provider docs for quota numbers. Fragile, adversarial to
the providers, and unnecessary once limits are learned (§8.4).

### 8.8 Local models are first-class citizens

Ollama, vLLM and llama.cpp sit in the same candidate pool as hosted providers,
with `cost = 0` and `tier = local`. This is what makes data-policy routing real:
a workload marked sensitive routes to local hardware automatically instead of
requiring a separate code path.

It is also the clearest structural difference from a hosted gateway: a
marketplace cannot route to the machine you are sitting at.

### 8.9 Conversation memory belongs here for one specific reason

Generic chat memory does not belong in a router. **Context-fitting** does: the
router is the only component that knows which model was chosen, and therefore
the only one that knows the context window the history must fit into.

So v1's `Thread` does exactly that — hold turns, and fit them to the selected
model's window (drop, or summarise via the `summarize` role). Anything more —
retrieval, persistence, entity memory — is a different library's job.

---

## 9. Low-level design

### 9.1 Caller API

```python
from fitroute import Router, Verdict

router = Router.from_env()          # NF4: keys from env, defaults for roles

# One-shot, verdict inferred from parsing
plan = router.complete(
    role="reasoning",
    system=SYSTEM, user=prompt,
    requires={"structured_output"},
    schema=PlanModel,               # parse failure ⇒ SCHEMA_INVALID ⇒ failover
)

# Explicit verdict, for outcomes only the caller can judge
with router.call(role="codegen", requires={"structured_output"}) as c:
    code = c.complete(system=SYSTEM, user=prompt)
    if not run_generated_code(code):
        c.verdict(Verdict.DOWNSTREAM_FAILED, "train.py raised")
```

`c.served` carries `provider`, `model`, `tier`, `cost`, `tokens`, `latency_ms`,
`cache_hit`, `degraded`, `pinned` — everything needed to attribute a result to
the model that produced it.

### 9.2 The pure decision (NF1)

```python
@dataclass(frozen=True)
class Candidate:
    provider: str; model: str; tier: str
    cost_per_1k_in: float; cost_per_1k_out: float
    caps: frozenset[str]; context: int
    posterior: tuple[float, float]      # Beta(α, β) for this role
    available_in: float                 # 0.0 = now, else seconds until

def select(
    role: RoleSpec,
    candidates: Sequence[Candidate],
    *,
    spend_remaining: float,
    rng: Random,                        # injected — determinism for tests
) -> Decision: ...
```

No clock, no network, no filesystem. `available_in` and `spend_remaining` are
computed by the ledger and passed in. Every routing property — entitlement, data
policy, capability, budget, wait-vs-degrade, exploration — is then a table-driven
unit test with a seeded RNG.

### 9.3 Registry record

```python
@dataclass(frozen=True)
class ModelRecord:
    id: str                     # "groq/llama-3.3-70b"
    provider: str
    base_url: str               # models.dev `api`
    api_key_env: str            # models.dev `env`
    context: int; max_output: int
    caps: frozenset[str]        # structured_output, tool_call, vision, …
    cost_in: float; cost_out: float; cost_cache_read: float
    tier: str                   # paid | free | local
    trains_on_input: bool | None
    released: date | None; expires: date | None
    source: str; synced_at: datetime
```

### 9.4 Learned-limit update

```
on response:
    headers → rpm/rpd/tpm hints → store as observed_limit(provider, account)
on 429:
    Retry-After → cooldown
    no Retry-After → exponential backoff, and record the request count in the
                     window as an observed ceiling
```

Keyed on `(provider, sha256(api_key)[:12])` — limits are per account, and two
keys for the same provider must not share a ledger. The hash never leaves the
local store (NF6).

### 9.5 Record & replay

Every call appends `{request_hash, role, served, response, verdict, ts}` to a
recording, with credentials and any header matching a secret pattern stripped.
`Router.replay(path)` serves from it and makes no network calls — which gives
free deterministic CI for anything built on the router, and reproducible agent
runs, which is the thing research users ask for most and get least.

---

## 10. Testing strategy

| Level | Covers | Network |
|---|---|---|
| Property | `select()` — capability, budget, policy, exploration, wait/degrade, seeded RNG | no |
| Contract | each adapter's request shape, incl. `json_mode` per family | no |
| Golden | registry merge from frozen source fixtures | no |
| Simulation | 10k synthetic calls; assert the bandit converges on the better model and explores enough to notice a change | no |
| Integration | one real call per configured provider | opt-in |
| End-to-end | labpilot campaign — §11 | yes |

Named tests that map to a measured failure rather than an imagined one:

1. `test_prose_response_is_not_success` — 200 + prose ⇒ `SCHEMA_INVALID` ⇒
   failover ⇒ posterior penalised. *The 3/3 labpilot failure.*
2. `test_model_without_structured_output_is_not_a_candidate` — preflight, F2.
3. `test_every_call_is_metered` — N calls ⇒ N ledger rows. *The one-cache-row
   failure.*
4. `test_cache_hit_does_not_spend` — repeat ⇒ 1 ledger row.
5. `test_pinned_calls_do_not_train_the_posterior` — §8.6.
6. `test_learned_limit_overrides_configured` — a 429 at 8 rpm when config said
   10 lowers the effective limit.
7. `test_sync_failure_serves_stale_and_warns` — NF2.
8. `test_recording_contains_no_credentials` — NF6, asserted by pattern scan.

### 10.1 Simulation is the one that earns the design

The bandit claim — "it learns the better model and keeps noticing changes" — is
untestable on real traffic in reasonable time. So: a simulator with known model
success rates, a mid-run capability shift, and assertions on regret and on time
to re-converge. If the simulation cannot show it beating "always use the
strongest affordable model", the mechanism is not worth its complexity and
should be cut to a static ranking.

---

## 11. Evaluation

### 11.1 The baseline that must be beaten

Not OpenRouter. The honest baseline is **"always use the strongest model you can
afford"** — which is what a careful engineer does by hand, and it is a strong
baseline. If routing cannot beat it, this is complexity for its own sake.

| Metric | Baseline | Target |
|---|---|---|
| Task success rate (verdict `OK`) | measure it | ≥ baseline |
| Cost per successful result | measure it | **materially lower** |
| Calls wasted on incapable models | measure it | **0** (F2 is a hard filter) |
| Time to recover from a provider outage | manual | automatic, < 1 call |

The headline claim is deliberately **cost per success**, not success rate. Equal
quality for less money is a real win and an honest one; claiming better quality
than a frontier model would not survive contact with users.

### 11.2 First workload: labpilot on rogii

labpilot is the first consumer and the forcing function. Protocol:

1. Baseline: rogii's honest validation MSE is **194.8** on 154 held-out wells,
   against a naive anchor of 226.3.
2. Run `research conduct` with the router live and ≥1 strong provider.
3. Accept only if a real codegen call produces a `train.py` that runs and writes
   metrics, and the served model is recorded per experiment.
4. Report cost per successful codegen call, and the verdict mix per model.

**Exclusion rule:** campaigns where the backlog gate suppressed all agent calls
must be excluded, not counted as clean. Counting them once turned "0 of 3 fell
back" into a false "6 of 9 clean".

### 11.3 What would falsify the design

- The simulation shows no regret advantage over static ranking → cut the bandit,
  keep capability preflight and budgets. Those two stand on their own.
- Callers don't report verdicts in practice → outcome memory starves. Watch the
  `UNKNOWN` rate in labpilot first; if it is high in the codebase that *wants*
  this feature, it will be higher everywhere else.
- Capability flags in the registries turn out to be wrong often enough that
  preflight rejects working models → preflight needs an observed-capability
  override, learned the same way limits are.

---

## 12. Observability

Per call at DEBUG, aggregated at INFO: `role`, `provider`, `model`, `tier`,
`cache_hit`, `pinned`, `degraded`, `wait_ms`, `tokens`, `cost`, `verdict`.

`router doctor`:

```
Registry: models.dev + openrouter, synced 2h ago, 412 models
Budget:   $0.83 / $5.00 today   ·  groq 41/1000 rpd  ·  ollama unlimited

Role       Chosen             Why                            Success  Cost/1k
codegen    anthropic/opus-5   only capable · structured_out   94% (n=68)  $0.021
reasoning  groq/llama-3.3-70b posterior 0.91 · free tier      91% (n=210) $0.000
summarize  ollama/qwen2.5     local · policy: no external     88% (n=902) $0.000

⚠ codegen has one capable provider — an outage stalls it. Add a fallback.
```

That last line is the kind of thing a router uniquely knows and nobody surfaces.
OpenTelemetry spans for anyone with a collector.

---

## 13. Production readiness

### 13.1 Build order

| Phase | Ships | Verified by |
|---|---|---|
| 1 | registry + adapters + `select()` (capability, entitlement, budget) | property + golden tests |
| 2 | gateway: cache, meter, verdicts, failover, learned limits | simulation + a local Ollama run |
| 3 | outcome memory + bandit; `router doctor`; override | simulation §10.1 |
| 4 | labpilot adapter — **the forcing function** | rogii, §11.2 |
| 5 | record/replay; conversation `Thread` | replay CI |
| v2 | Rust proxy + TS client | adoption, not correctness |

Phases 1–3 are inert for labpilot. Phase 4 is where behaviour changes, and it is
what proves phases 1–3 were real rather than merely tested.

### 13.2 The prerequisite only the user can supply

End-to-end verification needs at least one strong-provider API key. Creating
accounts is the user's to do. Groq, GitHub Models, OpenRouter, Cerebras and
Mistral all fit the OpenAI-compatible adapter; any paid key works too.

### 13.3 Risks

- **Adoption without differentiation.** If v1 ships capability preflight and
  budgets but not outcome memory, it is LiteLLM with fewer providers. The bandit
  is the reason to exist; do not ship the wrapper and call it done.
- **Registry rot.** Free tiers deprecate models constantly. Mitigated by sync,
  `expiration_date` handling, and never hardcoding a model name anywhere.
- **Provider ToS.** Rotating across providers within published limits is fine.
  Multi-account rotation is not, will not be supported, and should be refused as
  a feature request (NF5).
- **The wait becomes the hang.** Bounded waits, logged with a reason, total wait
  in the summary (§8.5).
- **Trust.** A router sits between users and their API keys. NF6 is
  non-negotiable, and the recording scrubber needs a test, not a code review.

### 13.4 Licensing and governance

Apache-2.0 — permissive, with the patent grant that makes companies comfortable
adopting infrastructure. Registry data files carry their upstream sources'
attribution. A `CONTRIBUTING.md` that states the NF5 position up front, because
"add multi-account rotation" will be filed within a month.

### 13.5 Suggested features, ranked by differentiation

Everything above is v1 or v2. Beyond it, in the order that would matter:

1. **Budget forecasting** — "at this rate you exhaust codegen's daily budget in
   40 minutes" before it happens, not after.
2. **Shadow routing** — send 5% of a role's calls to a candidate model, compare
   verdicts, auto-promote when it wins. This is how the registry's rankings stop
   being assertions.
3. **Prompt-cache-aware routing** — prefer a provider whose own prompt cache
   would hit; `cache_read` cost is already in the registry.
4. **Latency SLO per role** — "summarize must return in < 2s" as a filter.
5. **Cost attribution by tag** — per user, per feature, per customer. The first
   thing anyone running this in production asks for.
6. **Degradation ladders** — an explicit ordered fallback per role, so degrading
   is a declared path rather than whatever ranked next.
7. **Multi-region / residency routing** — EU-only endpoints as a policy filter.
8. **A `fitroute.eval` harness** — replay a recorded workload against a
   candidate model to get its posterior *before* routing traffic to it.

### 13.6 Name

`fitroute` is a placeholder chosen to keep this document readable. Others worth
considering: `aptly`, `modelfit`, `tokenwise`, `rightsize`. The name should say
*fit for the work* rather than *cheap*, since budget is a constraint here and
not the value proposition.

---

## 14. Relationship to labpilot

labpilot's [M10](../research-os/autonomy-roadmap/04-llm-tiering.md) becomes
**"adopt the router"** rather than "build routing". Its own design
([design/04-llm-tiering.md](../research-os/autonomy-roadmap/design/04-llm-tiering.md))
keeps the parts that are labpilot's: collapsing three internal routers into one,
binding a role to each micro agent, and the rogii exit criterion. The routing
internals move here.

The existing `llm/catalog.py`, `llm/budget.py` and `llm/router.py::select_route`
are the seed of phase 1 — entitlement, plans, data policy and the rate ledger
are already written and tested. What they lack is everything in §8.2–8.4.
