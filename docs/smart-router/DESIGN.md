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

### 2.1a Prior art — a correction (2026-08-05)

An earlier draft of this document claimed nobody routes on whether the output
worked. **That is wrong**, and the error matters enough to fix in place rather
than quietly amend.

| Prior art | What it does | Overlaps |
|---|---|---|
| [RouteLLM](https://arxiv.org/abs/2406.18665) (LMSYS, peer-reviewed) | strong/weak routing from preference data — reported 85% cost saving at 95% of GPT-4 quality | the cost-quality claim in §11.1 |
| [BaRP](https://arxiv.org/pdf/2510.07429) | **routing learned from bandit feedback**, trained under the same partial-feedback restriction as deployment | **§8.2 almost exactly** |
| [Not Diamond](https://github.com/Not-Diamond/awesome-ai-model-routing) | meta-model predicting the best LLM per query | per-prompt selection |
| [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (ulab-uiuc) | open-source routing library — RouterDC, AutoMix, Router-R1 | the OSS-library position |
| [xRouter](https://arxiv.org/pdf/2510.08439) | cost-aware orchestration via RL | budget-constrained selection |
| Martian | quality prediction before inference; **pivoted away from routing** in 2026 | — a market signal worth reading |

So the bandit is **not** novel. What survives scrutiny is narrower, and stating
it honestly is worth more than a bigger claim:

1. **Verdict source.** The prior art scores the *response* — preference models,
   judge models, learned quality predictors. This scores **application-defined
   outcomes**: the generated code ran, the metric improved, the schema parsed.
   That signal is free, exact, and unavailable to a general router because only
   the application knows it.
2. **Capability preflight as a hard filter** (F2). Not present in any of the
   above. It is unglamorous and it fixes a *measured* failure.
3. **Quota discovery and spend caps per role** (§8.4, §8.15). The routers above
   optimise a cost-quality trade-off; they assume you can call anything.
4. **Local models competing on merit** (§8.10), and record/replay (F11).

Items 2–4 are plumbing, not research. They are real, and they are the reason to
build this — but they will not carry a claim of novelty, and §11 should be read
with that in mind. **Read BaRP before writing any bandit code**: it either saves
months or shows the idea is taken.

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
| [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) | actual free-tier numbers across 13 providers | no JSON artifact — output is a generated README; **no license file** |

**Correction (2026-08-05).** An earlier draft of this section called the last
row "markdown, hand-maintained". That is wrong, and the truth is more useful:
its README is **generated** by `src/pull_available_models.py`, which discovers
limits by sending a one-token probe and reading the response headers —

```python
# their fetch_groq / get_groq_limits_for_model, condensed
r = requests.post(".../chat/completions",
                  json={"model": m, "messages": [...], "max_tokens": 1, "stream": True})
rpd = int(r.headers["x-ratelimit-limit-requests"])
tpm = int(r.headers["x-ratelimit-limit-tokens"])
```

29k stars, updated the day this was written, 13 provider fetchers. It is
independent confirmation of the mechanism in §8.4 — and it shows that mechanism
is stronger than assumed: limits are discoverable **proactively**, before a
single 429, for the price of one token.

**Conclusions that shape the design.** Sync is not impossible — it is layered,
and only the last row resists it:

| What | Syncable? | How |
|---|---|---|
| Model facts — cost, context, `structured_output`, base_url, env var | **yes** | models.dev · OpenRouter · LiteLLM |
| New models and retirements | **yes** | `release_date` · `created` · `expiration_date` |
| Models *your key* can reach | **yes** | the provider's own `/v1/models` |
| Your actual quota | **yes** | one-token probe, `x-ratelimit-limit-*` headers |
| **A provider you have never signed up for** | **no** | curation only (§8.15 layer 4) |

The unsyncable row is also the slowest-moving: models arrive weekly, providers a
few times a year. A community catalog file covers it, which is why §8.15
layer 4 is data rather than code.

Note on reuse: that repository publishes **no license**, so its data cannot be
vendored. Consuming it means reimplementing the probe (trivial — it is the
snippet above) or upstreaming a JSON output and asking for a licence.

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
- **F12** — **Local competes on merit.** A local model is ranked by the same
  posterior and cost arithmetic as a hosted one. There is no tier preference
  that makes it a fallback (§8.10).
- **F13** — **Bring your own inference.** Any OpenAI-compatible endpoint —
  self-hosted vLLM, TGI, SGLang, llama.cpp, an internal gateway — is registrable
  with declared capabilities, and those declarations are **probe-verified**
  rather than trusted (§8.11).
- **F14** — **Streaming**, with failover semantics stated rather than discovered
  (§8.12).
- **F15** — **Cost accounting in currency**, attributable by tag (user, feature,
  customer), with unmetered calls counted separately from zero-cost ones (§8.13).
- **F16** — **`role="auto"`** derives requirements from the call and biases
  toward the stronger role on ambiguity (§8.14).
- **F17** — **Discovery**: new models are found by sync, reachable models by the
  provider's own `/v1/models`, quotas by a one-token header probe, and new
  *providers* by a community catalog. A newly discovered capable model is
  **auto-enrolled into exploration**, not merely listed (§8.15).
- **F18** — **Retirement**: a model past `expiration_date` or absent from sync
  is withdrawn from candidates with a warning naming it and its current traffic
  share — never left to 404 mid-run (§8.15).

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
**custom endpoints with capability probing** (§8.11) · **streaming** (§8.12) ·
**cost attribution by tag** (§8.13) · **`role="auto"`** (§8.14) ·
**discovery: sync diff, `/v1/models`, limit probe, auto-enrolment, retirement**
(§8.15) · record/replay · labpilot adapter.

### Deferred (v2+)

OpenAI-compatible **proxy server** (§8.1) · semantic cache · multi-turn tool
loops · conversation memory beyond context-fitting · a hosted dashboard.

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

- **Rate** — rpm/rpd/tpm windows, per (provider, account). Discovered by a
  one-token probe at key registration (§2.3), then **corrected continuously by
  observation**: `Retry-After`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`,
  and 429 ceilings. Observed always beats configured, and a probe result beats
  documentation.
- **Spend** — currency, per role and global, per day/month. Computed from the
  registry's per-token costs and the response's usage. Free tier is not a
  special case: it is `cost == 0`.

Free-tier support therefore falls out of the general mechanism rather than being
the mechanism, which is the correction that started this redesign.

*Consequence to accept:* some endpoints do not report usage. Never estimate — an
invented number in a quota ledger produces a wait no one can explain. But
**unknown is not zero**: record `tokens = None`, distinct from a genuine zero,
so a provider that reports nothing shows up as an unmetered call rather than
silently understating the day's spend (§8.13).

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

See §8.10 — being *in* the pool is not enough, and the seed code inherited from
labpilot actively prevents local from ever winning.

### 8.9 Conversation memory belongs here for one specific reason

Generic chat memory does not belong in a router. **Context-fitting** does: the
router is the only component that knows which model was chosen, and therefore
the only one that knows the context window the history must fit into.

So v1's `Thread` does exactly that — hold turns, and fit them to the selected
model's window (drop, or summarise via the `summarize` role). Anything more —
retrieval, persistence, entity memory — is a different library's job.

### 8.10 Delete the tier preference — local wins when local is better

The seed code carried over from labpilot sorts candidates by tier:

```python
_TIER_RANK = {"paid": 0, "free": 1, "local": 2}   # llm/catalog.py:25
```

That single line makes local a permanent last resort. It is defensible in a
cost-control router and wrong in this one: a 14B model that answers a
summarisation prompt correctly 88% of the time for **zero cost and no network**
should beat a frontier model that answers it 91% of the time for real money —
and under a tier sort it never can.

**Tier stops being a ranking input and becomes only a filter input** (via
entitlement and data policy, §8.7 unchanged). Ranking is posterior × cost ×
latency, and `cost = 0` is a genuine advantage rather than a consolation.

Two guards this needs, or it becomes a different kind of wrong:

- **Latency belongs in the score.** Local inference can be 20× slower. A role
  with a latency SLO filters it out; a role without one accepts the trade
  knowingly. Without this, "local is free" quietly makes everything slow.
- **A cold local model has no posterior.** Seed it from the registry's
  capability tier like any other model (§8.2), so a 3B model is not tried for
  codegen merely because it is free. Capability preflight (F2) catches the worst
  of it first.

*Consequence to accept:* on a machine with a good local model, most traffic goes
local and the hosted providers' posteriors stay thin. Exploration (§8.2) is what
stops that from becoming a one-way door.

### 8.11 Bring your own inference — declared, then probed

Registering a private endpoint is the same shape as any provider, except the
registry knows nothing about it, so the operator declares its facts:

```yaml
providers:
  - id: internal/llama-70b
    base_url: https://inference.corp.internal/v1
    api_key_env: CORP_INFERENCE_KEY
    tier: local              # cost 0, data never leaves — both true here
    caps: [structured_output, tool_call]
    context: 131072
```

**Declared capabilities are not trusted.** `router probe internal/llama-70b`
runs a small suite — does `response_format` actually constrain output, do tool
calls come back well-formed, does the advertised context length hold at the top
end, what is p50 latency — and writes an observed record that overrides the
declaration, with a diff when they disagree.

This matters more for private endpoints than for anything else: a
self-hosted server behind an OpenAI-compatible facade very often implements a
*subset*, and the failure mode is the one this router exists to prevent — a
prompt requiring structured output sent to something that will return prose with
a 200.

Probing is also how F2 stays honest for public providers when a registry flag is
wrong (§11.3, third falsifier).

### 8.12 Streaming, with failover semantics stated up front

Streaming is table stakes for any user-facing product, so it ships in v1. But it
is in genuine tension with two core mechanisms, and the tension must be a
decision rather than a surprise:

- **Verdict failover (F5)** — you cannot un-send tokens already delivered.
- **Schema validation** — invalidity is knowable only at the end.

The rule: **first emitted token commits the choice.** Before it, failover works
normally (connection errors, refusals in the first chunk). After it, a late
`SCHEMA_INVALID` still penalises the posterior and is still reported to the
caller, but no retry happens — the caller owns recovery.

Roles may set `stream: never`, which is the right setting for anything whose
output is parsed rather than shown to a human. labpilot's micro agents all set
it: there is no one watching tokens arrive, so the only effect of streaming
there would be to disable failover.

*Rejected:* buffer the full response, validate, then replay it as a stream. It
preserves failover and destroys the only reason to stream.

### 8.13 Cost in currency — with unmetered calls visible

Every call carries an optional `tags={"user": ..., "feature": ...}`, and spend
rolls up by tag, role, model and day. This is the first thing anyone running an
LLM feature in production asks for and rarely gets without a vendor dashboard.

Two honesty constraints, both non-negotiable:

- **Cost is an estimate, labelled as one.** It is registry price × reported
  usage. Registry prices go stale; the number is for decisions, never presented
  as an invoice.
- **Unmetered calls are reported, not zero-filled.** A provider that returns no
  usage produces `$1.34 + 12 unmetered calls`, never `$1.34`. Silently treating
  unknown as zero is how a budget cap gets blown while the dashboard looks calm.

Spend caps per role already exist (§8.4); this adds attribution and reporting on
top of the same ledger, so it costs a schema column and a report command rather
than a new mechanism.

### 8.14 `role="auto"` — derive what you can, guess upward

Zero-config is what makes this usable by people who will never declare roles, so
`auto` ships. But a misclassification routes codegen to a weak model, which is
the exact failure the router exists to prevent. So `auto` is built to be
conservative rather than clever:

1. **Derive, don't guess, wherever possible.** A `schema=` argument *proves*
   structured output is required. Prompt length *measures* the context needed.
   Attached images *determine* the modality. Most of what matters is derivable
   with no classifier at all.
2. **Guess only the strength axis**, from cheap signals — length, imperative
   verbs, presence of code, output-size hints — never an LLM call. Adding a
   classification call to every call would tax the hot path and raise the
   question of which model classifies.
3. **Bias upward on ambiguity.** Guessing `summarize` when the work was codegen
   corrupts a result; guessing `codegen` when the work was summarisation costs
   money. Those are not symmetric, so ties resolve to the stronger role.
4. **Mark it.** `role_inferred=true` on the record, so auto-routed calls can be
   compared against declared ones. If auto's success rate is materially worse,
   that is measurable rather than folklore — and it is the signal that decides
   whether `auto` should ever become the default.

*Rejected:* an LLM-based intent classifier in v1. Latency and cost on every call,
plus a bootstrapping problem — the classifier needs a model, chosen by the
router, whose choice depends on the classification.

### 8.15 Discovery — four layers, because one mechanism cannot cover it

Learned limits (§8.4) answer *how much budget do I have here*. They say nothing
about a provider you have never signed up for, and nothing about a model that
launched this morning. Discovery is a separate problem with four distinct
sub-problems, and conflating them is why routers end up with stale hardcoded
model lists.

**Layer 1 — a new model on a provider I already use.** Solved by sync.
models.dev carries `release_date` / `last_updated`; OpenRouter carries
`created`. A daily sync diffs against the local registry and yields new arrivals.

Knowing is the cheap half. **The router auto-enrols them**: a new model that
passes capability preflight for a role enters that role's bandit with a
registry-seeded prior and receives exploration traffic (§8.2). Within a day
there is a real posterior measured on *your* prompts.

This is the sharpest difference from a marketplace. OpenRouter can tell you a
model exists. It cannot tell you that it is now your best summariser at a tenth
of the cost, because it never ran your work through it and never saw whether the
output parsed.

**Layer 2 — which models my key can actually reach.** Public catalogs list what
exists; a provider's own `/v1/models` lists what *this account* may call, which
differs by tier and by region. Queried on key registration and on a schedule.
Cheap, exact, no inference.

**Layer 3 — what my free quota actually is.** The probe from §2.3: one request,
`max_tokens: 1`, read `x-ratelimit-limit-*` from the response headers. Run on
key registration, per model, then corrected continuously from live traffic
(§8.4). Providers that return no such headers fall back to learn-from-429.

Layers 2 and 3 compose into `router discover`: give it a key, get back the
models that key can reach, their real limits, and their probe-verified
capabilities (§8.11).

**Layer 4 — a provider I have never heard of.** No API answers this, and no
amount of cleverness changes that. Three sources, in increasing order of value:

- **A community catalog file** in the repo — provider, signup URL, tier terms,
  `verified_on` date. PR-able. Deliberately data, not code, so contributing does
  not require understanding the router.
- **Upstream cooperation.** cheahjs/free-llm-api-resources already tracks 13
  providers with the same probe method and has 29k users watching it for
  breakage. The right move is to upstream a JSON output there rather than fork
  its effort — better for them, better here, and it needs a licence conversation
  (§2.3) that is worth having early.
- **Opt-in anonymous telemetry.** Users share observed limits only — provider,
  model, observed rpd/tpm, date. No prompts, no responses, no keys, no tags.
  That produces an empirically verified free-tier map that no documentation
  source has, and it improves as adoption grows.

  Non-negotiable conditions, because a router sits between users and their API
  keys and trust is the whole asset: **off by default**, one flag to enable,
  and `router telemetry --dry-run` prints the exact payload that would be sent.
  Anything less and this becomes the reason people do not adopt it.

**And discovery must produce an action, not a log line.** `router whatsnew`:

```
3 new models since 2026-08-01
  groq/llama-4-scout     free · 1000 rpd · structured_output ✓  → exploring for `summarize`
  openai/gpt-5.2-mini    $0.15/1M · structured_output ✓         → exploring for `codegen`
  xai/grok-4-fast        no structured_output                   → skipped: `codegen` requires it

1 model retiring
  google/gemini-3.0-flash  expires 2026-09-30 · currently 41% of `summarize` traffic
```

That last line is something only a router can know, and nothing surfaces it
today. Retirement is handled deliberately for the same reason: a model that
passes `expiration_date` or disappears from sync is withdrawn from candidates
with a warning naming it, never left to 404 mid-campaign.

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
9. `test_free_local_outranks_costly_hosted_when_posterior_close` — §8.10. The
   tier sort inherited from labpilot fails this today.
10. `test_probe_overrides_a_false_capability_declaration` — an endpoint claiming
    `structured_output` that ignores `response_format` is demoted. §8.11.
11. `test_stream_after_first_token_does_not_failover` — and still records the
    verdict. §8.12.
12. `test_unmetered_calls_are_not_zero_filled` — a usage-less provider shows as
    unmetered, not as free. §8.13.
13. `test_auto_biases_upward_on_ambiguity` — a prompt matching both roles routes
    to the stronger. §8.14.
14. `test_new_capable_model_is_auto_enrolled` — a model appearing in a sync
    fixture receives exploration traffic for a role it qualifies for; one
    lacking a required capability does not. §8.15 layers 1–2.
15. `test_limit_probe_beats_configured_limit` — probe reports 1000 rpd where
    config said 14400; the ledger uses 1000. §8.15 layer 3.
16. `test_expired_model_is_withdrawn_with_a_warning` — and the warning names its
    current traffic share. F18.
17. `test_telemetry_payload_carries_no_prompt_or_key` — asserted by field
    allowlist, not by pattern scan. §8.15 layer 4.

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

> **Where v0.1 is being built (2026-08-05).** Phases 1–2 land **inside
> `src/labpilot/llm/`**, not yet as a separate package. Standing up a repo and a
> release process before anything consumes the code would spend the session on
> packaging rather than on the forcing function.
>
> Extraction stays cheap by rule, not by intention: **`catalog.py`, `budget.py`,
> `adapters.py` and `select()` import nothing from `labpilot`.** A source guard
> test enforces it (`test_router_core_has_no_labpilot_imports`). When the
> open-source package is stood up, extraction is a file move plus a rename —
> which is exactly the property "extract later" usually fails to keep.
>
> Everything in §8.10–§8.15 and §13.5 stays in this document as designed future
> work. v0.1 implements none of it.

| Phase | Ships | Verified by |
|---|---|---|
| 1 | registry + adapters + `select()` (capability, entitlement, budget) | property + golden tests |
| 2 | **labpilot adapter — the forcing function, moved up** | rogii, §11.2 |
| 3 | gateway: cache, meter, verdicts, failover, learned limits; **cost attribution** (§8.13) | simulation + a real campaign |
| 4 | outcome memory + bandit, **no tier preference** (§8.10); `router doctor`; override | simulation §10.1 **on real verdicts** |
| 5 | **discovery + auto-enrolment + retirement** (§8.15); `router discover` / `whatsnew` | a new model appears in a sync fixture and receives exploration traffic |
| 6 | **custom endpoints + `router probe`** (§8.11); **streaming** (§8.12) | probe against a local vLLM |
| 7 | **`role="auto"`** (§8.14); record/replay; conversation `Thread` | auto-vs-declared success rates; replay CI |
| v2 | Rust proxy + TS client; community catalog; opt-in telemetry | adoption, not correctness |

**The forcing function moved from phase 4 to phase 2 deliberately.** The
original order built cache, metering, verdicts and the bandit before anything
real consumed them — which is `select_route`'s mistake at larger scale: tested,
unwired, described as done. Outcome memory in particular cannot be validated
without verdicts, and labpilot is the only source of them.

So phase 2 is the smallest thing that makes labpilot's codegen work through the
router, and every phase after it is justified by something observed in a real
campaign rather than by this document.

Phases 5–7 are ordered last for the same reason. Streaming, probing, discovery
and `auto` all widen the surface, and none of them makes routing better.

### 13.2 The prerequisite only the user can supply

End-to-end verification needs at least one strong-provider API key. Creating
accounts is the user's to do. Groq, GitHub Models, OpenRouter, Cerebras and
Mistral all fit the OpenAI-compatible adapter; any paid key works too.

### 13.3 Risks

- **Adoption without differentiation.** If v1 ships capability preflight and
  budgets but not outcome memory, it is LiteLLM with fewer providers. The bandit
  is the reason to exist; do not ship the wrapper and call it done.
- **Registry rot.** Free tiers deprecate models constantly. Mitigated by sync,
  `expiration_date` handling, retirement (F18), and never hardcoding a model
  name anywhere.
- **Discovery outruns judgement.** Auto-enrolment (§8.15) means unknown models
  get real traffic. Bounded by capability preflight, a small exploration budget,
  and per-role spend caps — and a new model must never be enrolled into a role
  whose `on_exhaustion` is `wait`, where a bad choice is expensive, until it has
  a posterior from a cheaper role.
- **Telemetry kills trust.** Opt-in, off by default, payload inspectable
  (§8.15). If in doubt, ship without it — the community catalog gets most of the
  value at none of the risk.
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
   Promoted in importance by §8.10: once local can win on cost, latency is the
   only thing stopping it from winning everywhere.
5. **Degradation ladders** — an explicit ordered fallback per role, so degrading
   is a declared path rather than whatever ranked next.
6. **Multi-region / residency routing** — EU-only endpoints as a policy filter.
7. **A `fitroute.eval` harness** — replay a recorded workload against a
   candidate model to get its posterior *before* routing traffic to it. Pairs
   naturally with `router probe` (§8.11): probe answers *can it*, eval answers
   *is it any good*.
8. **Local capacity awareness** — queue depth and VRAM on the local box as a
   routing input, so §8.10 does not send twenty concurrent calls to one GPU.

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
