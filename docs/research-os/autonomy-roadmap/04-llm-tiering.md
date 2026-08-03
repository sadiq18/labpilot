# M10 — LLM tiering and free-tier routing

**Status:** decision layer built and tested, **not wired into the live path** ·
**Blocks:** M7 in practice

---

## Design principle

> **Model capability is a product tier, not an architectural constraint.**

An earlier draft of this roadmap was shaped by the machine it was validated on —
a 14B local model — and quietly let that become the design's ceiling. That is
backwards. A paying customer supplies a frontier model; the free tier is a
**development and hobby mode with known limits**, not the substrate the
architecture is built for.

This is a research OS. A weak component does not merely run slower — it produces
*wrong research*, and wrongness propagates into beliefs, claims and memory where
it is expensive to remove. No component may be left weak because the current
development environment is.

Concretely, this changes two things:

- The system assumes a **competent model is reachable** and degrades explicitly
  when one is not (`research doctor` says so; `codegen` waits rather than
  downgrading). It does not assume weakness and design around it.
- **M10 lands before M7.** Its purpose is not cost control; it is making the
  reasoning substrate trustworthy so everything downstream can be trusted.

## Purpose

Two problems at once:

1. **The substrate cannot deliver the reasoning the architecture assumes.**
   `qwen2.5-coder:14b` returned English prose where JSON was required, and
   produced no usable training code at all. Conductor policy, hypothesis
   generation and codegen all need frontier-grade reasoning.
2. **A pre-revenue startup cannot fund frontier tokens**, but several providers
   offer permanent free tiers.

## Goal

Route each *class of work* to a model capable of it, within the user's
entitlement and published rate limits — free tiers today, paid models for
paying customers later, without a rewrite.

## Design — three concerns kept apart

Conflating these is what makes a router impossible to extend:

| Concern | Owner | Change frequency |
|---|---|---|
| **Role requirement** — what the task needs | the code doing the work | rarely |
| **Provider catalog** — models, limits, cost, data policy | YAML | weekly (free tiers churn) |
| **Entitlement** — what this user's plan permits | the customer's plan | per customer |

Routing is their intersection. Adding a paid tier is a config change plus a plan
name.

```yaml
plan: free            # free | pro | enterprise
allow_training_on_inputs: true
providers:
  - {name: github-models, tier: free, strong: true,  rpm: 10, models: {codegen: gpt-4o}}
  - {name: groq,          tier: free, strong: false, rpm: 30, models: {summarize: llama-3.3-70b}}
  - {name: ollama,        tier: local, strong: false}
  - {name: openai,        tier: paid, strong: true,  rpm: 500}   # inert until plan=pro
roles:
  codegen:   {requires_strong: true,  on_exhaustion: wait}
  reasoning: {requires_strong: true,  on_exhaustion: wait}
  summarize: {requires_strong: false, on_exhaustion: degrade}
```

### Two decisions worth preserving

**Enterprise excludes free tiers by default.** Not a flag someone must remember
— `allowed_tiers("enterprise")` simply does not contain `free`, because free
tiers generally reserve the right to train on submitted content. An unknown plan
name falls back to the most restrictive set, so a typo fails closed.
`allow_training_on_inputs: false` overrides plan independently, for a workspace
holding proprietary data on any tier.

**Codegen waits; it never degrades.** If a weak model implements a technique
badly, the system records *"technique X did not help"* — a false negative
indistinguishable from a real one, which then poisons hypothesis generation
permanently. Waiting costs minutes. A corrupted research memory costs the
product. Summarisation has no such property and degrades freely; every
degradation is stamped on the decision so results stay attributable.

## Workload fit

Per campaign, measured:

| Role | Calls | Reasoning |
|---|---|---|
| Conductor policy | ~25 | strong |
| Hypothesis generation | ~5 | strong |
| Codegen | ~10 | strongest |
| Analyzer / kernel summarisation | ~80 | weak — local is fine |

Only ~40 calls need a frontier model. At 10 RPM that is four minutes of quota.
**RPM is not the binding constraint** — a campaign is asynchronous and absorbs
pacing. Daily *tokens* bind first.

## Built (commit `6d4e930`) — decision layer only

> `select_route` has **no call sites under `src/`**. Production LLM resolution
> still goes through `resolve_route`, so the guarantees below hold for the
> function, not yet for the running system. Treat M10 as unfinished until the
> remaining items land.

- `llm/catalog.py` — `ProviderSpec`, `RoleSpec`, `RoutingConfig`, plan
  entitlement, `eligible_providers`
- `llm/budget.py` — persistent SQLite ledger: rpm/rpd/tpm rolling windows plus
  `Retry-After` cooldowns. On disk because campaigns span processes and a fresh
  run must not forget the day's spend.
- `llm/router.py::select_route` — entitlement ∩ data policy ∩ budget, returning
  either a provider or `wait_seconds`
- 15 tests covering entitlement, data policy, budget exhaustion, and
  wait-vs-degrade

## Remaining

1. **OpenAI-compatible adapter** honouring `base_url` + `api_key_env` — one
   client covers Groq, GitHub Models, OpenRouter, Mistral, Cerebras.
2. **Call sites pass a role** (`reasoning` / `codegen` / `summarize`) instead of
   a provider name.
3. **Pace on `wait_seconds`** — the decision already returns it.
4. **Record the served model on each experiment** — without this, a failed
   hypothesis cannot be attributed to the idea vs the writer.
5. **Trim `prior_train[:120_000]`** in `code_engineering/capability.py`. At ~4
   chars/token that is ~30k tokens per codegen call; ten calls eats a daily
   allowance. Do this first — it is ten minutes and it is the difference between
   the free tier lasting a day or an hour.

## Exit criteria

M10 has a mutual-dependency problem: its real purpose is "codegen gets a model
that can write training code", and the only way to verify that is *to generate
training code* — which is M7. Shipping M10 on unit tests alone would repeat the
mistake the review already caught once, where `select_route` was tested but
unwired and described as done.

So M10 is finished when a **thin M7 slice proves it end to end**:

1. `select_route` is on the live path — no call site resolves a provider by name
   for `codegen`, `reasoning` or `summarize`.
2. `research doctor` reports the resolved provider per role, and fails loudly
   when a role has no capable provider.
3. **A real codegen call produces a working `train.py`** for one technique on a
   reference dataset: it runs, writes metrics, and the technique is visible in
   the generated source. This is the forcing function M10 otherwise lacks.
4. The served model is recorded on the experiment record, so a later failure is
   attributable to the idea rather than the writer.
5. Budget exhaustion on `codegen` produces a wait, not a downgrade — observed,
   not just unit-tested.

Criterion 3 is the one that matters. Everything else can pass while the system
still cannot write code.

## Traps

- **Do not rotate multiple accounts per provider.** Rotating *across* providers
  within published limits is legitimate; multi-account key rotation to evade
  limits is ToS abuse and will kill the accounts at the worst moment.
- **Config-driven model names, always.** Free tiers deprecate models constantly;
  anything hardcoded will break unattended.
- **Cache keys must include the model.** Same prompt, different provider,
  different output — reproducing a stored experiment requires pinning what
  produced it.
- **Treat the free tier as a bridge, not the architecture.** It buys runway to
  prove M7. Once a technique demonstrably moves a score, that result is what
  justifies spending on tokens.
