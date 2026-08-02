# M10 — LLM tiering and free-tier routing

**Status:** decision layer built and tested, **not wired into the live path** ·
**Blocks:** M7 in practice

---

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
