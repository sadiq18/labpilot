# Design — M14: LLM required; retire silent rule-engine fallback

**Plan:** [../09-llm-required.md](../09-llm-required.md) ·
**Status:** design · **Owner:** unassigned · **Build phase:** 0 (phase 1 only)

---

## 1. Background

Twenty micro agents implement `_run_rule_engine`, and `BaseMicroAgent.run()`
(`micro_agents.py:139-170`) catches any LLM or parse failure and quietly uses
the deterministic path instead:

```python
except Exception as exc:  # noqa: BLE001 - soft-fail to deterministic path
    ...
    logger.warning("Micro agent %s LLM path failed (%s); using rule_engine fallback.")
    break
return self._run_rule_engine(context)
```

Observed during validation: `qwen2.5-coder:14b` answered a JSON-only prompt in
English prose, the page analyzer fell back, and the system ran deterministic
rules while presenting as though it were reasoning. The Knowledge Hub found zero
concepts, so no techniques, no beliefs, no hypotheses — and the campaign had
nothing to iterate on.

Nothing above that layer could tell. No error, no metric, and — critically — a
**provenance record that said the opposite of the truth**.

> **Measured since.** Constrained JSON decoding, shipped in this branch, removed
> that cause: fallbacks went from 3 of 3 campaigns to 0 of 2 (§11.1). M14 is
> therefore **not** justified by "the system is currently degraded" — it is not.
> It is justified by §11.2: one of the two fallback paths cannot be observed at
> all, so the current rate is knowable only by luck.

## 2. Problem statement

> Degradation is invisible, and where it is recorded at all, the record is
> inconsistent and in one place wrong.

Three distinct defects, verified:

**2.1 — Provenance lies.** `analyzers/competition.py:359`:

```python
source = "llm" if agent.uses_llm else "rule_engine"
# If LLM was configured but run fell back, BaseMicroAgent still has uses_llm True;
# detect via whether we intended LLM — soft note is enough.
```

`uses_llm` is `llm_client is not None` (`micro_agents.py:135-137`) — *a client
exists*. `last_used_llm` (`:133,140,148`) is *the call succeeded*. This records
`page_enrichment_source = "llm"` for a run that fell back, and the comment shows
it was known and shipped anyway.

In the live campaign both halves were visible at once: `[competition] page
enrichment: llm.` in the analyze notes, `CompetitionPageAnalyzerAgent LLM path
failed` in the log. A reader trusting the artifact would conclude the LLM ran.

**2.2 — Three names for one concept**, and no coverage rule:

| Field | Where | Durable? |
|---|---|---|
| `extraction_source` | `fetch/enrich.py:46,104` | yes — artifact metadata |
| `normalized_by` | `knowledge/models.py:56` | yes — model field |
| `generated_by` | `brief/models.py:30`, `planner/schemas/models.py:74`, `shared/experiments/models.py:115` | yes — model field |
| *(nothing)* | most agents | — |

**2.3 — The fallback is automatic.** Nothing distinguishes "I chose the
deterministic path" from "the LLM failed and I substituted one". A rule engine
firing is a *silent* event by construction.

## 3. Goal

Degradation is impossible to miss: either the LLM served the call, or the result
is stamped as degraded, or the command fails.

## 4. Requirements

### Functional

| # | Requirement |
|---|---|
| F1 | Every result carries `generated_by` reflecting **what actually happened**, from `last_used_llm` — never `uses_llm` |
| F2 | One canonical field name and value set, replacing `extraction_source` / `normalized_by` |
| F3 | A rule-engine result cannot reach a durable write unstamped |
| F4 | Fallback logs at WARNING with the agent name and the failure reason (exists at `:165`; must survive) |
| F5 | Automatic fallback is opt-in only — without an LLM and without the flag, commands **fail** |
| F6 | Rule engines that encode genuine deterministic domain logic are promoted to named first-class steps, not deleted |
| F7 | `research doctor` already reports provider health; a degraded *run* is reportable the same way |

### Non-functional

| # | Requirement |
|---|---|
| N1 | Phase 1 changes no behaviour — only labels. Safe to ship alone |
| N2 | Phase 1 is one base-class change plus write-path threading, not 20 agent edits |
| N3 | Tests stay hermetic and offline — they stub the **LLM client**, not the rule engine |
| N4 | An explicit deterministic mode remains for CI, reusing the existing `--offline` precedent |

## 5. Scope

### In scope

- Canonical `generated_by` on every micro-agent result, sourced from
  `last_used_llm`
- Fixing the `uses_llm` provenance bug
- Collapsing `extraction_source` / `normalized_by` onto the canonical field
- Making automatic fallback opt-in (phase 2)
- Triaging the 20 rule engines into *delete* vs *promote* (phase 3)
- Test migration from rule-engine reliance to stubbed LLM clients

### Out of scope

- Which model serves a role → [M10](../04-llm-tiering.md)
- Analyzer-level soft-fail (an unreachable arXiv is a missing *source*, not
  missing reasoning — that degradation is correct and stays)
- The template/stub fallback in code engineering — already fails loudly for a
  non-dry run
- Prompt quality

## 6. High-level design

```
            BaseMicroAgent.run()
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   LLM succeeded          LLM failed / absent
        │                       │
        │              ┌────────┴────────┐
        │              ▼                 ▼
        │      deterministic mode    normal mode
        │      (--deterministic)         │
        │              │                 ▼
        │              │            ┌─────────┐
        │              │            │  RAISE  │  ← phase 2
        │              │            └─────────┘
        ▼              ▼
  generated_by=    generated_by=
     "llm"          "rule_engine"   ← phase 1, always stamped
        │              │
        └──────┬───────┘
               ▼
    every durable write carries it
```

Phase 1 delivers the stamping and the bug fix — no behaviour change. Phase 2
flips the default. Phase 3 removes what is then dead.

## 7. Components and responsibility boundaries

| Component | Owns | Does **not** own |
|---|---|---|
| `BaseMicroAgent.run()` | Deciding which path ran; setting `last_used_llm`; raising in phase 2 | What the result means |
| `AgentOutcome` (new, thin) | Carrying `generated_by` + failure reason alongside the model | Persistence |
| Write paths (`enrich`, `merger`, `brief`, `assistant`, analyzers) | Propagating the stamp onto their durable record | Deciding the value |
| The 20 rule engines | Deterministic domain logic where genuinely wanted | Standing in for a failed LLM (phase 3) |
| `conduct --offline` / `--deterministic` | Operator's explicit choice | Silent substitution |

**Boundary that matters:** the agent decides *what happened*; the write path
records it. Today the write path *infers* it — and `competition.py:359` infers
wrong.

## 8. Design choices

### 8.1 Standardise on `generated_by`

**Chosen.** It already has three users (`brief`, `planner`, `experiments`)
against one each for `extraction_source` and `normalized_by`, and it names the
right thing. Values: `llm | rule_engine | template_fallback | stub`.

Old field names remain as read-aliases for one release so stored artifacts stay
readable.

### 8.2 Stamp before restricting

**Chosen.** Phase 1 (label everything, change nothing) ships alone and is
immediately valuable — it makes phases 2–3 *and* [M10](../04-llm-tiering.md)'s
wiring observable. Without it, a router change cannot be distinguished from a
fallback.

*Rejected:* deleting rule engines first. It breaks the suite before there is any
signal showing where degradation actually occurs.

### 8.3 Raise rather than warn when the LLM is unavailable (phase 2)

**Chosen.** A missing LLM means *no reasoning happened*. Returning a
deterministic approximation labelled as a result invites it into beliefs and
claims, where it is expensive to remove.

*Rejected:* warn-and-continue. That is today's behaviour with better logging,
and the whole defect is that warnings are not read while artifacts are.

### 8.4 Promote genuine deterministic logic; delete only stand-ins

**Chosen.** Of 20 rule engines, some are true LLM substitutes (a
`ResearchBriefNarrative` written by regex is a stand-in) while others encode
real domain rules (concept normalisation, some scoring). The latter deserve to
be *named* deterministic steps, not disguised as failed LLM calls.

Triage is per-agent and belongs in phase 3, where the phase-1 stamps show which
ones actually fire in practice.

### 8.5 Reuse the existing `--offline` precedent

**Chosen.** `conduct --offline` → `prefer_offline` already means "deterministic
catalog order, no LLM policy" (`cli/conduct.py:202-203,271`). Extending that
vocabulary to agents is cheaper and less confusing than a second flag with
different semantics.

### 8.6 Soft-fail stays correct for *sources*, not for *reasoning*

**Chosen, and load-bearing.** An analyzer that cannot reach arXiv should degrade
and continue — a source is missing, and the system still knows what it knows. An
agent whose LLM call failed produced *no thinking*.

The current code treats these identically. M14 splits them; the analyzer
soft-fail in `orchestrator.py::_run_one` is deliberately left alone.

## 9. Low-level design

### 9.1 Outcome carries provenance

`run()` returns the model as today (call sites unchanged) but records
`self.last_generated_by` and `self.last_failure_reason` alongside the existing
`last_used_llm`. Write paths read those instead of inferring.

```python
GeneratedBy = Literal["llm", "rule_engine", "template_fallback", "stub"]
```

### 9.2 The one-line bug fix

```python
# analyzers/competition.py:359
- source = "llm" if agent.uses_llm else "rule_engine"
+ source = agent.last_generated_by
```

and delete the comment conceding the inaccuracy.

### 9.3 Phase 2 gate

```python
if self.llm_client is None and not deterministic_allowed:
    raise LLMUnavailableError(
        f"{self.name} requires an LLM. Start Ollama or configure a provider "
        "(`research doctor`), or pass --deterministic."
    )
```

Message style matches `research doctor`'s existing actionable-fix convention.

### 9.4 Phase 3 triage

For each of the 20 agents, decide *delete* (stand-in) or *promote* (named
deterministic step). Phase-1 telemetry — which rule engines actually fired, and
how often — is the input to that decision, so phase 3 must not start first.

### 9.5 Test migration

The real cost, and the **only** blocker for phase 2a.

**Measured, not estimated.** Simulating 2a (raise when `llm_client is None`) and
running the suite: **76 tests fail** — roughly 11% of 709.

Grep undercounts this badly. Only 4 sites construct a micro agent directly; the
other ~46 construct a *capability* with `llm_client=None`, which builds an agent
internally. The blast radius is only visible by running it, which is why the
number in this section is measured rather than counted.

Migration: a `stub_llm_client` fixture returning canned JSON per agent, so tests
exercise the **shipped path**. A test asserting rule-engine output is not
testing what production does — which is why this is an improvement rather than
a tax.

Tests that legitimately cover deterministic behaviour keep it via the explicit
flag.

## 10. Testing strategy

Same rule as [M7](01-technique-to-model.md): **assert the effect, not the call.**

| Level | Check |
|---|---|
| Unit | LLM success ⇒ `generated_by == "llm"`; forced failure ⇒ `"rule_engine"` **and** a non-empty reason |
| Unit | `uses_llm` is never the source of a provenance value (regression for §2.1) |
| Write-path | For each durable writer, a fallback result lands with the stamp on the stored record — read it back, do not assert the call |
| Phase 2 | No client and no flag ⇒ raises; with `--deterministic` ⇒ succeeds and stamps `rule_engine` |
| Suite-wide | **No durable write path can produce an unstamped record** — enumerate writers and assert coverage, so a new agent cannot quietly skip it |
| Regression | Phase 1 changes no output except the added field (N1) |

The suite-wide coverage test is the one that keeps this fixed. Individual stamps
rot; an enumeration test fails when someone adds writer number twelve.

## 11. Evaluation

Testing proves the stamp is correct. Eval answers **how much degradation is
actually happening** — currently unknown, because it was never recorded.

### 11.1 Measured baseline — rogii, 9 campaigns (2026-08-02)

Run before writing the rest of this section, and it **contradicted the design's
premise**. Recovered from the nine `research conduct` campaign logs:

| Campaign | analyze dispatched | fallbacks | reading |
|---|---|---|---|
| 1–3 | 2 each | **1 each** | ran, fell back every time |
| 4–5 | 2 each | **0** | ran, no fallback |
| 6–9 | **0** | 0 | contributes nothing — backlog gate skipped analyze |

All three fallbacks were `CompetitionPageAnalyzerAgent`, all
`Response did not contain a JSON object` — the model answering in prose.

**Clean before/after: 3 of 3 campaigns fell back before constrained JSON
decoding; 0 of 2 after.** The `format: "json"` fix already shipped in this
branch removed the observed cause.

Two corrections this forces on the design:

1. **"Expected: high fallback rate on a 14B model" was wrong for this
   workload.** The acute problem is already fixed. M14 cannot be justified by
   "the system is currently running on rule engines" — post-fix, it is not.
2. **Campaigns 6–9 must be excluded**, not counted as successes. Zero fallbacks
   there means zero agent calls. Counting them would have produced "6 of 9
   campaigns clean" — a measurement artifact reported as an improvement, which
   is the exact disease this roadmap exists to remove.

### 11.2 The blind spot — why M14 still matters

The measurement above can only see **one of two fallback paths**.
`BaseMicroAgent.run()` (`micro_agents.py:139-170`):

```python
if self.llm_client is not None:
    ...try/except, WARNING on failure...      # ← path 1: logged, measurable
return self._run_rule_engine(context)          # ← path 2: no client, NO LOG AT ALL
```

| Path | Trigger | Logged? | In the numbers above? |
|---|---|---|---|
| 1 | client present, call fails | WARNING at `:165` | yes — 3 events |
| 2 | **client absent** | **nothing** | **no — invisible** |

So "0 fallbacks" means *0 among agents that had a client*. An agent constructed
with `llm_client=None` degrades in complete silence, and no amount of log
analysis will ever find it.

**This re-justifies M14 on firmer ground than the original premise.** Not "we are
currently degraded" — measurably, post-fix, we are not. But *one entire
degradation path is unobservable by construction*, and the only reason the
current fallback rate is knowable at all is that the failures happened to take
the logged path.

That is also why phase 1 stamps the **artifact** rather than adding more logging:
the stamp is set where the branch is taken, so path 2 becomes visible for the
first time.

### 11.3 Rogii eval protocol

Repeatable, and specified so the same numbers can be produced before and after
each phase.

**Setup.** The rogii workspace, `LABPILOT_HYPOTHESIS_BACKLOG_TARGET=0` to force
analyze to run (otherwise the backlog gate skips the agents entirely and the
measurement is vacuous — see campaigns 6–9).

**Runs.**

| Run | Config | Measures |
|---|---|---|
| A | local 14B, current `main` | today's true fallback rate, both paths |
| B | local 14B, phase 1 shipped | same, now including path 2 |
| C | frontier model via [M10](../04-llm-tiering.md), phase 1 | does capability remove fallback? |
| D | no LLM reachable, phase 2 | does it fail loudly rather than degrade? |

**Metrics** (per run, per agent):

| Metric | Source | Target |
|---|---|---|
| Fallback rate, path 1 | WARNING count ÷ agent invocations | reported |
| **Fallback rate, path 2** | `generated_by == "rule_engine"` with no WARNING | **unknown today — the point of A→B** |
| Degraded-artifact share | durable records where `generated_by != "llm"` | reported |
| **Silent-lie count** | records claiming `llm` where the call fell back | **0** — release blocker |
| Unstamped-record count | durable records with no `generated_by` | **0** after phase 1 |

**A→B is the key comparison.** If B reports meaningfully more degradation than
A, the difference is exactly the silent path — and that number is phase 1's
justification, measured rather than argued.

**C answers M10's question**, and does so with the same harness: if a frontier
model does not collapse the fallback rate, the problem is not model capability
and M10's premise needs revisiting.

### 11.4 Phase 3 input

Rank rule engines by fire rate, using run B's stamps (run A cannot see path 2).
One that never fires is dead code. One that fires constantly is either
load-bearing domain logic — promote it to a named deterministic step — or it is
masking a persistent LLM failure, in which case fix the cause rather than the
symptom.

Run C decides which: if a frontier model stops a rule engine firing, it was
masking a failure; if it keeps firing, it is doing real work.

## 12. Observability

| Signal | Where | Answers |
|---|---|---|
| `generated_by` on every durable record | artifacts, knowledge.db, plans, experiments | Was this reasoned or approximated? |
| WARNING with agent + reason | log (exists, `:165`) | Why did it fall back? |
| Fallback counters per campaign | `conduct status` | How degraded was this run? |
| `LLMUnavailableError` | command exit | Phase 2 refusing to pretend |

Campaign-level counters matter most: one fallback is noise, forty is a broken
substrate, and today both look identical.

## 13. Production readiness

**Phasing.** Phase 1 alone is shippable and reversible — additive field, no
behaviour change (N1). Phase 2 needs the test migration budgeted first. Phase 3
needs phase 1's telemetry.

**Rollout risk.**

| Risk | Mitigation |
|---|---|
| Phase 2 breaks a workflow depending on implicit fallback | `--deterministic` preserves it explicitly; error names the flag |
| Test migration underestimated | **Measured at 76 failing tests** by simulating the raise, not estimated from grep (which said 4). Budget it, do not discover it |
| A new agent skips the stamp | Suite-wide coverage test (§10) |
| Stored artifacts with old field names | Read-aliases for one release (§8.1) |

**Ordering, with trigger conditions rather than vague sequencing.**

| Phase | Status | Ships when | Blocked by |
|---|---|---|---|
| 1 | **done** | — | — |
| 2a | **done** | — | — |
| **2b** | deferred by decision | after [M10](../04-llm-tiering.md) is live | M10, measured by §11.3 run C |
| **3** | deferred by decision | after M10 + several stamped campaigns | phase-1 telemetry |

**Decision (2026-08-03).** 2b waits for M10 entirely rather than shipping
opt-in behind a flag. An opt-in build was offered and declined, on the grounds
of keeping the branch to what is verified.

> **Guard for whoever builds 2b.** Because it is not being written now, it will
> arrive after M10 with no exercise history. Do not ship it on unit tests alone
> — that is exactly how `select_route` reached review tested, unwired and
> described as done. 2b's acceptance is a **real campaign that completes** with
> strict mode on, not a passing suite.

**Why 2b is riskier than §8.3 implies.** The failure actually observed —
`Response did not contain a JSON object` — is **not** classified transient by
`_is_transient_llm_error`, so it gets no retry. Under 2b a single prose reply
aborts the whole command. Post-JSON-fix the observed rate is 0 of 2 campaigns,
which is too thin to justify a total-abort blast radius on the current
substrate.

Phase 1 is roadmap phase 0 — before [M10](../04-llm-tiering.md) — because it is
what makes M10's wiring observable. 2a can follow immediately after; it was
deferred on a reason that did not survive checking.

**Explicit non-goal.** This does not make the LLM output *good*. It makes
degradation *visible*. A system that fails loudly is not yet a system that
reasons well — that is [M10](../04-llm-tiering.md).

**A note on how this design was sized.** The eval in §11 was run *before* the
rest of the document was finished, and it refuted the premise the plan had been
written on ("expect a high fallback rate on a 14B model"). The acute problem was
already fixed; what remained was a structural blind spot. Running the
measurement first cost an hour and changed what the milestone is for — worth
repeating on the next design rather than treating §11 as something to write
after the fact.
