# M14 — LLM is a hard dependency; delete the rule-engine fallbacks

**Status:** decided, not started · **Decision owner:** product

---

## Decision

> The system requires an LLM. **Ollama is the floor**, not a bonus. If no LLM is
> reachable, commands **fail with a clear error** instead of silently producing
> deterministic filler.

The rule-engine path is deprecated as an *automatic* fallback.

## Purpose

This is the root cause of the "silent failure at every layer" pattern, not one
instance of it.

**20 micro agents** each implement `_run_rule_engine`, and `BaseMicroAgent`
catches any LLM or parse failure and quietly uses it:

```
research_engine/intelligence/micro_agents/  hypothesis_generator, paper_analyzer,
  competition_page_analyzer, concept_normalizer, repository_analyzer,
  research_brief, combo_portfolio, forum_analyzer, experiment_reviewer,
  intent_classifier, repo_query_planner
research_engine/reflection/  critic, lessons, synthesis, confidence,
  contradiction, hypotheses, recommendation
research_engine/planner/     planning_engine
research_engine/execution/   code_engineer
```

Observed consequence: `qwen2.5-coder:14b` answered a JSON-only prompt in English
prose, the parse failed, and the agent silently used its rule engine. The system
*looked* like it was doing LLM analysis while running deterministic rules. The
Knowledge Hub then found zero concepts, so no techniques, no beliefs, no
hypotheses — and the campaign had nothing to iterate on.

Nothing above that layer could tell. There was no error, no degraded flag, no
metric. The only symptom was "the research is oddly shallow".

The same shape produced the rule-engine hypotheses `vit`, `cnn`, `Mixed` and
`test` on a tabular regression task.

## Goal

Degradation is impossible to miss: either the LLM served the call, or the
command failed, or the result is explicitly stamped `degraded`.

## Approach — phased, because it touches 20 agents

**Phase 1 — make it loud (cheap, do first).**
Keep the fallback, but a rule-engine result must set `generated_by="rule_engine"`
on the artifact *and* log at WARNING with the reason. Any downstream durable
write carries the flag. This alone would have made the whole failure visible on
day one.

**Phase 2 — make it opt-in.**
Automatic fallback off by default. A rule engine runs only under an explicit
`--deterministic` flag. Without an LLM and without the flag, the command raises
with the same actionable message style `research doctor` already produces
("Start Ollama, or `ollama pull <model>`").

**Phase 3 — delete or demote.**
Rule engines that exist purely as LLM stand-ins are deleted. Ones encoding
genuine deterministic domain logic (some normalisers and scorers are legitimately
rule-based) are promoted to *first-class deterministic steps* with their own
names — not disguised as failed LLM calls.

## The cost, stated honestly

**This will break the test suite.** Tests were made hermetic by forcing the
Ollama liveness probe closed — which means ~600 tests currently exercise the
**rule-engine path**. Removing automatic fallback makes them fail.

The migration is: tests stub the *LLM client* (returning canned JSON) rather than
relying on rule engines to fill the gap. That is better practice anyway — a test
asserting rule-engine output is not testing the shipped behaviour — but it is
real work and should be budgeted, not discovered.

Phase 1 is safe and immediately valuable; phases 2–3 need that test migration
first.

## Exit criteria

1. With Ollama stopped, `research analyze` and `research conduct` **fail** with
   an actionable message rather than producing a report.
2. No durable artifact can be written from a rule-engine result without
   `generated_by="rule_engine"` recorded on it.
3. `grep -c "_run_rule_engine"` trends to the small set of genuinely
   deterministic steps, each named as such.
4. The test suite passes against stubbed LLM clients, not rule engines.

## Traps

- **Do not confuse soft-fail in *analysis* with soft-fail in *reasoning*.** An
  analyzer that cannot reach arXiv should degrade and continue — that is a
  missing *source*. An agent whose LLM call failed produced *no thinking* and
  must not pretend otherwise.
- **Ollama reachable ≠ Ollama useful.** `research doctor` already checks the
  model is pulled. A 14B model returning prose is a *usable client producing
  unusable output* — which is why [M10](04-llm-tiering.md)'s constrained JSON
  decoding and role-based routing are part of the same story.
- **Keep one escape hatch for CI.** A `--deterministic` mode that is explicit,
  logged, and never the default is fine. The disease was that it was automatic
  and invisible.

## Related code

- `src/labpilot/accessor/common/micro_agents.py` — `BaseMicroAgent`, the catch-and-fallback site
- `src/labpilot/llm/json_utils.py` — parse hardening (shipped)
- `src/labpilot/llm/ollama.py` — `format: "json"` constrained decoding (shipped)
- `src/labpilot/diagnostics.py` — `_check_llm_provider` (shipped)
- `tests/conftest.py` — the fixture that currently forces the rule-engine path
