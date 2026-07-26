# Research Reflection

Back to [MILESTONES.md](../../MILESTONES.md).

**Status:** Design Phase A complete; Phase B Plans **1–2 Done**; Plans 3–10 ready.
**Milestone:** 6 — close the research loop.
**Product name:** Research Reflection.
**CLI (target):** `research reflect` / `research journal` (+ Engineer auto-hook after `run --plan`).

This directory is the architecture/design workspace for **Research Reflection** —
turning experimental outcomes into durable research knowledge.

| Doc | Role |
|-----|------|
| **This README** | Mission, researcher vs automation, success criteria, non-goals |
| [architecture.md](architecture.md) | Evidence → Critic → Beliefs → Hypotheses → Synthesis → Journal |
| [schema.md](schema.md) | `experiment_evidence`, `belief_updates`, `lessons`, `research_claims` |
| [package-layout.md](package-layout.md) | `research_engine/reflection/` + legacy migration |
| [beliefs-and-claims.md](beliefs-and-claims.md) | Belief vs Research Claim lifecycle |
| [cli.md](cli.md) | `reflect`, `journal`, `claims` |

### Phase B implementation plans

| # | Plan | Focus |
|---|------|--------|
| 1 | [plan-1-schema-stores.md](plan-1-schema-stores.md) | DDL + `ReflectionStore` |
| 2 | [plan-2-evidence-extractor.md](plan-2-evidence-extractor.md) | Deterministic evidence (no LLM) |
| 3 | [plan-3-experiment-critic.md](plan-3-experiment-critic.md) | Critic + Micro Agent |
| 4 | [plan-4-belief-hypothesis.md](plan-4-belief-hypothesis.md) | BeliefUpdater + HypothesisEvaluator |
| 5 | [plan-5-engineer-cutover.md](plan-5-engineer-cutover.md) | Wire Reporting TaskTypes |
| 6 | [plan-6-lessons-synthesis.md](plan-6-lessons-synthesis.md) | Lessons + Current Understanding |
| 7 | [plan-7-research-claims.md](plan-7-research-claims.md) | Research Claim SoR |
| 8 | [plan-8-journal-cli.md](plan-8-journal-cli.md) | `research journal` / `reflect` |
| 9 | [plan-9-legacy-cleanup.md](plan-9-legacy-cleanup.md) | Delete/migrate top-level `reflection/` |
| 10 | [plan-10-capstone.md](plan-10-capstone.md) | End-to-end run → reflect → journal |

Ship-and-review one plan at a time (same style as Research Engineer).

---

## 1. Mission

> **Convert experimental outcomes into durable research knowledge.**

Not “generate a nicer report.” After every experiment, answer:

> What have we learned, and how should that change our future decisions?

Without reflection, LabPilot is an automation pipeline.  
With reflection, it behaves like a researcher (Bayesian updating of beliefs and claims).

### Researcher vs report writer (locked)

| | Report writer (rejected) | Research Reflection (this milestone) |
|--|--------------------------|-------------------------------------|
| Product | Narrative HTML/markdown | Durable knowledge + audit trail |
| End state | “Done” | Updated beliefs / claims / hypotheses / lessons |
| Analogy | Lab notebook page | Scientific memory that compounds |

---

## 2. Where this sits in the Research OS

```
analyze → knowledge / beliefs / hypotheses
              ↓
           plan  (Research Planner — shipped)
              ↓
    research run  (Research Engineer — shipped)
              ↓
         reflect  ← this milestone
              ↓
    updated beliefs / claims / lessons / journal
              ↓
         next plan (human or Planner)
```

---

## 3. Loop (SoR)

```text
Experiment result
  → EvidenceExtractor   (deterministic)
  → ExperimentCritic    (LLM + rule_engine)
  → BeliefUpdater       (deterministic math + audit)
  → HypothesisEvaluator (status + why)
  → KnowledgeSynthesizer / Lessons
  → Research Journal + next recommendation
```

---

## 4. Success criteria

- After `research run --plan`, durable updates without manual `hypothesize update`:
  evidence rows, `belief_updates`, hypothesis status+why (when plan linked).
- `research journal --competition <slug>` shows strong/moderate/weak/rejected evidence,
  open questions, and a recommended next experiment.
- **Research Claims** are queryable and distinct from beliefs.
- Offline CI works with `rule_engine` (no LLM required).
- Top-level `labpilot.reflection` removed after migration.

---

## 5. Non-goals

- Multi-agent debate or autonomous plan execution from reflection
- Replacing Intelligence Analyze
- Full causal / counterfactual *measurement* (propose alternatives only)
- Kernel-mode submission parity
- Resurrecting Pipeline `improve` / linear `write_reflection`

---

## 6. CLI sketch (target)

```bash
research run --plan P-001 --competition <slug>   # auto-reflect via Reporting tasks
research reflect --execution E-001 --competition <slug>
research journal --competition <slug>
research claims list --competition <slug>        # Plan 7+
```

Details: [cli.md](cli.md).
