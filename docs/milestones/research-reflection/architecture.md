# Research Reflection — Architecture

Back to [README](README.md).

---

## 1. Placement

New pillar under `src/labpilot/research_engine/reflection/`, peer of
`intelligence/`, `planner/`, `execution/`.

| Rule | Choice |
|------|--------|
| Trigger | After Engineer execution (Reporting TaskTypes) + CLI `research reflect` / `journal` |
| Capabilities | Deterministic extractors + belief/hypothesis writers; LLM only in Critic / synthesis / lessons / recommendations |
| No | Standalone “reflection agent” that owns control flow |
| Import hygiene | `reflection` → `accessor`, `labpilot.research_engine.shared.experiments`, planner/execution **read** APIs; execution may call reflection library; intelligence may **read** claims/beliefs — reflection does not deep-import intelligence analyzers |

---

## 2. Pipeline

```mermaid
flowchart TD
  Exp[Engineer execution result]
  Ev[EvidenceExtractor]
  Crit[ExperimentCritic]
  Bel[BeliefUpdater]
  Hyp[HypothesisEvaluator]
  Syn[KnowledgeSynthesizer]
  Mem[ResearchMemory / Lessons]
  Next[NextSuggestions]
  Exp --> Ev --> Crit --> Bel --> Hyp --> Syn --> Mem --> Next
  Next -.->|feeds| Plan[Planner / next plan]
```

Bayesian posture: each experiment updates understanding of the world (beliefs,
claims, lessons), with an audit trail (`belief_updates`).

---

## 3. Component contracts

| Component | Input | Output | LLM? |
|-----------|-------|--------|------|
| **EvidenceExtractor** | Execution evidence, metrics, config, comparison | `ExperimentEvidence` rows | No |
| **ExperimentCritic** | Evidence + plan/hypothesis context | Assessment, confidence, recommendation | Yes (+ rule_engine) |
| **BeliefUpdater** | Critic verdict + prior beliefs | Updated `beliefs` + `belief_updates` audit | No (arithmetic) |
| **HypothesisEvaluator** | Critic + plan `hypothesis_id` | Status + why (`confirmed`/`rejected`/…) | No status; yes why text optional |
| **KnowledgeSynthesizer** | Beliefs + evidence + claims | “Current Understanding” rollup | Optional |
| **Lessons** | Cross-competition patterns | `lessons` rows | Yes (+ rule_engine) |
| **ClaimPromoter** | Strong beliefs + evidence | `research_claims` (+ edges) | Optional narrative |
| **Journal** | Projection over SoR | Human-readable research journal | Assembly only |
| **Recommend** | Journal + open questions | Next-experiment suggestion | Yes (+ rule_engine) |

---

## 4. Engineer hook

Existing Reporting TaskTypes in
`execution/capabilities/reporting/capability.py`:

- `REFLECT`
- `UPDATE_BELIEF`
- `CREATE_HYPOTHESIS`

**Today:** workspace JSON stubs — no durable DB mutation.  
**After Plan 5:** call into `research_engine.reflection` library; auto-run on
execution success/fail when those tasks are in the plan DAG.

LLM reflection slices live co-located with their domain packages
(`critic/`, `contradiction/`, `confidence/`, `synthesis/`, `lessons/`,
`hypotheses/`, `recommendation/`) — each has `micro_agent.py` + `skill.md`.
`RootCauseAgent` aliases the legacy `ReflectionGeneratorAgent` name; execution
still re-exports that name for compatibility. Facades compose the agents;
Reporting can call `run_reflection`.

---

## 5. LLM boundary (locked)

| Use LLM (Micro Agent) | Keep deterministic |
|----------------------|-------------------|
| RootCauseAgent | Metric deltas, ranking, storage, EvidenceExtractor |
| ContradictionDetectorAgent | Belief confidence arithmetic (BeliefUpdater) |
| EvidenceSynthesisAgent | Journal assembly / history queries |
| ConfidenceEstimatorAgent (qualitative) | Hypothesis **status** enum writes |
| LessonGeneratorAgent | Claim promotion thresholds |
| HypothesisRevisionAgent (why / revised text) | |
| RecommendationAgent | |

Offline: every Micro Agent has a `rule_engine` path (same posture as Intelligence).

Each agent is imported from its domain package (e.g. `reflection.critic.RootCauseAgent`).

---

## 6. Reuse (don’t rewrite)

| Asset | Role |
|-------|------|
| `experiments/comparator.py` | Deterministic critic **input** |
| `experiments/hypothesis.py` + models | Hypothesis SoR |
| `experiments/knowledge.py` | File KB — migrate writers into reflection; dual-write briefly if needed |
| SQLite `beliefs` / `experiments` / `evidence_links` | Extend; add reflection tables |
| M2 `StructuredReflection` + top-level `reflection/` | Schema/prompts migrate; delete top-level after Plan 9 |

These live at `labpilot.research_engine.shared.experiments` — see
[package-layout.md](package-layout.md) §2.


---

## 7. Belief-graph note

Technique ↔ effects ↔ experiments/papers as an **explicit graph UI** can be
Plan 7b or follow-on. Schema uses `evidence_links` / claim edges so the graph is
a **projection**, not a second SoR.
