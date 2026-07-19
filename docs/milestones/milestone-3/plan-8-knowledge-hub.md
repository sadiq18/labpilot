# Plan 8 — Knowledge Extraction hub

Back to [Milestone 3](README.md). Design: README §7 · §8 · knowledge-system merge.

**Status:** Not started. **Depends on:** Plans 2, 3; benefits from 4–7 artifacts. **Unlocks:** Plans 9–10.

---

## Goal

Normalize analyzer outputs into merged knowledge objects (techniques / datasets /
architectures / tasks) via the Extraction hub + `ConceptNormalizerAgent`. Write
`references` / join tables and Suggested beliefs — never auto-promote external claims to
Established.

## Why this matters

Five papers mentioning SpecAugment variants must become one Technique with evidence —
otherwise retrieval and hypotheses stay noisy.

## In scope

- Knowledge Extraction hub API (consume `ResearchArtifacts` → upsert store)
- `ConceptNormalizerAgent` + skill.md (aliases → canonical)
- Merge flagship: one technique row + many evidence links
- Belief rows at status Suggested for external evidence (§12.4)
- rule_engine path when Agent disabled

## Out of scope

- Multi-stage retrieval / ContextBuilder (Plan 9)
- Hypothesis top-10 (Plan 10)
- Neo4j / GraphRAG

## Implementation checklist

| Path | Work |
|------|------|
| `knowledge/extractor.py`, `merger.py` | Hub |
| `micro_agents/concept_normalizer/` | Agent |
| Belief writer | Suggested only for external |
| Tests | Alias bag → one technique + references |

## Acceptance criteria

- Running hub on fixture artifacts yields single SpecAugment-like technique with ≥2 evidence
  refs.
- External-only evidence stays Suggested; does not write Established into M2 KB.
- Works with ConceptNormalizerAgent disabled (heuristics/rule_engine).

## Test plan

- Unit: merge five alias strings.
- Unit: belief status gate.
- No network.

## Review notes

- Extractors must not skip hub and write TechniqueBelief directly.
