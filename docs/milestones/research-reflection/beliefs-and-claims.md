# Beliefs vs Research Claims

Back to [README](README.md).

---

## 1. Locked distinction

| Object | Role |
|--------|------|
| **Belief** | Fast-moving working assumption (technique × effect × confidence) — already in SQLite `beliefs` |
| **Research Claim** | Synthesized conclusion the system will stand behind — first-class object with evidence + contradictions |

Claims are what Planner/ranker should prefer as **stable inputs**.  
Beliefs remain for **short-horizon** updates after each experiment.

---

## 2. Belief lifecycle

```text
suggested  →  validated  →  established
                 ↓
              rejected
```

- Created/updated from Analyze (Intelligence) and from Reflection (BeliefUpdater).
- Every Reflection mutation appends a `belief_updates` row (prior → new, reason, evidence_id).
- Confidence arithmetic is **deterministic** (rules); Critic supplies direction/strength, not raw float math.

---

## 3. Claim lifecycle

```text
candidate  →  supported
      ↓           ↓
  contested ←—————┘
      ↓
  withdrawn
```

Promotion rules (Plan 7 — tune in implementation):

| From | When |
|------|------|
| Belief → Claim (`candidate`) | Confidence ≥ threshold **and** ≥ N supporting evidence rows of strength ≥ moderate |
| → `supported` | No unresolved contradictions; additional confirming runs |
| → `contested` | Contradicting evidence or rival claim |
| → `withdrawn` | Explicit reject or stronger superseding claim |

---

## 4. Hypothesis interaction

- Plan start (Engineer): `mark_testing` on linked hypothesis (Plan 4/5 hook).
- After Critic: confirm / reject / partial / inconclusive **with why**.
- Optional: Critic may suggest a new hypothesis (`CREATE_HYPOTHESIS` TaskType).

Status vocab: prefer `proposed` over SQLite legacy `suggested` (see [schema.md](schema.md)).

---

## 5. What Planner should consume

| Prefer | Avoid as sole input |
|--------|---------------------|
| `supported` claims | Single unvalidated belief from one run |
| Journal “open questions” | One-off workspace JSON reflections |
| Lessons with confidence | Unaudited belief bumps |
