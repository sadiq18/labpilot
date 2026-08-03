---
name: design-doc
description: >-
  Write focused, concise design documents for any system, feature, or architectural work. Skip sections that don't apply - only include what's needed to make the design clear and implementable. Use this whenever working on: system architecture, microservices, API design, infrastructure changes, feature design, or technical decisions. Include only sections that require explanation or buy-in, skip obvious ones. Better a 2-page focused doc than a 20-page document with filler.
compatibility: []
---

# Design Document - Minimal Framework

Write ONLY the sections relevant to your design. Skip anything that doesn't add information.

---

## When to Include Sections

**Include:**
- Background (if context matters)
- Problem Statement (if not obvious why this is needed)
- Requirements (usually always)
- Goals & Metrics (if success isn't clear)
- Scope (if boundaries aren't obvious)
- Design/Architecture (if approach matters)
- Components (if multiple pieces exist)
- Implementation Details (if complex)
- Tradeoffs (if alternative approaches exist)
- Observability (for production systems)
- Testing (for risky changes)
- Rollout Plan (for production changes)

**Skip:**
- Anything obvious or documented elsewhere
- Sections with no relevance to your change
- Filler or boilerplate

---

## 1. Background & Context
*Skip if team already knows why this is needed.*

State current situation in 2-3 sentences. Why does this matter?

---

## 2. Problem Statement
*Skip if problem is obvious.*

What's broken? With metrics if possible. Why current approach doesn't work?

---

## 3. Requirements
*Usually needed.*

**Functional:** What must it do? (list key features)
**Non-Functional:** Performance targets, scale, reliability (use numbers)

---

## 4. Goals & Success Metrics
*Skip if success is obvious.*

What are we optimizing for? How do we measure success? (specific metrics)

---

## 5. Scope
*Skip if boundaries are obvious.*

**In-Scope:** What this covers  
**Out-of-Scope:** What it doesn't (and why)

---

## 6. Design/Architecture
*Skip for tiny changes.*

Simple diagram or description:
- Major components
- How they interact
- Key decisions

Keep it simple. ASCII art, flowchart, or 1-2 paragraphs is fine.

---

## 7. Components & Responsibility
*Skip if only changing one component.*

For each significant component:
- What it does
- Interfaces (inputs/outputs)
- Dependencies
- Constraints (if any)

Keep it brief - a table or 2-3 lines per component.

---

## 8. Implementation Details
*Skip if implementation is obvious from context.*

Code examples, configuration, data structures, algorithms.
Include what's needed to clarify the approach.

**Only include if complex:**
- Algorithm details (if non-standard)
- Configuration examples (if non-obvious)
- Data structure design (if critical)
- Concurrency/threading (if relevant)
- Error handling edge cases (if important)

---

## 9. Design Choices & Tradeoffs
*Skip if only one reasonable option exists.*

For key decisions:
- What options were considered?
- Why this option?
- What's the tradeoff?

Example:
| Choice | Option A | Option B (chosen) | Tradeoff |
|--------|----------|------------------|----------|
| Storage | SQL | NoSQL | Consistency vs scale |

---

## 10. Observability
*Skip for internal tools. Include for production systems.*

For production systems:
- Key metrics to track (3-5)
- Alert thresholds (what matters?)
- How to debug issues

---

## 11. Testing Strategy
*Skip for low-risk changes.*

For significant changes:
- What scenarios to test? (1-2 key ones)
- How to verify it works?
- Failure scenarios (if critical)?

---

## 12. Deployment & Rollout
*Skip for non-production changes.*

For production:
- How to roll out? (canary %, phases?)
- How to rollback?
- What could go wrong?

---

## Evaluation Criteria

For your design, verify:
- ✓ Clear problem statement
- ✓ Requirements are quantified (use numbers)
- ✓ Design approach is explained
- ✓ Tradeoffs are documented
- ✓ Testing/validation approach exists
- ✓ Rollback/safety plan exists (for prod)

---

## Write for Clarity

- Use numbers, not "fast" or "scalable"
- One diagram > thousand words
- Code example > 100 lines of description
- Assume reader is smart but doesn't know your context
- Focus on decisions that require buy-in or clarity
- Skip anything that's obvious or documented elsewhere

---

## Length Guidelines

- **Simple change:** 1-2 pages
- **Medium change:** 2-4 pages
- **Major architecture:** 4-8 pages
- **If over 10 pages:** You're probably including too much detail

---

## Quick Checklist

Before publishing:
- [ ] Problem is clear
- [ ] Requirements make sense
- [ ] Design solves the problem
- [ ] Important decisions are explained
- [ ] Reader knows how to validate it works
- [ ] Reader knows risks/how to rollback

If all checked, you're done. Don't add more just to fill pages.
