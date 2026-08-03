# Claude Code Configuration for LabPilot

## Project Overview
LabPilot is a machine learning research platform for competition/kaggle work with focus on modular agent design and research pipelines.

---

## 1. Design Documents

**For any system/feature/architectural work, use the design-doc skill.**

This applies to:
- New research pipelines or agents
- System architecture changes
- Feature designs
- Technical proposals
- Technology selections

**Result:** Focused 1-8 page design doc with only relevant sections. No filler.

See: `.claude/skills/design-doc/SKILL.md`

---

## 2. Code Philosophy

### Simplicity First
- Minimum code that solves the problem
- No speculative features or premature abstraction
- No error handling for impossible scenarios
- If unsure between two approaches, pick the simpler one

### Surgical Changes
- Touch only what's needed for the task
- Don't refactor adjacent unrelated code
- Match existing style even if you'd do it differently
- Remove only YOUR dead code, not pre-existing orphans

### Goal-Driven Execution
- Define success criteria before starting
- For multi-step tasks, state plan with verification steps
- Transform vague requests into measurable goals

---

## 3. Research Code Standards

### Dataset/Competition Work
- Keep competition-specific configs in `configs/competitions/` (not committed)
- Use `.cache/` for kaggle downloads (safe to delete, not committed)
- Store research artifacts in `runs/` (local only)
- Use `/knowledge/` for research memory and hypotheses (local only)

### Agent/Pipeline Design
- Each agent should have clear responsibility boundary
- Use decorators and type hints for clarity
- Document decision tradeoffs in design docs, not code comments
- Test against real competition data, not synthetic

---

## 4. Documentation

### When to Write
- Design decisions that need buy-in (use design-doc skill)
- API/function contracts (docstrings, not comments)
- Architecture overview (README, not inline)

### When NOT to Write
- What well-named code already expresses
- "How this works" comments (refactor to be obvious instead)
- Future plans or TODOs (if it's not being done now, don't document it)

---

## 5. Git & Commits

- Commit frequently with clear messages
- Describe WHY not WHAT (diff shows what)
- One concern per commit when possible
- Include context: what problem this solves

---

## 6. Claude Code Behavior

### Skills Available
- `design-doc` - Write system/feature design documents
- Others available via marketplace

### Prompts
- Be specific: file paths, line numbers, examples
- For open questions: ask for one opinion, not analysis
- For implementations: describe what success looks like
- For ambiguous tasks: ask before implementing

---

## 7. Local Config (Not Committed)

Files in `.gitignore`:
- `CLAUDE.md` (this file, local to each developer)
- `.env` (credentials, configs)
- `runs/`, `/knowledge/`, `/competitions/` (research artifacts)
- `.cache/` (kaggle downloads)
- `.vscode/`, `.idea/` (IDE configs)

These are intentionally local so team members can customize while keeping repo clean.

---

## 8. Test Before Commit

- Run linters/formatters if configured
- Test against real data if applicable
- For features: verify the golden path works
- For fixes: verify the bug is fixed AND no regressions

---

**These guidelines are working if:** clearer commit history, fewer rewrites, fewer "why did you change that?" comments, and clarifying questions happen before implementation.
