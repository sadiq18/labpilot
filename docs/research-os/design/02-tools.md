# Design — Tool registry

Back to [../README.md](../README.md) · Milestone: [../milestones/01-foundation/](../milestones/01-foundation/).

**Milestone:** M1 · **Impl branch:** `research-os-m1-foundation`

---

## Goal

Wrap every V1 capability as a **tool** with a stable name, typed inputs/outputs
(artifacts), and side-effect boundaries. Stage CLIs become thin wrappers.

```text
research plan create  →  tools.generate_plan(...)
research run --plan   →  tools.run_plan(...)   # Engineer underneath
research reflect run  →  tools.reflect(...)
```

---

## Initial catalog (illustrative)

| Tool | Wraps | Outputs |
|------|-------|---------|
| `analyze_competition` | AnalyzeOrchestrator | CompetitionAnalysis, knowledge updates |
| `generate_plan` | Planner | ResearchPlan |
| `run_plan` | ResearchEngineer | Execution, Experiment artifacts |
| `evaluate` / `compare` | Eval capability | metrics, Evidence Card path |
| `submit` / `submit_learn` | Submission | Submission, LB patches |
| `reflect` | Reflection pipeline | journal / claims / beliefs |
| `query_memory` | Knowledge / graph / hypotheses | retrieved bundles |
| `store_memory` | stores | ack / ids |

Later (M1 expand / M5): `search_papers`, `run_shell`, `edit_file`, `run_training`,
`commit_git`, … — add when a milestone needs them; do not boil the ocean in M1.

---

## Registry shape (design)

Each tool declares:

- `name`, `description`
- `input_schema` / `output_artifacts`
- `required_workspace_fields`
- `cost_hint` / `duration_hint` (optional; richer in M5)
- `handler` → existing library function (prefer in-process, not CLI subprocess)

```text
Tool.run(workspace, task, context?) -> ArtifactRefs
```

No special tool framework required. Conductor (M2+) selects tools by name; it does
not import capability internals.

### Expanding catalog — data plane (reuse) vs intelligence (build)

| Tool family | Build or reuse? | Implementation |
|-------------|-----------------|----------------|
| Stage tools (analyze, plan, run, reflect, …) | Wrap V1 | In-process |
| `coding_agent` / implement | **Reuse** via adapter | Claude Code / Aider / OpenHands / Codex-style behind `CodingTool.execute(task, workspace, context)` |
| `execute_python` / train sandbox | **Reuse** | Docker → K8s/Ray jobs; API: code, env, timeout |
| `search` / `open_page` / `extract` | **Reuse** | Playwright, Browserbase, or MCP browser tools |
| `search_papers` | Reuse **providers**; build **relevance** | Semantic Scholar / arXiv / OpenAlex + our technique extraction |
| `git_*` | **Reuse** | git CLI + GitPython |
| `query_memory` / research graph | **Build** | Our memory hierarchy |
| Experiment metrics store | **Hybrid** | MLflow (or similar) under Hypothesis / Decision / Reflection |

```text
Research Conductor
        → Coding Agent Adapter
              → Claude Code | Aider | OpenHands | (later) own coding agent
```

Contract for coding (Phase A — black box):

- **In:** task (“Implement EfficientNet baseline”), dataset/code context, prior experiments
- **Out:** changed files, test/result report, commit hash

Research runtime must not care which backend implemented the code. Build an in-house
coding agent only in Phase B after gaps are proven.

Do not boil the ocean in M1 — add data-plane tools when a milestone needs them.

### MCP

**MCP is a transport**, not the tool semantics. Ship custom in-process tools first
(M1). Add MCP adapters when IDE/desktop clients need the same catalog remotely.
Never let MCP own scheduling or approvals.

---

## Rules

1. Tools return artifacts (or references), not “please run the next stage.”
2. Idempotency / resume remain Engineer/tool concerns where already solved.
3. Legacy CLIs keep working by calling the same tools (Strangler Phase A).
4. Log tool name, args summary, duration, cost, and artifact ids (observability).

---

## Non-goals

- LLM tool-calling protocol in M1 (plain Python registry is enough)
- MCP as a M1 requirement
- Building a full coding agent / browser / sandbox in M1
- Specialist agents (M5)
- Replacing Engineer internals wholesale (adapter can sit beside V1 code eng)
---

## Acceptance (when implementing)

- Registry module with ≥ stage tools above registered
- CLI path for plan/run/reflect invokes tools
- Unit test: registry lookup + fake tool round-trip
