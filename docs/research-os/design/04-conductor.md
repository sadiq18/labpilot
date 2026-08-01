# Design — Research Conductor

Back to [../README.md](../README.md) · Milestone: [../milestones/02-conductor/](../milestones/02-conductor/).

**Milestone:** M2 · **Impl branch:** `research-os-m2-conductor`

---

## Role

The Conductor is the **brain only**.

- Understands the goal
- Inspects workspace + memory
- Creates / reprioritizes tasks
- Stops when objective or budgets say so
- Does **not** train models or write experiment code

```text
observe → think → enqueue tasks → (approvals) → tools/Engineer execute → memory updates → loop
```

---

## M2 scope (kernel)

- Durable campaign/session + objective
- Decision records (reasoning, chosen action, inputs)
- Fixed-sequence mode (Strangler Phase B): Analyze → Plan → Run → Reflect
- Approval hooks (default: before new plan batch + before LB submit)
- Append-only **decision/task event log** (not full pub/sub — that is M5)

---

## Policy inputs

- Goal / target metric / time / submission budget
- Workspace summary
- Memory ports (`query_memory`)
- Task queue state
- Last experiment / Evidence Card outcomes

---

## Custom runtime (not a graph framework)

The Conductor **is** the orchestration. Do not encode “what happens next” as a
LangGraph/CrewAI graph — those frameworks make the graph *the* architecture and
fight dynamic planning, debugging, and parallel scheduling.

Treat long-running research like a **job control plane** (Kubernetes metaphor):

```text
Conductor → Task Queue → Workers (tools / Engineer / agents)
         → Events → Retry → Checkpoint
```

Agent/campaign states: Running | Paused | Sleeping | Waiting | Failed | Retry | Completed.

---

## LLM layer (isolated)

Policy LLMs must not leak provider APIs into agents/tools.

```text
Provider → Model Router → Prompt Builder → Structured Output → Retry → Logging
```

| Piece | Choice |
|-------|--------|
| Gateway | **LiteLLM** when Conductor policy lands (thin router OK until then) |
| Structured out | Pydantic models; optional **PydanticAI** / Instructor (**M5** agent paths) |
| Logging | Decision, prompt id, tokens/cost, tool calls — structured logs in **M2**; OTel later |

Swap Claude / GPT / Gemini / Qwen / DeepSeek / local without changing Conductor
callers — only the router config changes.

---

## Observability (Conductor-owned)

Every decision, tool invocation, retry, approval, and artifact ref should be
durable. Start with structured JSONL / DB rows on the decision/task log; add
**OpenTelemetry** (+ optional Phoenix/Langfuse) when dashboards matter. Do not
block M2 on a full OTel stack.

---

## Non-goals (M2)

- Dynamic insertion of arbitrary research tasks (M3)
- Goal CLI UX (M3 Phase D)
- Specialist agents (M5)
- Replacing Engineer DAG walk with LLM next-task selection
- Adopting LangGraph/CrewAI as the Conductor