# Plan F — Forum Intelligence (Future)

Back to [Milestone 3](README.md). Design: README §6 · Spike.

**Status:** Future / not started. **Depends on:** Plan 1; Kaggle provider gated on
[spike-kaggle-discussions.md](spike-kaggle-discussions.md) go **or** ship GitHub Issues
provider first without waiting on Kaggle. **Unlocks:** Forum signal in retrieval / hyps.

---

## Goal

Ship `DiscussionAnalyzer` + `ForumKnowledgeExtractor` / `ForumAnalyzerAgent` and content-type
providers (GitHub Issues, then Kaggle if spike allows, later Reddit/blogs). Extract practical
knowledge: common mistakes, discoveries, dataset bugs, LB shakeups, OOD — **not** thread
summaries.

## Why this matters

Forums often carry failure modes absent from papers. Design is first-class; access is the
spike. Keeping this as Plan F prevents Kaggle ToS from blocking Phase 1.

## In scope

- `analyzers/discussions/` + providers package
- `micro_agents/forum_analyzer/` agent.py + skill.md
- Persist discussions raw/extracted; hub merge via Plan 8 patterns
- Soft-fail when no provider registered

## Out of scope

- Production Kaggle scrape before spike go
- Full-thread summarization product
- Blocking Plans 1–11

## Acceptance criteria

- At least one provider (recommend GitHub Issues first) can feed ForumKnowledge artifacts.
- `research analyze discussions <slug>` soft-fails clearly if no provider.
- Kaggle provider only after spike go/no-go allows it.

## Test plan

- Unit: extract ForumKnowledge from fixture thread JSON via rule_engine/Agent.
- Provider tests mocked.

## Review notes

- Content-type naming — never `KaggleForumAnalyzer` as the plugin id.
- origin=forum → Suggested beliefs only until local validation.
