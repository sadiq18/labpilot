# Spike notes — Kaggle discussion & kernel access

Back to [spike-kaggle-discussions.md](spike-kaggle-discussions.md).  
**Date:** 2026-07-26. **Package inspected:** installed `kaggle` / `kagglesdk` under LabPilot `.venv`.

---

## Summary recommendation

| Surface | Verdict | Mechanism |
|---------|---------|-----------|
| Competition **kernels / notebooks** (votes + score sorts) | **GO** | Official `KaggleApi.kernels_list` + `kernels_pull` |
| Competition **forum discussions** (vote/top sort) | **GO** | Official `KaggleApi.competition_list_topics` + `competition_list_topic_messages` (and/or `forums_topic_show`) |
| Related-competition search | **Conditional GO** | `competitions_list(search=…)` already wrapped indirectly; keep SeriesRelated year-decrement as primary |
| Winning writeups / solution dumps | **NO-GO (v1)** | Keep `NullWinningSolutionProvider`; no dedicated official “winning solutions” list API. Kernels sorted by `scoreDescending` / `voteCount` are a **proxy**, not a writeup catalog. Do not HTML-scrape writeups. |

**HTML / web-crawler:** **not required** for discussions or kernels given the official CLI/API surfaces above. Prefer API-only in LabPilot. Authenticated HTML remains a last resort only if an API regresses — gated by a future ToS re-check, not shipped now.

**Plan F:** Kaggle `DiscussionProvider` may proceed behind the content-type analyzer (never `KaggleForumAnalyzer` as the plugin id). GitHub Issues can still ship in parallel.

---

## Kernels — evidence

Official docs (`kaggle kernels list`):

- `--competition <slug>`
- `--sort-by voteCount` ↔ UI “most votes”
- `--sort-by scoreDescending` ↔ UI “best scores”
- `--page` / `--page-size` (API caps page_size at 100)

Python:

```text
KaggleApi.kernels_list(..., competition=slug, sort_by="voteCount"|"scoreDescending", page=…, page_size=…)
KaggleApi.kernels_pull(kernel_ref, path=…, metadata=True)
```

`ApiKernelMetadata` exposes `ref`, `title`, `author`, `slug`, `total_votes`, `language`, `kernel_type`, `current_version_number` (no dedicated public-score field on the list object — score sort is server-side).

Auth: existing `KAGGLE_API_TOKEN` / OAuth / legacy key — same as download/submit.

**LabPilot action:** wrap list+pull on `KaggleClient`; persist as `ResearchArtifact(type=repository, source=kaggle)` with `metadata.kind="kaggle_kernel"`.

---

## Discussions — evidence

Official CLI (`kaggle competitions topics list`):

- Lists competition forum topics
- Sort: `hot` | `top` | `new` | `recent` | `active` | `relevance`
- UI “sort=votes” maps to API **`top`**

Python:

```text
KaggleApi.competition_list_topics(competition, sort_by="top", page=…)
KaggleApi.competition_list_topic_messages(competition, topic_id, page_size=-1)
# alternate full topic: forums_topic_show(topic_id)
```

`ApiCompetitionTopic`: `id`, `title`, `topic_url`, `author_name`, `comment_count`, `votes`, `post_date`, …

`ApiTopicMessage`: `content` / `raw_markdown`, nested `replies`.

Also documented OAuth scopes: `forum_topics.get`, `forum_messages.get`.

**LabPilot action:** wrap list+messages on `KaggleClient`; persist as `ResearchArtifact(type=discussion, source=kaggle)`. Soft-fail with `unavailable` + reason if auth/API fails — never invent empty “success”.

**Note:** `kernels topics list` is **per-kernel** commentary, not the competition forum. Out of scope for this spike’s competition discussion fetch.

---

## ToS / robots / caching

- Prefer **official authenticated API** (Kaggle Public API docs + CLI). Rate limits: respect HTTP 429 / “Too many requests”; LabPilot should pace and soft-fail.
- Do **not** scrape the SPA HTML for `/competitions/.../discussion` or `/code` while the official topics/kernels APIs work.
- Cache: immutable `RawStore` versions under `research/raw/kernels/` and `research/raw/discussions/`; re-fetch only with `--refresh`.

---

## Related competitions & winning solutions

- Related: keep API metadata / series heuristics; optional `competitions_list(search=)` remains unused in production — may wire later without HTML.
- Winning solutions: stay on `NullWinningSolutionProvider` until a ToS-safe official dump exists. High-score kernels via `research fetch --source kernels --sort score` are **not** claimed as “winning solutions” in `analyze.json`.

---

## Go / no-go for Plan F Kaggle provider

**GO** — implement Kaggle discussion fetch via official topics API; Plan F may register a `DiscussionProvider` that reads store-backed artifacts or calls the same client. No HTML scrape required for v1.
