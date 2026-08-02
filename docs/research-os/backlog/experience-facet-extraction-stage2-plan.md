# Plan — Experience facet extraction Stage 2

Back to [experience-facet-extraction.md](experience-facet-extraction.md) ·
[backlog-grooming.md](backlog-grooming.md).

**Status:** Done (Stage 2 + histogram logging)  

**Scope:** **Stage 2** — artifact-aware extractors + merge + confidence histogram logs.  
**Deferred (explicit triggers):**

| Stage | Start when |
|-------|------------|
| 3 Embeddings | ContextBundle BM25 misses cross-comp paraphrases ([hybrid-semantic-retrieval](hybrid-semantic-retrieval.md)) |
| 4 LLM ambiguous | Histogram shows a clear share of extracts below threshold **and** those misses hurt seed/inspect |
| 5 Facet graph | Enough evidenced edges; Kuzu only if SQL hurts |

**Principle:** Evidence-backed hints, not a taxonomy product. Wrong extraction
must remain recoverable (confidence + evidence + source).

---

## Goal

Replace the monolithic keyword path inside `ExperienceExtractor._facets` with a
small **FacetExtractor pipeline**: each source extractor emits
`ExperienceFacet` hits; a merger aggregates confidence + unions evidence.

```text
ExperienceExtractor.extract
        |
        v
FacetPipeline.extract(context)
        |
        +-- MetadataExtractor   (competition, problem_type, description)
        +-- CodeExtractor       (workspace pipeline imports / APIs)
        +-- DatasetExtractor    (file extensions / loaders when available)
        +-- PaperExtractor      (linked paper titles/claims when available)
        +-- ResultExtractor     (metrics narrative / reflection / comparison)
        |
        v
FacetMerger.merge(hits) → list[ExperienceFacet]
        |
        v
ExperienceRecord.facets (unchanged SoR shape)
```

---

## Roles and responsibilities

| Role | Owns | Does **not** own |
|------|------|------------------|
| **FacetPipeline** | Orchestrate extractors; pass bounded `FacetContext`; return sorted facets | Persist Experience Records; Conductor policy |
| **MetadataExtractor** | Competition slug, `problem_type`, description, technique field | Reading workspace files |
| **CodeExtractor** | Bounded scan of workspace `pipeline/` (imports, known APIs) | Training / executing code |
| **DatasetExtractor** | Extensions / loader hints from workspace data paths or experiment metadata | Downloading datasets |
| **PaperExtractor** | Soft signals from optional paper titles/abstracts already on the experiment / knowledge artifacts | Live literature search |
| **ResultExtractor** | Reflection observation/cause, comparison verdict text, metric keys | Belief / claim promotion |
| **FacetMerger** | Per-facet aggregation (max confidence + evidence union + source priority) | Inventing new facet names beyond extractor outputs |
| **ExperienceExtractor** | Build record fields; call `FacetPipeline`; upsert via store | Facet hint tables / merge math |
| **ExperienceStore / Context ExperienceProvider** | Persist & retrieve `facets` as today | Changing facet schema (Stage 2 keeps `ExperienceFacet`) |

**Source priority (merge ties):**  
`metadata` > `code` > `dataset` > `paper` > `result` > `rules` > `legacy`

---

## What is new

| New | Purpose |
|-----|---------|
| `memory/facets/__init__.py` | Public exports: `FacetPipeline`, `FacetContext`, extractors |
| `memory/facets/context.py` | `FacetContext` dataclass (competition, payload, texts, optional `workspace_path`, optional paper snippets) |
| `memory/facets/base.py` | `FacetExtractor` protocol / ABC: `extract(ctx) -> list[ExperienceFacet]` |
| `memory/facets/merge.py` | `FacetMerger` (move/extend `_merge_facet_hits` + source priority) |
| `memory/facets/pipeline.py` | Run extractors (soft-fail each), merge, sort by confidence |
| `memory/facets/metadata.py` | Today’s metadata + problem_type path |
| `memory/facets/rules.py` | Today’s modality/technique keyword hints (`source="rules"`) — kept as one extractor |
| `memory/facets/code.py` | Import/API needles (`librosa`, `MelSpectrogram`, `timm`, `xgboost`, …) |
| `memory/facets/dataset.py` | Extension / loader needles (`.wav`, `.flac`, `.jpg`, `.parquet`, …) |
| `memory/facets/paper.py` | Optional; empty list when no paper text in context |
| `memory/facets/result.py` | Reflection / comparison / metric-key hints |

Extend `FacetSource` literal in `models.py` to include:  
`metadata | rules | code | dataset | paper | result | legacy`.

---

## Impacted files (existing)

| File | Change |
|------|--------|
| [`memory/models.py`](../../../src/labpilot/research_engine/memory/models.py) | Widen `FacetSource` |
| [`memory/extractor.py`](../../../src/labpilot/research_engine/memory/extractor.py) | Delete `_MODALITY_HINTS` / `_facets` / local merge; call `FacetPipeline` |
| [`memory/__init__.py`](../../../src/labpilot/research_engine/memory/__init__.py) | Optional re-export of pipeline types |
| [`context/providers/experience.py`](../../../src/labpilot/research_engine/context/providers/experience.py) | Prefer including top evidence snippets in item text (optional boost); keep `facet_names()` for filters |
| [`experience-facet-extraction.md`](experience-facet-extraction.md) | Mark Stage 2 exit criteria when done |
| Write hooks / CLI | **No API change** — they already consume `ExperienceRecord.facets` |

**Unchanged SoR:** `experiences.db` `tags` JSON column still stores facet objects.  
No schema migration if JSON already holds `ExperienceFacet` dicts (Stage 1).

---

## Merge policy (locked for Stage 2)

For the same `facet` name:

1. **confidence** = `max(hits.confidence)` (simple; noisy-OR deferred)
2. **evidence** = de-duped union (preserve order: higher-confidence hit first)
3. **source** = highest-priority source among hits (table above)
4. Cap evidence list length (e.g. 8) so ContextBundle text stays bounded

---

## CodeExtractor / DatasetExtractor bounds

- Scan at most N files under `workspace_path/pipeline` (and optional `src/`), each truncated (e.g. 40KB) — same spirit as `LocalCodeProfiler`.
- Soft-fail on missing workspace: return `[]`, never raise into extract pipeline.
- No network; no LLM.

---

## Test coverage

| Test file | Cases |
|-----------|--------|
| `tests/unit/test_facet_pipeline.py` | Pipeline merges multi-extractor hits; soft-fail extractor does not abort; sort by confidence |
| `tests/unit/test_facet_metadata_rules.py` | Metadata `problem_type=audio` → high conf + source metadata; rules keyword path still emits evidence; low-signal → low conf |
| `tests/unit/test_facet_code_extractor.py` | Fixture `pipeline/train.py` with `librosa` / `MelSpectrogram` → `audio`; no false hit on empty tree |
| `tests/unit/test_facet_dataset_extractor.py` | Paths/names with `.wav` → audio; `.parquet` → tabular; missing data root → `[]` |
| `tests/unit/test_facet_result_extractor.py` | Reflection “minority classes” → imbalance; comparison text contributes evidence |
| `tests/unit/test_facet_merger.py` | Max confidence; evidence union; source priority (`code` beats `rules`); evidence cap |
| `tests/unit/test_experience_extractor.py` | Update: still produces facets via pipeline; workspace with code boosts audio evidence/source |
| `tests/unit/test_experience_provider.py` | Still surfaces cross-comp items when facets come from code/dataset (smoke) |

**CI:** default slice  
`uv run pytest -m "not llm and not image and not deep"`  
(no LLM tests for Stage 2).

---

## Acceptance (Stage 2)

- [x] `ExperienceExtractor` no longer owns keyword tables; uses `FacetPipeline`
- [x] At least Metadata + Rules + Code + Result extractors registered; Dataset/Paper may no-op when inputs missing
- [x] Merge policy documented and unit-tested
- [x] `FacetSource` includes new sources; store round-trip unchanged
- [x] Existing M6 memory/CLI/provider tests green
- [x] Backlog Stage 2 exit criteria checked off in [experience-facet-extraction.md](experience-facet-extraction.md)
- [x] Confidence histogram logged (`experience_facet_confidence_histogram`) for Stage 4 gating

---

## Follow-ons (explicitly later)

| Stage | Trigger to start |
|-------|------------------|
| 3 Embeddings | BM25 gaps measured; [hybrid-semantic-retrieval](hybrid-semantic-retrieval.md) |
| 4 LLM ambiguous | Stage 2 live + confidence histogram shows many mid-band misses |
| 5 Facet graph | Enough evidenced edges; [kuzu-graph-backend](kuzu-graph-backend.md) only if SQL hurts |

---

## Non-goals

- Taxonomy wiki / first-class prompt-HP-paper tables
- Auto warm-start from facets ([automatic-transfer-confidence](automatic-transfer-confidence.md))
- Splitting oversized modules outside `memory/extractor.py` facet path
- Conductor or campaign behavior changes
