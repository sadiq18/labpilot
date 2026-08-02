# Backlog — Evidence-backed experience facet extraction

**Status:** Backlog (stages 2–5). **Stage 1 shipped** in
[`ExperienceExtractor`](../../milestones/06-transfer-memory/plan-2-experience-extractor.md)
(`ExperienceFacet`: confidence + evidence + source). Stages 2+ remain future work.

**Stage 2 design:** [experience-facet-extraction-stage2-plan.md](experience-facet-extraction-stage2-plan.md)
(roles, impacted files, new modules, tests).

**Principle:** Research OS memory wants **evidence-backed extraction**, not a
fixed taxonomy. Wrong extraction must be recoverable; LLMs later reason over
evidence, not over opaque labels.

Related: [experience-pattern-extraction](experience-pattern-extraction.md)
(emergent pattern libraries), [hybrid-semantic-retrieval](hybrid-semantic-retrieval.md)
(embeddings), [automatic-transfer-confidence](automatic-transfer-confidence.md).

---

## Problem

Keyword matching cannot:

- Express uncertainty (BirdCLEF ≠ always “audio” with certainty 1.0)
- Cite *why* a facet was chosen
- Transfer to competitions with no shared keywords (“underwater sound” ↔ BirdCLEF)
- Grow relationships (SpecAugment → useful_for → audio) without a wiki product

---

## Evolution stages

### Stage 1 — Rule extractor → confidence + evidence — **Done**

Keep hints, stop treating matches as truth. Stored as `ExperienceRecord.facets`:

```json
{
  "facet": "audio",
  "confidence": 0.82,
  "evidence": ["mel spectrogram", "BirdCLEF", "wav"],
  "source": "metadata"
}
```

**Why:** LLM can later reason over evidence; wrong extraction is recoverable.

**Exit criteria:**

- [x] Experience Record uses structured `facets` (legacy string tags coerce on read)
- [x] Each hit has `confidence`, `evidence[]`, `source`
- [x] Unit tests: keyword path produces evidence; low-signal → low confidence

### Stage 2 — Artifact-aware extractors — **Done**

Design: [experience-facet-extraction-stage2-plan.md](experience-facet-extraction-stage2-plan.md).

Different sources reveal different signals.

| Source | Signals | Example |
|--------|---------|---------|
| Competition metadata | title, description | BirdCLEF → audio |
| Code | imports, APIs | `librosa`, `MelSpectrogram` → audio |
| Model | architecture names | EfficientNet → image |
| Dataset | extensions / loaders | `.wav` / `.jpg` / `.parquet` |
| Papers / results | claims, metrics narrative | modality / technique hints |

```text
FacetExtractor
  ├── MetadataExtractor
  ├── CodeExtractor
  ├── DatasetExtractor
  ├── PaperExtractor
  └── ResultExtractor
```

Merge with confidence aggregation (max / noisy-OR / calibrated weights — pick at impl).

**Exit criteria:**

- [x] `ExperienceExtractor` uses `FacetPipeline` (no inline keyword tables)
- [x] Metadata + Rules + Code + Dataset + Paper + Result extractors registered
- [x] Merge policy unit-tested; confidence histogram logged per extract
- [x] Stage 4 deferred until mid/low band share hurts seed/inspect
- [x] Stage 3 deferred until BM25 cross-comp paraphrase misses

### Stage 3 — Embeddings for similarity, not classification

**Trigger:** ContextBundle BM25 misses cross-comp paraphrases — see
[hybrid-semantic-retrieval](hybrid-semantic-retrieval.md).

Do **not** ask “Is this audio?”  
Ask “What previous research experiences look similar?”

Example: new competition “Underwater sound classification” has no keyword `audio`,
but embedding neighbors include BirdCLEF / whale detection / bioacoustics →
transfer memory still works.

Depends on experience corpus size + [hybrid-semantic-retrieval](hybrid-semantic-retrieval.md)
signals. Prefer similarity into ContextBundle over hard modality labels.

### Stage 4 — LLM extraction for ambiguous cases only

**Trigger:** Stage 2 confidence histograms show a clear share of extracts in the
low/mid band **and** those misses hurt `research memory seed|inspect`.

```text
Fast extractor
      |
      v
confidence < threshold?
      |
      v
LLM extraction
      |
      v
merge evidence
```

Example output:

```json
{
  "modality": "multimodal",
  "confidence": 0.91,
  "reason": "Uses image encoder with text captions",
  "evidence": ["…"]
}
```

Non-goals: LLM on every extract; taxonomy enforcement via prompt.

### Stage 5 — Research Facet Graph

Do not ship a fixed taxonomy product. Store growing relationships:

```text
Experiment --uses--> Technique(SpecAugment)
Technique --useful_for--> Facet(Audio)
Technique --useful_for--> Facet(Imbalance)
```

Graph grows from evidenced extracts + experiment outcomes. Align with Evidence Card /
Research Graph ports; Kuzu only when [kuzu-graph-backend](kuzu-graph-backend.md)
signals justify it.

---

## Non-goals (keep deferred)

- Replacing Experience Record SoR with a facet wiki
- Hardcoding prompt / HP / paper category tables as first-class stores
- Stage 3–5 as M6 exit criteria (Plan 2 MVP stays keyword + flat tags)

## Out of scope here

M6 Plans 3–6 (context provider, CLI, write hooks, capstone) — they consume today’s
flat tags; migrate call sites when Stage 1 lands.
