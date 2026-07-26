# Research Reflection — Schema

Back to [README](README.md). Extends `accessor/sqlite/schema.sql`
(`SCHEMA_VERSION` = `5` after Plan 1).

---

## 1. Design principles

- Additive DDL only; no destructive renames of existing tables.
- Audit trail for belief mutations (`belief_updates`).
- Claims are first-class; beliefs remain short-horizon working assumptions.
- Align hypothesis status vocabulary across file SoR and SQLite.

---

## 2. Status vocabulary alignment

| Object | Values | Notes |
|--------|--------|-------|
| Hypothesis (file / CLI) | `proposed`, `testing`, `confirmed`, `rejected`, `inconclusive` | Experiment Scientist SoR |
| Hypothesis (SQLite `hypotheses.status`) | historically `suggested` | Plan 1: accept both; prefer `proposed`; map `suggested` ↔ `proposed` on read/write |
| Belief (`beliefs.status`) | `suggested`, `validated`, `established`, `rejected` (existing Knowledge Hub) | Unchanged |
| Claim | `candidate`, `supported`, `contested`, `withdrawn` | New |
| Evidence strength | `strong`, `moderate`, `weak`, `rejected` | Journal projection |

---

## 3. New tables

### 3.1 `experiment_evidence`

One structured evidence blob per execution (or per comparison), durable for
critic input and journal.

```sql
CREATE TABLE IF NOT EXISTS experiment_evidence (
    id               TEXT PRIMARY KEY,           -- EE-xxx
    competition_slug TEXT,
    execution_id     TEXT,                       -- E-xxx (nullable for legacy)
    experiment_id    TEXT,                       -- experiments.id / run id
    plan_id          TEXT,
    hypothesis_id    TEXT,
    metrics          TEXT NOT NULL DEFAULT '{}', -- JSON
    config_summary   TEXT NOT NULL DEFAULT '{}', -- JSON
    runtime_summary  TEXT NOT NULL DEFAULT '{}', -- JSON
    comparison       TEXT NOT NULL DEFAULT '{}', -- JSON (vs baseline / prior)
    strength         TEXT NOT NULL DEFAULT 'moderate', -- strong|moderate|weak|rejected
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
```

### 3.2 `belief_updates`

Append-only audit of belief confidence/status changes.

```sql
CREATE TABLE IF NOT EXISTS belief_updates (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id        TEXT NOT NULL REFERENCES beliefs(id),
    competition_slug TEXT,
    execution_id     TEXT,
    experiment_id    TEXT,
    prior_confidence REAL NOT NULL,
    new_confidence   REAL NOT NULL,
    prior_status     TEXT NOT NULL DEFAULT '',
    new_status       TEXT NOT NULL DEFAULT '',
    reason           TEXT NOT NULL DEFAULT '',
    evidence_id      TEXT,                       -- experiment_evidence.id
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL
);
```

### 3.3 `lessons`

Cross-competition research memory (durable takeaways).

```sql
CREATE TABLE IF NOT EXISTS lessons (
    id               TEXT PRIMARY KEY,           -- L-xxx
    competition_slug TEXT,                       -- null = cross-competition
    summary          TEXT NOT NULL,
    category         TEXT NOT NULL DEFAULT '',   -- technique|process|pitfall|…
    confidence       REAL NOT NULL DEFAULT 0.5,
    source_execution TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
```

### 3.4 `research_claims` + `claim_evidence`

Synthesized conclusions the system will stand behind.

```sql
CREATE TABLE IF NOT EXISTS research_claims (
    id               TEXT PRIMARY KEY,           -- C-xxx
    competition_slug TEXT,
    statement        TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'candidate',
    -- candidate|supported|contested|withdrawn
    confidence       REAL NOT NULL DEFAULT 0.5,
    technique        TEXT NOT NULL DEFAULT '',
    effect           TEXT NOT NULL DEFAULT '',
    promoted_from    TEXT,                       -- belief id if promoted
    contradictions   TEXT NOT NULL DEFAULT '[]', -- JSON list of claim/evidence ids
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id    TEXT NOT NULL REFERENCES research_claims(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL,                   -- experiment_evidence.id or artifact
    relation    TEXT NOT NULL DEFAULT 'supports', -- supports|contradicts
    weight      REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (claim_id, evidence_id, relation)
);
```

---

## 4. Existing tables (reuse)

| Table | Use |
|-------|-----|
| `beliefs` | Fast-moving working assumptions; BeliefUpdater mutates |
| `hypotheses` | HypothesisEvaluator mutates status + metadata.why |
| `experiments` | Link from evidence / execution |
| `evidence_links` | Generic edges; claim graph may also use `claim_evidence` |
| `research_executions` | Source of execution_id |
| `ideas` / `idea_links` | Optional later link from lessons |

---

## 5. File SoR dual-write (transitional)

- Hypothesis file store (`experiments/hypothesis.py`) remains authoritative for
  CLI `hypothesize` until Plan 4 unifies write path through reflection.
- KnowledgeHub file KB: reflection owns post-run writers; dual-write briefly in
  Plans 4–6 if needed.

---

## 6. Migration

- Bump `SCHEMA_VERSION` in `accessor/sqlite/migrate.py`.
- Idempotent `CREATE TABLE IF NOT EXISTS` in `schema.sql`.
- Unit tests: migrate v4 DB → new version; insert evidence + belief_update.
