-- Knowledge Store schema (knowledge-system.md §4).
-- Ontology + a graph stored relationally: uniform research_artifacts, merged
-- knowledge objects (techniques/datasets/architectures/tasks), and link tables
-- (artifact_techniques + evidence_links) that model the reference edges.
--
-- Type-specific detail lives in `metadata` JSON; the query engine filters the
-- common interface first (SQL indexes), then walks joins for evidence.

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Layer 2 — one row per source (ResearchArtifact). List-valued fields are kept
-- as JSON here for lossless round-trip; querying goes through join tables.
CREATE TABLE IF NOT EXISTS research_artifacts (
    id               TEXT PRIMARY KEY,
    type             TEXT NOT NULL,
    source           TEXT NOT NULL,
    title            TEXT NOT NULL DEFAULT '',
    summary          TEXT NOT NULL DEFAULT '',
    confidence       REAL NOT NULL DEFAULT 0.5,
    competition_slug TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}',
    techniques       TEXT NOT NULL DEFAULT '[]',
    models           TEXT NOT NULL DEFAULT '[]',
    datasets         TEXT NOT NULL DEFAULT '[]',
    claims           TEXT NOT NULL DEFAULT '[]',
    refs             TEXT NOT NULL DEFAULT '[]',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON research_artifacts(type);
CREATE INDEX IF NOT EXISTS idx_artifacts_comp ON research_artifacts(competition_slug);

-- Knowledge Hub processing receipt. The fingerprint changes whenever the
-- stored artifact changes; signature changes when hub semantics/config change.
-- Missing or mismatched rows are pending ingestion.
CREATE TABLE IF NOT EXISTS artifact_ingestions (
    artifact_id TEXT PRIMARY KEY REFERENCES research_artifacts(id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    signature   TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifact_ingestions_signature
    ON artifact_ingestions(signature);

-- Layer 3 — merged knowledge objects (one per concept across many sources).
CREATE TABLE IF NOT EXISTS techniques (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    category     TEXT NOT NULL DEFAULT '',
    domain       TEXT NOT NULL DEFAULT '',
    summary      TEXT NOT NULL DEFAULT '',
    known_issues TEXT NOT NULL DEFAULT '',
    confidence   REAL NOT NULL DEFAULT 0.5,
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    domain     TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS architectures (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    domain     TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    domain     TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Reference edge: any artifact ↔ a merged technique. paper_techniques /
-- experiment_techniques are views over this table filtered by artifact type.
CREATE TABLE IF NOT EXISTS artifact_techniques (
    artifact_id  TEXT NOT NULL REFERENCES research_artifacts(id) ON DELETE CASCADE,
    technique_id TEXT NOT NULL REFERENCES techniques(id) ON DELETE CASCADE,
    relation     TEXT NOT NULL DEFAULT 'mentions',
    weight       REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (artifact_id, technique_id)
);
CREATE INDEX IF NOT EXISTS idx_arttech_tech ON artifact_techniques(technique_id);

-- Generic evidence links (knowledge-system.md `references`): artifact ↔
-- technique / finding / hypothesis / dataset / … Named evidence_links because
-- `references` is a reserved word in SQL.
CREATE TABLE IF NOT EXISTS evidence_links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT,
    target_kind TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    relation    TEXT NOT NULL DEFAULT 'supports',
    weight      REAL NOT NULL DEFAULT 1.0,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_target ON evidence_links(target_kind, target_id);

CREATE TABLE IF NOT EXISTS experiments (
    id               TEXT PRIMARY KEY,
    competition_slug TEXT,
    summary          TEXT NOT NULL DEFAULT '',
    outcome          TEXT NOT NULL DEFAULT 'unknown',
    metrics          TEXT NOT NULL DEFAULT '{}',
    techniques       TEXT NOT NULL DEFAULT '[]',
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_comp ON experiments(competition_slug);

CREATE TABLE IF NOT EXISTS hypotheses (
    id               TEXT PRIMARY KEY,
    competition_slug TEXT,
    observation      TEXT NOT NULL DEFAULT '',
    prediction       TEXT NOT NULL DEFAULT '',
    rationale        TEXT NOT NULL DEFAULT '',
    expected_impact  REAL NOT NULL DEFAULT 0.0,
    confidence       REAL NOT NULL DEFAULT 0.5,
    status           TEXT NOT NULL DEFAULT 'suggested',
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hypotheses_comp ON hypotheses(competition_slug);

CREATE TABLE IF NOT EXISTS beliefs (
    id               TEXT PRIMARY KEY,
    competition_slug TEXT,
    technique        TEXT NOT NULL DEFAULT '',
    effect           TEXT NOT NULL DEFAULT 'unknown',
    status           TEXT NOT NULL DEFAULT 'suggested',
    confidence       REAL NOT NULL DEFAULT 0.5,
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_beliefs_comp ON beliefs(competition_slug);

CREATE TABLE IF NOT EXISTS findings (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL DEFAULT '',
    finding       TEXT NOT NULL,
    applicability TEXT NOT NULL DEFAULT '[]',
    confidence    REAL NOT NULL DEFAULT 0.5,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- Research Memory — designed now, populated in a Future plan.
CREATE TABLE IF NOT EXISTS ideas (
    id               TEXT PRIMARY KEY,
    competition_slug TEXT,
    problem          TEXT NOT NULL DEFAULT '',
    summary          TEXT NOT NULL DEFAULT '',
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idea_links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id     TEXT NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    target_kind TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    relation    TEXT NOT NULL DEFAULT 'related',
    created_at  TEXT NOT NULL
);

-- Named relationship views over the uniform artifact↔technique edge, so callers
-- can ask "papers using technique X" / "experiments using technique X" directly.
CREATE VIEW IF NOT EXISTS paper_techniques AS
    SELECT at.artifact_id AS paper_id, at.technique_id, at.relation, at.weight
    FROM artifact_techniques at
    JOIN research_artifacts a ON a.id = at.artifact_id
    WHERE a.type = 'paper';

CREATE VIEW IF NOT EXISTS experiment_techniques AS
    SELECT at.artifact_id AS experiment_id, at.technique_id, at.relation, at.weight
    FROM artifact_techniques at
    JOIN research_artifacts a ON a.id = at.artifact_id
    WHERE a.type = 'experiment';

CREATE VIEW IF NOT EXISTS repository_techniques AS
    SELECT at.artifact_id AS repository_id, at.technique_id, at.relation, at.weight
    FROM artifact_techniques at
    JOIN research_artifacts a ON a.id = at.artifact_id
    WHERE a.type = 'repository';
