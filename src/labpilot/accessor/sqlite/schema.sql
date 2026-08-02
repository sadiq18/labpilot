-- Unified SQLite schema — single source of record for `knowledge.db`.
--
-- Owns the intelligence ontology (research_artifacts, merged knowledge objects,
-- link tables), experiment/hypothesis/belief mirrors, AND the Research Planner
-- entities (research_plans / research_tasks / research_task_deps). One schema,
-- one migrator (accessor/sqlite/migrate.py); pillars reach it via SqliteClient.
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

-- Layer-3 knowledge ontology: named ML task/problem types (e.g. "Audio
-- Classification"). NOTE: this is NOT a planner execution node — planner DAG
-- nodes live in `research_tasks`. Do not overload this table.
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

-- ==========================================================================
-- Research Planner — compiled plans and their task DAGs.
-- research_plans: one compiled plan (usually from one hypothesis).
-- research_tasks: DAG nodes (WRITE_CODE, RUN_TRAINING, …); the planner never
--   performs the side effect — a future executor dispatches on task_type.
-- research_task_deps: DAG edges (task_id depends on depends_on).
-- research_executions: one run attempt of a plan (Research Engineer).
-- ==========================================================================
CREATE TABLE IF NOT EXISTS research_plans (
    id                 TEXT PRIMARY KEY,
    competition_slug   TEXT,
    hypothesis_id      TEXT,
    goal               TEXT NOT NULL DEFAULT '',
    current_state      TEXT NOT NULL DEFAULT '',
    expected_outcome   TEXT NOT NULL DEFAULT '',
    status             TEXT NOT NULL DEFAULT 'draft',   -- draft|ready|in_progress|done|abandoned
    priority           INTEGER NOT NULL DEFAULT 0,
    estimated_gain     REAL NOT NULL DEFAULT 0.0,
    risk               TEXT NOT NULL DEFAULT '',
    estimated_cost     REAL,                            -- hook; null in early MVP
    estimated_duration TEXT,                            -- hook
    runtime_target     TEXT,                            -- hook: local|docker|kaggle|cpu|p100|a100
    success_criteria   TEXT NOT NULL DEFAULT '[]',      -- JSON list
    rollback           TEXT NOT NULL DEFAULT '',
    artifacts          TEXT NOT NULL DEFAULT '[]',      -- JSON list
    generated_by       TEXT NOT NULL DEFAULT 'rule_engine',  -- llm|rule_engine
    metadata           TEXT NOT NULL DEFAULT '{}',
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_comp ON research_plans(competition_slug);
CREATE INDEX IF NOT EXISTS idx_plans_hyp  ON research_plans(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_plans_status ON research_plans(status);

CREATE TABLE IF NOT EXISTS research_tasks (
    id             TEXT PRIMARY KEY,
    plan_id        TEXT NOT NULL REFERENCES research_plans(id) ON DELETE CASCADE,
    parent_task_id TEXT REFERENCES research_tasks(id) ON DELETE SET NULL,
    task_type      TEXT NOT NULL,                       -- TaskType enum value
    description    TEXT NOT NULL DEFAULT '',
    inputs         TEXT NOT NULL DEFAULT '[]',          -- JSON list
    outputs        TEXT NOT NULL DEFAULT '[]',          -- JSON list
    status         TEXT NOT NULL DEFAULT 'pending',     -- pending|running|done|failed|skipped
    verification   TEXT NOT NULL DEFAULT '{}',          -- JSON TaskVerification
    retry_policy   TEXT NOT NULL DEFAULT '{}',          -- JSON RetryPolicy
    order_index    INTEGER NOT NULL DEFAULT 0,
    estimated_cost REAL,
    estimated_time TEXT,
    metadata       TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_tasks_plan ON research_tasks(plan_id);
CREATE INDEX IF NOT EXISTS idx_research_tasks_status ON research_tasks(status);

-- DAG edges: task_id depends on depends_on (must complete first).
CREATE TABLE IF NOT EXISTS research_task_deps (
    task_id    TEXT NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    depends_on TEXT NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    PRIMARY KEY (task_id, depends_on),
    CHECK (task_id != depends_on)
);
CREATE INDEX IF NOT EXISTS idx_research_task_deps_on ON research_task_deps(depends_on);

-- Research Engineer — durable execution attempts for a plan (E-xxx).
-- Task status remains on research_tasks (MVP: one active execution per plan).
CREATE TABLE IF NOT EXISTS research_executions (
    id                 TEXT PRIMARY KEY,          -- E-001
    plan_id            TEXT NOT NULL REFERENCES research_plans(id),
    competition_slug   TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    -- pending|running|succeeded|failed|cancelled
    workspace_path     TEXT,
    runtime_target     TEXT,
    experiment_id      TEXT,
    error              TEXT,
    metadata           TEXT NOT NULL DEFAULT '{}',
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    started_at         TEXT,
    completed_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_exec_plan ON research_executions(plan_id);
CREATE INDEX IF NOT EXISTS idx_exec_status ON research_executions(status);

-- ==========================================================================
-- Research Reflection — durable evidence, belief audit, lessons, claims.
-- ==========================================================================
CREATE TABLE IF NOT EXISTS experiment_evidence (
    id               TEXT PRIMARY KEY,
    competition_slug TEXT,
    execution_id     TEXT,
    experiment_id    TEXT,
    plan_id          TEXT,
    hypothesis_id    TEXT,
    metrics          TEXT NOT NULL DEFAULT '{}',
    config_summary   TEXT NOT NULL DEFAULT '{}',
    runtime_summary  TEXT NOT NULL DEFAULT '{}',
    comparison       TEXT NOT NULL DEFAULT '{}',
    strength         TEXT NOT NULL DEFAULT 'moderate',
    -- strong|moderate|weak|rejected
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_evidence_comp ON experiment_evidence(competition_slug);
CREATE INDEX IF NOT EXISTS idx_exp_evidence_exec ON experiment_evidence(execution_id);

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
    evidence_id      TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_belief_updates_belief ON belief_updates(belief_id);
CREATE INDEX IF NOT EXISTS idx_belief_updates_comp ON belief_updates(competition_slug);

CREATE TABLE IF NOT EXISTS lessons (
    id               TEXT PRIMARY KEY,
    competition_slug TEXT,
    summary          TEXT NOT NULL,
    category         TEXT NOT NULL DEFAULT '',
    confidence       REAL NOT NULL DEFAULT 0.5,
    source_execution TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lessons_comp ON lessons(competition_slug);

CREATE TABLE IF NOT EXISTS research_claims (
    id               TEXT PRIMARY KEY,
    competition_slug TEXT,
    statement        TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'candidate',
    -- candidate|supported|contested|withdrawn
    confidence       REAL NOT NULL DEFAULT 0.5,
    technique        TEXT NOT NULL DEFAULT '',
    effect           TEXT NOT NULL DEFAULT '',
    promoted_from    TEXT,
    contradictions   TEXT NOT NULL DEFAULT '[]',
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_comp ON research_claims(competition_slug);
CREATE INDEX IF NOT EXISTS idx_claims_status ON research_claims(status);

CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id    TEXT NOT NULL REFERENCES research_claims(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL,
    relation    TEXT NOT NULL DEFAULT 'supports',
    weight      REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (claim_id, evidence_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_ev ON claim_evidence(evidence_id);

-- ---------------------------------------------------------------------------
-- Research Conductor (M2) — OS session, task queue, decisions, operator feedback
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS os_sessions (
    id              TEXT PRIMARY KEY,
    competition     TEXT NOT NULL,
    goal            TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'running',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_os_sessions_comp ON os_sessions(competition);

CREATE TABLE IF NOT EXISTS os_tasks (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES os_sessions(id) ON DELETE CASCADE,
    tool_name       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    priority        INTEGER NOT NULL DEFAULT 0,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 1,
    args_json       TEXT NOT NULL DEFAULT '{}',
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    artifact_refs_json TEXT NOT NULL DEFAULT '[]',
    error           TEXT,
    decision_id     TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_os_tasks_session ON os_tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_os_tasks_status ON os_tasks(status);

CREATE TABLE IF NOT EXISTS os_decisions (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES os_sessions(id) ON DELETE CASCADE,
    tool_name       TEXT,
    rationale       TEXT NOT NULL DEFAULT '',
    stop            INTEGER NOT NULL DEFAULT 0,
    args_json       TEXT NOT NULL DEFAULT '{}',
    observe_json    TEXT NOT NULL DEFAULT '{}',
    approval_json   TEXT,
    artifact_refs_json TEXT NOT NULL DEFAULT '[]',
    task_id         TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_os_decisions_session ON os_decisions(session_id);

CREATE TABLE IF NOT EXISTS os_operator_feedback (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES os_sessions(id) ON DELETE CASCADE,
    gated_tool      TEXT NOT NULL,
    decision        TEXT NOT NULL,
    comment         TEXT NOT NULL DEFAULT '',
    decision_id     TEXT,
    task_id         TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_os_feedback_session ON os_operator_feedback(session_id);

-- ---------------------------------------------------------------------------
-- Campaign Engine (M3) — suggestions + metrics for capability gaps
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS os_suggestions (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES os_sessions(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL DEFAULT 'no_capability',
    message         TEXT NOT NULL,
    context_json    TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_os_suggestions_session ON os_suggestions(session_id);

CREATE TABLE IF NOT EXISTS os_campaign_metrics (
    session_id      TEXT PRIMARY KEY REFERENCES os_sessions(id) ON DELETE CASCADE,
    tasks_failed    INTEGER NOT NULL DEFAULT 0,
    tasks_blocked   INTEGER NOT NULL DEFAULT 0,
    unmet_goal      INTEGER NOT NULL DEFAULT 0,
    human_interventions INTEGER NOT NULL DEFAULT 0,
    no_capability   INTEGER NOT NULL DEFAULT 0,
    submissions     INTEGER NOT NULL DEFAULT 0,
    llm_cost_usd    REAL NOT NULL DEFAULT 0.0,
    updated_at      TEXT NOT NULL
);

-- Capability gap ledger (cross-session aggregate of no_capability suggestions)
CREATE TABLE IF NOT EXISTS os_capability_gaps (
    gap_key         TEXT PRIMARY KEY,
    kind            TEXT NOT NULL DEFAULT 'no_capability',
    count           INTEGER NOT NULL DEFAULT 0,
    first_seen_at   TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    sample_contexts TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'open',
    promoted_tool   TEXT,
    decision_reason TEXT NOT NULL DEFAULT '',
    decided_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_os_capability_gaps_status
    ON os_capability_gaps(status);

CREATE TABLE IF NOT EXISTS os_capability_decisions (
    id              TEXT PRIMARY KEY,
    gap_key         TEXT NOT NULL REFERENCES os_capability_gaps(gap_key),
    decision        TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    promoted_tool   TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_os_capability_decisions_gap
    ON os_capability_decisions(gap_key);

-- Cross-competition Experience Records (shared experiences.db SoR).
-- Episodes: goal/hypothesis/action/result/outcome + artifact links.
-- Column ``tags`` stores JSON list of facet objects
-- ({facet, confidence, evidence, source}); legacy string tags still readable.
CREATE TABLE IF NOT EXISTS experience_records (
    id                  TEXT PRIMARY KEY,
    source_competition  TEXT NOT NULL,
    goal                TEXT NOT NULL DEFAULT '',
    hypothesis          TEXT NOT NULL DEFAULT '',
    hypothesis_id       TEXT,
    action              TEXT NOT NULL DEFAULT '',
    result              TEXT NOT NULL DEFAULT '',
    outcome             TEXT NOT NULL DEFAULT 'fail',
    artifacts           TEXT NOT NULL DEFAULT '{}',
    tags                TEXT NOT NULL DEFAULT '[]',
    idempotency_key     TEXT NOT NULL UNIQUE,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experience_source
    ON experience_records(source_competition);
CREATE INDEX IF NOT EXISTS idx_experience_outcome
    ON experience_records(outcome);
CREATE INDEX IF NOT EXISTS idx_experience_idempotency
    ON experience_records(idempotency_key);


