from pathlib import Path

import pytest

from labpilot.research_engine.intelligence.knowledge.sources import RawStore
from labpilot.research_engine.intelligence.knowledge.store import (
    KnowledgeStore,
    technique_id,
)
from labpilot.research_engine.intelligence.models import (
    ResearchArtifact,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.paths import ResearchPaths


def _artifact(id_: str, type_: ResearchArtifactType, **kw) -> ResearchArtifact:
    return ResearchArtifact(id=id_, type=type_, source=kw.pop("source", "test"), **kw)


# --- Layout / tree ----------------------------------------------------------


def test_creating_store_yields_locked_tree(tmp_path: Path):
    store = KnowledgeStore(tmp_path / "knowledge", "birdclef-2026")
    paths = ResearchPaths(tmp_path / "knowledge", "birdclef-2026")

    for directory in (
        paths.raw_dir,
        paths.extracted_dir,
        paths.knowledge_dir,
        paths.experiments_dir,
        paths.reports_dir,
        paths.embeddings_dir,
    ):
        assert directory.is_dir()
    assert (paths.raw_dir / "papers").is_dir()
    assert (paths.raw_dir / "discussions").is_dir()
    assert (paths.raw_dir / "kernels").is_dir()
    assert (paths.raw_dir / "competitions").is_dir()
    assert (paths.extracted_dir / "forums").is_dir()
    assert (paths.knowledge_dir / "techniques").is_dir()
    assert paths.db_path.is_file()
    assert store.schema_version() == "2"
    store.close()


def test_research_paths_match_analyze_context(tmp_path: Path):
    from labpilot.research_engine.intelligence.context import build_context

    ctx = build_context(
        "birdclef-2026", runs_dir=tmp_path / "runs", knowledge_dir=tmp_path / "knowledge"
    )
    paths = ResearchPaths(tmp_path / "knowledge", "birdclef-2026")
    assert ctx.research_dir == paths.root
    assert ctx.report_path == paths.report_path


# --- Artifact upsert --------------------------------------------------------


def test_upsert_artifact_writes_row_and_extracted_json(tmp_path: Path):
    with KnowledgeStore(tmp_path / "knowledge", "birdclef-2026") as store:
        art = _artifact(
            "paper:123",
            ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="SpecAugment",
            techniques=["SpecAugment", "EMA"],
            claims=["+1.2% macro f1"],
            confidence=0.8,
        )
        store.upsert_artifact(art)

        fetched = store.get_artifact("paper:123")
        assert fetched is not None
        assert fetched.techniques == ["SpecAugment", "EMA"]
        assert fetched.competition_slug == "birdclef-2026"  # defaulted from store

        extracted = tmp_path / "knowledge/birdclef-2026/research/extracted/papers/paper_123.json"
        assert extracted.is_file()


def test_upsert_artifact_is_idempotent_update(tmp_path: Path):
    with KnowledgeStore(tmp_path / "knowledge", "c") as store:
        store.upsert_artifact(_artifact("paper:1", ResearchArtifactType.PAPER, title="v1"))
        store.upsert_artifact(_artifact("paper:1", ResearchArtifactType.PAPER, title="v2"))
        assert store.count("research_artifacts") == 1
        assert store.get_artifact("paper:1").title == "v2"


def test_list_artifacts_filters_by_type(tmp_path: Path):
    with KnowledgeStore(tmp_path / "knowledge", "c") as store:
        store.upsert_artifact(_artifact("paper:1", ResearchArtifactType.PAPER))
        store.upsert_artifact(_artifact("repo:o/r", ResearchArtifactType.REPOSITORY))
        store.upsert_artifact(_artifact("exp:9", ResearchArtifactType.EXPERIMENT))
        assert [a.id for a in store.list_artifacts(type=ResearchArtifactType.PAPER)] == ["paper:1"]
        assert len(store.list_artifacts()) == 3


# --- Technique merge + joins (flagship) -------------------------------------


def test_technique_id_is_deterministic():
    assert technique_id("SpecAugment") == technique_id("specaugment")
    assert technique_id("Focal Loss") == "tech_focal_loss"


def test_merge_technique_and_join_to_papers_and_experiments(tmp_path: Path):
    with KnowledgeStore(tmp_path / "knowledge", "birdclef-2026") as store:
        store.upsert_artifact(_artifact("paper:1", ResearchArtifactType.PAPER))
        store.upsert_artifact(_artifact("paper:2", ResearchArtifactType.PAPER))
        store.upsert_artifact(_artifact("exp:12", ResearchArtifactType.EXPERIMENT))
        store.upsert_artifact(_artifact("repo:o/r", ResearchArtifactType.REPOSITORY))

        tid = store.merge_technique(
            "SpecAugment",
            category="augmentation",
            domain="audio",
            confidence=0.9,
            evidence=["paper:1", "paper:2", "exp:12", "repo:o/r"],
        )
        assert tid == "tech_specaugment"

        papers = store.artifacts_for_technique(tid, type=ResearchArtifactType.PAPER)
        experiments = store.artifacts_for_technique(tid, type=ResearchArtifactType.EXPERIMENT)
        assert papers == ["paper:1", "paper:2"]
        assert experiments == ["exp:12"]
        assert set(store.artifacts_for_technique(tid)) == {
            "paper:1",
            "paper:2",
            "exp:12",
            "repo:o/r",
        }
        assert store.techniques_for_artifact("paper:1") == [tid]


def test_merge_technique_upserts_and_keeps_max_confidence(tmp_path: Path):
    with KnowledgeStore(tmp_path / "knowledge", "c") as store:
        store.merge_technique("EMA", confidence=0.4, summary="first")
        store.merge_technique("EMA", confidence=0.7)
        assert store.count("techniques") == 1
        tech = store.get_technique(technique_id("EMA"))
        assert tech["confidence"] == 0.7
        assert tech["summary"] == "first"  # preserved when new is empty


def test_paper_techniques_view(tmp_path: Path):
    with KnowledgeStore(tmp_path / "knowledge", "c") as store:
        store.upsert_artifact(_artifact("paper:1", ResearchArtifactType.PAPER))
        store.upsert_artifact(_artifact("exp:1", ResearchArtifactType.EXPERIMENT))
        tid = store.merge_technique("Mixup", evidence=["paper:1", "exp:1"])
        rows = store._conn.execute(
            "SELECT paper_id FROM paper_techniques WHERE technique_id = ?", (tid,)
        ).fetchall()
        assert [r["paper_id"] for r in rows] == ["paper:1"]


def test_evidence_link_insert(tmp_path: Path):
    with KnowledgeStore(tmp_path / "knowledge", "c") as store:
        store.upsert_artifact(_artifact("paper:1", ResearchArtifactType.PAPER))
        store.add_evidence_link(
            target_kind="hypothesis", target_id="H-001", artifact_id="paper:1"
        )
        assert store.count("evidence_links") == 1


# --- Raw store: immutability / version append -------------------------------


def test_raw_write_once_then_no_overwrite(tmp_path: Path):
    raw = RawStore(tmp_path / "knowledge", "birdclef-2026")
    v1 = raw.write("papers", "attention.pdf", b"original", ext="pdf")
    assert v1.version == 1
    # second write without refresh does not overwrite; returns existing latest
    v_again = raw.write("papers", "attention.pdf", b"tampered", ext="pdf")
    assert v_again.version == 1
    assert raw.read("papers", "attention.pdf") == b"original"
    assert len(raw.versions("papers", "attention.pdf")) == 1


def test_raw_refresh_appends_new_version(tmp_path: Path):
    raw = RawStore(tmp_path / "knowledge", "c")
    raw.write("discussions", "thread-1", b"v1 body")
    v2 = raw.write("discussions", "thread-1", b"v2 body", refresh=True)
    assert v2.version == 2
    assert [v.version for v in raw.versions("discussions", "thread-1")] == [1, 2]
    assert raw.read("discussions", "thread-1", version=1) == b"v1 body"
    assert raw.read("discussions", "thread-1") == b"v2 body"


def test_raw_rejects_unknown_kind(tmp_path: Path):
    raw = RawStore(tmp_path / "knowledge", "c")
    with pytest.raises(ValueError):
        raw.write("videos", "x", b"data")


def test_raw_read_missing_returns_none(tmp_path: Path):
    raw = RawStore(tmp_path / "knowledge", "c")
    assert raw.read("papers", "nope") is None
    assert raw.latest("papers", "nope") is None
