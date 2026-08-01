"""Experience DB path resolution + flat client knowledge layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.memory import ExperienceStore
from labpilot.workspace import (
    EXPERIENCE_DB_ENV,
    EXPERIENCE_DB_FILENAME,
    USER_EXPERIENCE_DB,
    competition_data_root,
    migrate_nested_client_knowledge,
    resolve_experience_db_path,
    scaffold_workspace,
    update_workspace_experience_path,
)


def test_client_knowledge_is_flat(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "birdclef-2026", "birdclef-2026")
    root = competition_data_root(client.knowledge_dir, "birdclef-2026")
    assert root == client.knowledge_dir
    paths = ResearchPaths(client.knowledge_dir, "birdclef-2026").ensure()
    assert paths.root == client.knowledge_dir / "research"
    assert paths.db_path.is_relative_to(client.knowledge_dir)
    assert not (client.knowledge_dir / "birdclef-2026").exists()


def test_legacy_knowledge_keeps_slug_nest(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    root = competition_data_root(knowledge, "titanic")
    assert root == knowledge / "titanic"
    paths = ResearchPaths(knowledge, "titanic").ensure()
    assert paths.root == knowledge / "titanic" / "research"


def test_migrate_nested_client_knowledge(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "demo", "demo")
    nested_research = client.knowledge_dir / "demo" / "research" / "plans"
    nested_research.mkdir(parents=True)
    marker = nested_research / "P-001.json"
    marker.write_text("{}", encoding="utf-8")

    moved = migrate_nested_client_knowledge(client.knowledge_dir, "demo")
    assert "research" in moved
    assert (client.knowledge_dir / "research" / "plans" / "P-001.json").is_file()
    assert not (client.knowledge_dir / "demo").exists()

    # ensure() also migrates
    nested2 = client.knowledge_dir / "demo" / "hypotheses"
    nested2.mkdir(parents=True)
    (nested2 / "H-001.json").write_text("{}", encoding="utf-8")
    ResearchPaths(client.knowledge_dir, "demo").ensure()
    assert (client.knowledge_dir / "hypotheses" / "H-001.json").is_file()


def test_resolve_experience_db_parent_research_root(tmp_path: Path) -> None:
    research_root = tmp_path / "kaggle"
    client = scaffold_workspace(research_root / "birdclef-2026", "birdclef-2026")
    path = resolve_experience_db_path(workspace=client)
    assert path == research_root / EXPERIENCE_DB_FILENAME


def test_resolve_experience_db_yaml_override(tmp_path: Path) -> None:
    research_root = tmp_path / "kaggle"
    client = scaffold_workspace(research_root / "titanic", "titanic")
    # scaffold writes ../experiences.db — point elsewhere via reload
    marker = client.marker_path
    text = marker.read_text(encoding="utf-8")
    text = text.replace("../experiences.db", str(tmp_path / "custom" / "mem.db"))
    marker.write_text(text, encoding="utf-8")
    from labpilot.workspace import load_workspace

    ws = load_workspace(marker)
    path = resolve_experience_db_path(workspace=ws)
    assert path == (tmp_path / "custom" / "mem.db").resolve()


def test_resolve_experience_db_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = scaffold_workspace(tmp_path / "slug", "slug")
    env_path = tmp_path / "from-env.db"
    monkeypatch.setenv(EXPERIENCE_DB_ENV, str(env_path))
    path = resolve_experience_db_path(workspace=client)
    assert path == env_path.resolve()


def test_resolve_experience_db_user_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(EXPERIENCE_DB_ENV, raising=False)
    # knowledge_dir not named "knowledge" and no workspace → user global
    path = resolve_experience_db_path(knowledge_dir=tmp_path / "odd-name")
    assert path == USER_EXPERIENCE_DB.resolve()


def test_update_workspace_experience_path_relative(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "kaggle" / "slug", "slug")
    updated = update_workspace_experience_path(
        client,
        tmp_path / "kaggle" / EXPERIENCE_DB_FILENAME,
        store_as=f"../{EXPERIENCE_DB_FILENAME}",
    )
    assert updated.memory.experience_store.path == f"../{EXPERIENCE_DB_FILENAME}"
    assert resolve_experience_db_path(workspace=updated) == (
        tmp_path / "kaggle" / EXPERIENCE_DB_FILENAME
    ).resolve()


def test_configure_experience_db_custom_noninteractive(tmp_path: Path) -> None:
    from labpilot.cli.init_workspace import configure_experience_db_for_init

    client = scaffold_workspace(tmp_path / "kaggle" / "slug", "slug")
    custom = tmp_path / "shared" / "mem.db"
    updated = configure_experience_db_for_init(
        client,
        experience_db=custom,
        experience_db_fallback=False,
        interactive=False,
    )
    assert resolve_experience_db_path(workspace=updated) == custom.resolve()
    assert custom.parent.is_dir()


def test_resolve_user_fallback_prompt_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(EXPERIENCE_DB_ENV, raising=False)
    chosen = tmp_path / "chosen.db"

    def _prompt(fallback: Path) -> Path:
        assert fallback == USER_EXPERIENCE_DB.resolve()
        return chosen

    path = resolve_experience_db_path(
        knowledge_dir=tmp_path / "odd-name",
        on_user_fallback=_prompt,
    )
    assert path == chosen.resolve()


def test_experience_store_uses_shared_parent_db(tmp_path: Path) -> None:
    research_root = tmp_path / "kaggle"
    bird = scaffold_workspace(research_root / "bird", "bird")
    boat = scaffold_workspace(research_root / "boat", "boat")
    store_a = ExperienceStore(bird.knowledge_dir, workspace=bird)
    store_b = ExperienceStore(boat.knowledge_dir, workspace=boat)
    try:
        assert store_a.db_path == store_b.db_path == research_root / EXPERIENCE_DB_FILENAME
        store_a.create(
            source_competition="bird",
            idempotency_key="run-1",
            outcome="success",
            tags=["audio"],
        )
        listed = store_b.list(source_competition="bird")
        assert len(listed) == 1
        assert listed[0].idempotency_key == "run-1"
    finally:
        store_a.close()
        store_b.close()
