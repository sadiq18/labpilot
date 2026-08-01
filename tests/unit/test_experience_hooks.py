"""Write-only experience memory hooks (Blinker + callable)."""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.agents.events import EXPERIMENT_COMPLETED, EventBus
from labpilot.research_engine.memory import ExperienceStore
from labpilot.research_engine.memory.hooks import (
    install_experience_memory_subscriber,
    persist_experience_from_completion,
)
from labpilot.workspace import scaffold_workspace


def _payload(knowledge: Path, competition: str, **overrides: object) -> dict:
    base: dict = {
        "competition": competition,
        "knowledge_dir": str(knowledge),
        "workspace_root": str(knowledge.parent),
        "experiment_id": "exp-hook-1",
        "execution_id": "exec-hook-1",
        "plan_id": "P-001",
        "status": "completed",
        "metrics": {"lb_score": 0.71, "delta": 0.008},
        "git_commit": "abc123def456",
        "files_changed": ["pipeline/train.py"],
        "description": "Added SpecAugment for minority audio classes",
        "comparison": {
            "delta": 0.008,
            "verdict": "worth_keeping",
            "maximize": True,
        },
    }
    base.update(overrides)
    return base


def test_persist_from_completion_idempotent(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "birdclef-2026", "birdclef-2026")
    knowledge = client.knowledge_dir
    payload = _payload(knowledge, "birdclef-2026")

    first = persist_experience_from_completion(payload)
    assert first is not None
    assert first.artifacts.git_commit == "abc123def456"
    assert first.idempotency_key == "exp-hook-1"

    second = persist_experience_from_completion(
        {
            **payload,
            "description": "Added SpecAugment for minority audio classes (retry)",
            "metrics": {"lb_score": 0.712, "delta": 0.01},
        }
    )
    assert second is not None
    assert second.id == first.id

    store = ExperienceStore(knowledge, workspace=client)
    try:
        assert len(store.list()) == 1
        got = store.get(first.id)
        assert got is not None
        assert got.artifacts.git_commit == "abc123def456"
        assert "(retry)" in got.action or "retry" in got.action.lower()
    finally:
        store.close()


def test_subscriber_writes_on_experiment_completed(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "audio-comp", "audio-comp")
    bus = EventBus()
    install_experience_memory_subscriber(bus)
    bus.publish(
        EXPERIMENT_COMPLETED,
        _payload(client.knowledge_dir, "audio-comp", experiment_id="exp-bus-1"),
    )
    store = ExperienceStore(client.knowledge_dir, workspace=client)
    try:
        rows = store.list()
        assert len(rows) == 1
        assert rows[0].idempotency_key == "exp-bus-1"
        assert rows[0].artifacts.git_commit == "abc123def456"
    finally:
        store.close()


def test_persist_swallows_errors_and_skips_incomplete(tmp_path: Path) -> None:
    assert persist_experience_from_completion({}) is None
    assert persist_experience_from_completion({"competition": "x"}) is None
    client = scaffold_workspace(tmp_path / "x", "x")
    # Valid payload must not raise into the experiment pipeline
    record = persist_experience_from_completion(
        {
            "competition": "x",
            "knowledge_dir": str(client.knowledge_dir),
            "experiment_id": "e2",
            "description": "tabular baseline",
        }
    )
    assert record is not None
    assert record.idempotency_key == "e2"


def test_subscriber_is_write_only(tmp_path: Path) -> None:
    """Memory subscriber only upserts ExperienceStore — no follow-on bus events."""
    client = scaffold_workspace(tmp_path / "comp", "comp")
    bus = EventBus()
    follow_ons: list[str] = []
    original_publish = bus.publish

    def _tracking_publish(event: str, payload: dict | None = None) -> None:
        if event != EXPERIMENT_COMPLETED:
            follow_ons.append(event)
        original_publish(event, payload)

    bus.publish = _tracking_publish  # type: ignore[method-assign]
    install_experience_memory_subscriber(bus)
    bus.publish(
        EXPERIMENT_COMPLETED,
        _payload(client.knowledge_dir, "comp", experiment_id="exp-only"),
    )
    assert follow_ons == []
    store = ExperienceStore(client.knowledge_dir, workspace=client)
    try:
        assert len(store.list()) == 1
    finally:
        store.close()

