"""M6 capstone — extract → store → ContextBundle → seed/inspect; no Conductor bypass."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.research_engine.agents.events import EXPERIMENT_COMPLETED, EventBus
from labpilot.research_engine.context import ContextRequest, ExperienceProvider, build_context
from labpilot.research_engine.memory import ExperienceStore
from labpilot.research_engine.memory.hooks import (
    install_experience_memory_subscriber,
    persist_experience_from_completion,
)
from labpilot.research_engine.memory.seed import load_seeded_experience_ids
from labpilot.workspace import scaffold_workspace

runner = CliRunner()
_HELP_ENV = {
    "COLUMNS": "200",
    "NO_COLOR": "1",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "LABPILOT_LLM_MODE": "cloud",
}


def test_m6_capstone_transfer_memory_smoke(tmp_path: Path) -> None:
    research = tmp_path / "kaggle"
    bird = scaffold_workspace(research / "birdclef-2026", "birdclef-2026")
    whale = scaffold_workspace(research / "whale-sound", "whale-sound")

    bus = EventBus()
    install_experience_memory_subscriber(bus)
    bus.publish(
        EXPERIMENT_COMPLETED,
        {
            "competition": "birdclef-2026",
            "knowledge_dir": str(bird.knowledge_dir),
            "workspace_root": str(bird.root),
            "experiment_id": "bird-exp-1",
            "execution_id": "bird-exec-1",
            "plan_id": "P-bird",
            "status": "completed",
            "metrics": {"lb_score": 0.706, "delta": 0.006},
            "git_commit": "deadbeefcafebabe",
            "description": "Added SpecAugment + EMA for minority bird calls",
            "comparison": {
                "delta": 0.006,
                "verdict": "worth_keeping",
                "maximize": True,
            },
        },
    )

    # Shared experiences.db under parent research root
    store = ExperienceStore(bird.knowledge_dir, workspace=bird)
    try:
        bird_rows = store.list(source_competition="birdclef-2026")
        assert len(bird_rows) == 1
        record = bird_rows[0]
        assert record.artifacts.git_commit == "deadbeefcafebabe"
        assert record.artifacts.experiment_id == "bird-exp-1"
    finally:
        store.close()

    # Idempotent re-fire
    again = persist_experience_from_completion(
        {
            "competition": "birdclef-2026",
            "knowledge_dir": str(bird.knowledge_dir),
            "experiment_id": "bird-exp-1",
            "metrics": {"lb_score": 0.706, "delta": 0.006},
            "git_commit": "deadbeefcafebabe",
            "description": "Added SpecAugment + EMA for minority bird calls",
            "comparison": {
                "delta": 0.006,
                "verdict": "worth_keeping",
                "maximize": True,
            },
        }
    )
    assert again is not None
    store = ExperienceStore(whale.knowledge_dir, workspace=whale)
    try:
        assert len(store.list()) == 1
    finally:
        store.close()

    # ContextBundle for B surfaces A without auto-seeding
    request = ContextRequest(
        competition="whale-sound",
        goal="Underwater sound classification",
        query="audio spectrogram SpecAugment bird minority",
        knowledge_dir=whale.knowledge_dir,
        max_items=16,
        max_chars=8000,
    )
    bundle = build_context(request, providers=[ExperienceProvider()])
    exp_items = [i for i in bundle.items if i.source == "experience"]
    assert exp_items, bundle.provider_errors
    assert any(
        i.metadata.get("source_competition") == "birdclef-2026" for i in exp_items
    )
    assert load_seeded_experience_ids(whale.knowledge_dir, "whale-sound") == set()

    # Explicit seed is auditable
    seeded = runner.invoke(
        app,
        [
            "memory",
            "seed",
            "--from",
            "birdclef-2026",
            "--competition",
            "whale-sound",
            "--knowledge-dir",
            str(whale.knowledge_dir),
        ],
        env=_HELP_ENV,
    )
    assert seeded.exit_code == 0, seeded.output
    manifest = whale.knowledge_dir / "memory" / "seeds" / "birdclef-2026.json"
    assert manifest.is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["source_competition"] == "birdclef-2026"
    assert record.id in data["experience_ids"]
    assert load_seeded_experience_ids(whale.knowledge_dir, "whale-sound") == {record.id}

    inspected = runner.invoke(
        app,
        [
            "memory",
            "inspect",
            "--similar-to",
            "birdclef-2026",
            "-q",
            "audio SpecAugment",
            "--competition",
            "whale-sound",
            "--knowledge-dir",
            str(whale.knowledge_dir),
        ],
        env=_HELP_ENV,
    )
    assert inspected.exit_code == 0, inspected.output
    assert record.id in inspected.stdout or "birdclef" in inspected.stdout.lower()
    assert (
        "does not change Conductor" in inspected.stdout
        or "ContextBundle" in inspected.stdout
    )
