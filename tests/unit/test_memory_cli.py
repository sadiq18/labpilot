"""CLI tests for ``research memory`` seed/inspect/list/show."""

from __future__ import annotations

import json
import re
from pathlib import Path

from helpers.cli import cli_runner

from labpilot.cli.main import app
from labpilot.research_engine.memory import ExperienceStore
from labpilot.research_engine.memory.models import ExperienceFacet
from labpilot.workspace import scaffold_workspace

runner = cli_runner()
_HELP_ENV = {
    "COLUMNS": "200",
    "NO_COLOR": "1",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "LABPILOT_LLM_MODE": "cloud",
}


def _plain(text: str) -> str:
    without_ansi = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"\s+", "", without_ansi)


def _seed_pair(tmp_path: Path) -> tuple[Path, Path]:
    research = tmp_path / "kaggle"
    bird = scaffold_workspace(research / "birdclef-2026", "birdclef-2026")
    whale = scaffold_workspace(research / "whale-sound", "whale-sound")
    store = ExperienceStore(bird.knowledge_dir, workspace=bird)
    try:
        store.create(
            source_competition="birdclef-2026",
            idempotency_key="bird-1",
            goal="Improve BirdCLEF score",
            hypothesis="SpecAugment helps minority classes",
            action="Added SpecAugment",
            result="+0.006",
            outcome="success",
            facets=[
                ExperienceFacet(
                    facet="audio",
                    confidence=0.8,
                    evidence=["bird"],
                    source="rules",
                )
            ],
        )
    finally:
        store.close()
    return bird.knowledge_dir, whale.knowledge_dir


def test_memory_help() -> None:
    result = runner.invoke(app, ["memory", "--help"], env=_HELP_ENV)
    assert result.exit_code == 0, result.output
    plain = _plain(result.stdout)
    assert "seed" in plain
    assert "inspect" in plain
    assert "list" in plain
    assert "show" in plain
    assert "ContextBundle" in result.stdout or "operator" in result.stdout.lower()


def test_memory_list_and_show(tmp_path: Path) -> None:
    bird_k, _ = _seed_pair(tmp_path)
    listed = runner.invoke(
        app,
        ["memory", "list", "--competition", "birdclef-2026", "--knowledge-dir", str(bird_k)],
        env=_HELP_ENV,
    )
    assert listed.exit_code == 0, listed.output
    assert "XR-001" in listed.stdout
    assert "birdclef-2026" in listed.stdout

    shown = runner.invoke(
        app,
        ["memory", "show", "XR-001", "--knowledge-dir", str(bird_k)],
        env=_HELP_ENV,
    )
    assert shown.exit_code == 0, shown.output
    assert "SpecAugment" in shown.stdout
    assert "audio" in shown.stdout


def test_memory_seed_and_inspect(tmp_path: Path) -> None:
    bird_k, whale_k = _seed_pair(tmp_path)
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
            str(whale_k),
        ],
        env=_HELP_ENV,
    )
    assert seeded.exit_code == 0, seeded.output
    assert "Seeded" in seeded.output
    # Client layout is flat under knowledge/
    manifest = whale_k / "memory" / "seeds" / "birdclef-2026.json"
    assert manifest.is_file(), list(whale_k.rglob("*.json"))
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["source_competition"] == "birdclef-2026"
    assert "XR-001" in data["experience_ids"]

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
            str(whale_k),
        ],
        env=_HELP_ENV,
    )
    assert inspected.exit_code == 0, inspected.output
    assert "XR-001" in inspected.stdout or "birdclef" in inspected.stdout.lower()
    assert "does not change Conductor" in inspected.stdout or "ContextBundle" in inspected.stdout
