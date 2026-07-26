"""CI validations for ``research hypothesize`` (generate + list/show/update)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus
from labpilot.research_engine.intelligence.knowledge import KnowledgeHub, KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    ResearchArtifact,
    ResearchArtifactType,
)

runner = CliRunner()
_HELP_ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def _plain(text: str) -> str:
    """Strip ANSI codes and collapse whitespace for CI-stable help assertions."""
    without_ansi = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"\s+", "", without_ansi)


def _seed_artifact(knowledge: Path, competition: str = "birdclef-2026") -> None:
    with KnowledgeStore(knowledge, competition) as store:
        store.upsert_artifact(
            ResearchArtifact(
                id="paper:1",
                type=ResearchArtifactType.PAPER,
                source="semantic_scholar",
                title="SpecAugment for ASR",
                techniques=["SpecAugment", "Mixup", "Focal Loss"],
                summary="SpecAugment improves generalization.",
                confidence=0.8,
            )
        )
        KnowledgeHub(store).ingest(store.list_artifacts())


def test_hypothesize_group_help_documents_subcommands() -> None:
    result = runner.invoke(app, ["hypothesize", "--help"], env=_HELP_ENV)
    assert result.exit_code == 0, result.output
    plain = _plain(result.stdout)
    for name in ("new", "list", "show", "update"):
        assert name in plain
    assert "Generate,inspect,andupdatehypotheses" in plain or "hypotheses" in plain.lower()


def test_hypothesize_new_help_documents_flags() -> None:
    result = runner.invoke(app, ["hypothesize", "new", "--help"], env=_HELP_ENV)
    assert result.exit_code == 0, result.output
    plain = _plain(result.stdout)
    for flag in ("--question", "--pipeline", "--limit", "--format", "--knowledge-dir"):
        assert flag in plain


def test_ingest_help_documents_skip_hypothesize() -> None:
    result = runner.invoke(app, ["ingest", "--help"], env=_HELP_ENV)
    assert result.exit_code == 0, result.output
    plain = _plain(result.stdout)
    assert "--skip-hypothesize" in plain
    assert "--force" in plain


def test_hypothesis_group_is_gone() -> None:
    result = runner.invoke(app, ["hypothesis", "list", "-c", "titanic"], env=_HELP_ENV)
    assert result.exit_code != 0
    assert "No such command 'hypothesis'" in result.output


def test_hypothesis_add_is_gone() -> None:
    result = runner.invoke(
        app,
        ["hypothesis", "add", "--competition", "titanic", "--observation", "x"],
        env=_HELP_ENV,
    )
    assert result.exit_code != 0
    assert "No such command 'hypothesis'" in result.output


def test_hypothesize_cli_list_update_show(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    runs = tmp_path / "runs"
    runs.mkdir()

    # Hypotheses are generated, never hand-authored via the CLI.
    HypothesisStore(knowledge, "titanic").create(
        observation="Rare classes perform poorly",
        reason="Dataset imbalance",
        prediction="Focal Loss will improve Macro F1",
        confidence=0.74,
        tags=["loss", "class-imbalance"],
    )
    assert (knowledge / "titanic" / "hypotheses" / "H-001.json").is_file()

    list_result = runner.invoke(
        app,
        [
            "hypothesize",
            "list",
            "--competition",
            "titanic",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert list_result.exit_code == 0, list_result.output
    assert "H-001" in list_result.output
    assert "proposed" in list_result.output

    update = runner.invoke(
        app,
        [
            "hypothesize",
            "update",
            "H-001",
            "--competition",
            "titanic",
            "--status",
            "confirmed",
            "--evidence-run",
            "20260714-run",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert update.exit_code == 0, update.output
    store = HypothesisStore(knowledge, "titanic")
    hypothesis = store.get("H-001")
    assert hypothesis is not None
    assert hypothesis.status == HypothesisStatus.CONFIRMED
    assert hypothesis.evidence_for == ["20260714-run"]

    show = runner.invoke(
        app,
        [
            "hypothesize",
            "show",
            "H-001",
            "--competition",
            "titanic",
            "--knowledge-dir",
            str(knowledge),
            "--runs-dir",
            str(runs),
        ],
    )
    assert show.exit_code == 0, show.output
    assert "Focal Loss will improve Macro F1" in show.output
    assert "confirmed" in show.output


def test_hypothesize_list_filters_by_status(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    store = HypothesisStore(knowledge, "titanic")
    store.create(
        observation="a", reason="b", prediction="proposed idea", confidence=0.5, tags=["a"]
    )
    testing = store.create(
        observation="c", reason="d", prediction="testing idea", confidence=0.6, tags=["b"]
    )
    store.update_status(testing.id, HypothesisStatus.TESTING)

    proposed = runner.invoke(
        app,
        [
            "hypothesize",
            "list",
            "-c",
            "titanic",
            "--status",
            "proposed",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert proposed.exit_code == 0, proposed.output
    assert "H-001" in proposed.output
    assert "H-002" not in proposed.output

    testing_list = runner.invoke(
        app,
        [
            "hypothesize",
            "list",
            "-c",
            "titanic",
            "--status",
            "testing",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert testing_list.exit_code == 0, testing_list.output
    assert "H-002" in testing_list.output
    assert "H-001" not in testing_list.output


def test_hypothesize_list_rejects_invalid_status(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "hypothesize",
            "list",
            "-c",
            "titanic",
            "--status",
            "bogus",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
        ],
    )
    assert result.exit_code != 0


def test_hypothesize_show_requires_competition() -> None:
    result = runner.invoke(app, ["hypothesize", "show", "H-001"])
    assert result.exit_code != 0


def test_hypothesize_show_missing_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "hypothesize",
            "show",
            "H-404",
            "-c",
            "titanic",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_hypothesize_update_rejected_routes_evidence(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    HypothesisStore(knowledge, "titanic").create(
        observation="a",
        reason="b",
        prediction="c",
        confidence=0.5,
        tags=["loss"],
    )
    result = runner.invoke(
        app,
        [
            "hypothesize",
            "update",
            "H-001",
            "-c",
            "titanic",
            "--status",
            "rejected",
            "--evidence-run",
            "run-bad",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert result.exit_code == 0, result.output
    hyp = HypothesisStore(knowledge, "titanic").get("H-001")
    assert hyp is not None
    assert hyp.status == HypothesisStatus.REJECTED
    assert hyp.evidence_against == ["run-bad"]
    assert "evidence_against" in result.output


def test_hypothesize_bare_slug_does_not_steal_subcommands(tmp_path: Path) -> None:
    """``list`` / ``show`` / ``update`` must route as subcommands, not as slugs."""
    knowledge = tmp_path / "knowledge"
    HypothesisStore(knowledge, "titanic").create(
        observation="a", reason="b", prediction="c", confidence=0.5
    )
    # If "list" were rewritten to ``new list``, this would fail looking up competition.
    result = runner.invoke(
        app,
        ["hypothesize", "list", "-c", "titanic", "--knowledge-dir", str(knowledge)],
    )
    assert result.exit_code == 0, result.output
    assert "H-001" in result.output


def test_hypothesize_cli_json_format_and_dual_write(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_artifact(knowledge)

    result = runner.invoke(
        app,
        [
            "hypothesize",
            "birdclef-2026",
            "--pipeline",
            "EMA",
            "--limit",
            "5",
            "--format",
            "json",
            "--knowledge-dir",
            str(knowledge),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "recommendations" in payload
    assert "new_count" in payload
    assert payload["new_count"] == len(payload["recommendations"])
    assert payload["new_count"] >= 1

    store = HypothesisStore(knowledge, "birdclef-2026")
    hyps = store.list(status=HypothesisStatus.PROPOSED)
    assert len(hyps) == payload["new_count"]

    with KnowledgeStore(knowledge, "birdclef-2026") as kstore:
        db_ids = {row["id"] for row in kstore.list_hypotheses(status="proposed")}
    assert db_ids == {h.id for h in hyps}

    report = knowledge / "birdclef-2026/research/reports/hypotheses.json"
    assert report.is_file()


def test_hypothesize_cli_rerun_reports_zero_new(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_artifact(knowledge)
    args = [
        "hypothesize",
        "birdclef-2026",
        "--pipeline",
        "EMA",
        "--knowledge-dir",
        str(knowledge),
        "--runs-dir",
        str(tmp_path / "runs"),
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert re.search(r"[1-9]\d* new hypothesis generated", first.output)

    count_after_first = len(HypothesisStore(knowledge, "birdclef-2026").list())
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "0 new hypothesis generated" in second.output
    assert len(HypothesisStore(knowledge, "birdclef-2026").list()) == count_after_first


def test_hypothesize_new_explicit_matches_bare_slug(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_artifact(knowledge)
    bare = runner.invoke(
        app,
        [
            "hypothesize",
            "birdclef-2026",
            "--pipeline",
            "EMA",
            "--format",
            "json",
            "--knowledge-dir",
            str(knowledge),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    # Wipe and regenerate via explicit ``new`` — same surface, fresh store.
    knowledge2 = tmp_path / "knowledge2"
    _seed_artifact(knowledge2)
    explicit = runner.invoke(
        app,
        [
            "hypothesize",
            "new",
            "birdclef-2026",
            "--pipeline",
            "EMA",
            "--format",
            "json",
            "--knowledge-dir",
            str(knowledge2),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert bare.exit_code == 0, bare.output
    assert explicit.exit_code == 0, explicit.output
    bare_payload = json.loads(bare.stdout)
    explicit_payload = json.loads(explicit.stdout)
    assert bare_payload["new_count"] == explicit_payload["new_count"]
    assert bare_payload["new_count"] >= 1
