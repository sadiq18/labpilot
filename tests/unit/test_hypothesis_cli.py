from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.experiments.hypothesis import HypothesisStore
from labpilot.experiments.models import HypothesisStatus


def test_hypothesis_cli_add_list_update_show(tmp_path: Path):
    runner = CliRunner()
    knowledge = tmp_path / "knowledge"
    runs = tmp_path / "runs"
    runs.mkdir()

    add = runner.invoke(
        app,
        [
            "hypothesis",
            "add",
            "--competition",
            "titanic",
            "--observation",
            "Rare classes perform poorly",
            "--reason",
            "Dataset imbalance",
            "--prediction",
            "Focal Loss will improve Macro F1",
            "--confidence",
            "0.74",
            "--tags",
            "loss,class-imbalance",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert add.exit_code == 0, add.output
    assert "H-001" in add.output
    assert (knowledge / "titanic" / "hypotheses" / "H-001.json").is_file()

    list_result = runner.invoke(
        app,
        [
            "hypothesis",
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
            "hypothesis",
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
            "hypothesis",
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


def test_hypothesis_show_requires_competition(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(app, ["hypothesis", "show", "H-001"])
    assert result.exit_code != 0
