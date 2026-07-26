import json
from pathlib import Path

from labpilot.research_engine.intelligence.competition.models import CompetitionSpec
from labpilot.accessor.kaggle.exporter import _adapt_train_script, export_kernel


def test_kernel_exporter_writes_metadata(tmp_path: Path):
    run_dir = tmp_path / "run"
    pipeline_dir = run_dir / "pipeline"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "train.py").write_text(
        'from pathlib import Path\n'
        'from labpilot.research_engine.execution.metrics import compute_metric\n'
        'DATA_DIR = Path("/local/data/raw")\n'
        'OUTPUT_DIR = Path("/local/run")\n'
        'def main():\n'
        '    pass\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    competition = CompetitionSpec(
        slug="aerial-cactus-identification",
        title="Aerial Cactus Identification",
        submission_mode="kernel",
    )

    kernel_dir = export_kernel(run_dir, competition)

    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    assert metadata["kernel_type"] == "script"
    assert metadata["language"] == "python"
    assert metadata["code_file"] == "run.py"
    assert metadata["competition_sources"] == ["aerial-cactus-identification"]
    assert len(metadata["id"]) <= 60
    assert "labpilot-baseline" not in metadata["id"] or len(metadata["id"].split("/")[-1]) <= 30

    run_py = (kernel_dir / "run.py").read_text()
    assert "labpilot.evaluation.metrics" not in run_py
    assert "labpilot.research_engine.execution.metrics" not in run_py
    assert "/kaggle/input/competitions/aerial-cactus-identification" in run_py
    assert "/kaggle/working" in run_py
    assert "def compute_metric" in run_py


def test_kernel_exporter_injects_kaggle_bootstrap_for_image_template(tmp_path: Path):
    source = (Path(__file__).resolve().parents[2] / "src/labpilot/research_engine/execution/capabilities/code_engineering/templates/image_classification/train.py.j2").read_text()
    adapted = _adapt_train_script(
        source,
        "/kaggle/input/competitions/aerial-cactus-identification",
        "/kaggle/working",
    )

    assert "_labpilot_prepare_kaggle_data()" in adapted
    assert "_labpilot_resolve_image_path" in adapted
    assert "zipfile.ZipFile" in adapted
    assert "competition_sources" in adapted
    assert "return _labpilot_resolve_image_path(value)" in adapted
    assert "    _labpilot_prepare_kaggle_data()" in adapted
