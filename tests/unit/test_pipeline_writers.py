from pathlib import Path

from labpilot.config import AppConfig
from labpilot.orchestrator.pipeline import Pipeline


def test_write_config_snapshot_never_contains_secrets(tmp_path: Path):
    """Milestone 2, Plan 1, acceptance criterion: `runs/<id>/config.json`
    must never contain secret values, for a fixture run created with
    real-looking (fake) credentials set. Covers the `Pipeline._start()`
    writer directly (no full pipeline run needed); `fork_run`'s equivalent
    writer is covered in `test_improvement.py`."""
    config = AppConfig()
    config.kaggle.api_token = "kaggle-fake-api-token-123456"
    config.kaggle.username = "fake-kaggle-user"
    config.kaggle.key = "fake-kaggle-key-abcdef"
    config.llm.api_key = "sk-fake-openai-key-0000000000"

    pipeline = Pipeline(config)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pipeline._write_config_snapshot(run_dir)

    config_text = (run_dir / "config.json").read_text()
    for secret in (
        "kaggle-fake-api-token-123456",
        "fake-kaggle-user",
        "fake-kaggle-key-abcdef",
        "sk-fake-openai-key-0000000000",
    ):
        assert secret not in config_text


def test_write_config_snapshot_is_non_fatal_on_write_failure(tmp_path: Path):
    """A write failure (e.g. run_dir missing/unwritable) must never raise —
    Research Memory is a nice-to-have, not a pipeline dependency."""
    config = AppConfig()
    pipeline = Pipeline(config)

    missing_dir = tmp_path / "does-not-exist"
    pipeline._write_config_snapshot(missing_dir)  # no exception raised
    assert not (missing_dir / "config.json").exists()
