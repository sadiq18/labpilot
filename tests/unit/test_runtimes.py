from pathlib import Path

import yaml

from labpilot.config import load_config
from labpilot.runtimes.doctor import check_runtime
from labpilot.runtimes.models import GoogleColabRuntime, KaggleKernelRuntime, LocalRuntime
from labpilot.runtimes.registry import get_runtime, load_runtimes


def test_load_runtimes_includes_builtin_and_yaml(tmp_path: Path):
    runtimes_dir = tmp_path / "runtimes"
    runtimes_dir.mkdir()
    (runtimes_dir / "custom.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "custom-local",
                "provider": "local",
                "enabled": True,
                "priority": 5,
            }
        )
    )
    runtimes = load_runtimes(runtimes_dir)
    assert "local-default" in runtimes
    assert "custom-local" in runtimes


def test_kaggle_runtime_schema_validation():
    runtime = KaggleKernelRuntime(
        id="kaggle-test",
        accelerator="gpu",
        labels=["gpu"],
    )
    assert runtime.provider == "kaggle_kernel"


def test_colab_doctor_checks_token_env(monkeypatch):
    runtime = GoogleColabRuntime(id="colab-test")
    monkeypatch.delenv("COLAB_AUTH_TOKEN", raising=False)
    result = check_runtime(runtime)
    assert not result.ok
    assert any(check.name == "Colab auth token" for check in result.checks)


def test_local_runtime_doctor_passes():
    result = check_runtime(LocalRuntime(id="local-default"))
    assert result.ok


def test_app_config_runtime_defaults():
    config = load_config(Path("configs/default.yaml"))
    assert config.runtime.default_runtime == "local-default"


def test_get_runtime_from_shipped_config():
    runtime = get_runtime("local-default", Path("configs/runtimes"))
    assert runtime is not None
    assert runtime.provider == "local"
