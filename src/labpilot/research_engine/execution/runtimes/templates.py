"""Scaffold YAML files for `research runtime register`."""

from labpilot.research_engine.execution.runtimes.models import (
    GoogleColabRuntime,
    KaggleKernelRuntime,
    LocalRuntime,
    OtherRuntime,
    RuntimeConfig,
)


def scaffold_runtime(provider: str, runtime_id: str) -> RuntimeConfig:
    if provider == "local":
        return LocalRuntime(id=runtime_id, priority=0)
    if provider == "kaggle_kernel":
        return KaggleKernelRuntime(
            id=runtime_id,
            priority=10,
            labels=["gpu", "free-tier"],
            accelerator="gpu",
        )
    if provider == "google_colab":
        return GoogleColabRuntime(
            id=runtime_id,
            priority=20,
            labels=["gpu"],
            runtime_type="gpu",
            install_extras=["deep"],
        )
    if provider == "other":
        return OtherRuntime(
            id=runtime_id,
            adapter="labpilot.research_engine.execution.runtimes.adapters.ssh:SSHAdapter",
            host="gpu.example.com",
            user="lab",
            key_env="SSH_PRIVATE_KEY_PATH",
            remote_runs_dir="/data/labpilot/runs",
            sync_method="rsync",
        )
    raise ValueError(f"Unknown provider: {provider!r}")


def runtime_to_yaml_dict(runtime: RuntimeConfig) -> dict:
    data = runtime.model_dump(exclude_none=True)
    return data
