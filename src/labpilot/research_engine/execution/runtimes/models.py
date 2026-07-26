"""Runtime configuration models (P2 v0.3 — remote dispatch deferred to P2 execution)."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PollConfig(BaseModel):
    interval_seconds: int = 15
    timeout_seconds: int = 3600


class QuotaConfig(BaseModel):
    daily_gpu_hours: float | None = None
    weekly_runs: int | None = None
    concurrent_jobs: int = 1
    reset_timezone: str = "UTC"


class RuntimeBase(BaseModel):
    schema_version: int = 1
    id: str
    provider: str
    enabled: bool = True
    priority: int = 0
    labels: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(
        default_factory=lambda: ["models/", "oof.csv", "metrics.json", "training.log"]
    )
    poll: PollConfig = Field(default_factory=PollConfig)
    quotas: QuotaConfig = Field(default_factory=QuotaConfig)


class LocalRuntime(RuntimeBase):
    provider: Literal["local"] = "local"
    python: str | None = None
    timeout_seconds: int | None = None
    env: dict[str, str] = Field(default_factory=dict)


class KaggleKernelRuntime(RuntimeBase):
    provider: Literal["kaggle_kernel"] = "kaggle_kernel"
    username: str | None = None
    kernel_type: Literal["script", "notebook"] = "script"
    accelerator: Literal["none", "gpu", "tpu"] = "none"
    language: str = "python"
    push_dir: str = "kernel"
    slug_template: str = "{competition}-lp-{run_suffix}"


class ColabAuthConfig(BaseModel):
    token_env: str = "COLAB_AUTH_TOKEN"


class ColabDriveSyncConfig(BaseModel):
    folder_id_env: str = "LABPILOT_COLAB_DRIVE_FOLDER"


class GoogleColabRuntime(RuntimeBase):
    provider: Literal["google_colab"] = "google_colab"
    runtime_type: Literal["cpu", "gpu", "tpu"] = "cpu"
    auth: ColabAuthConfig = Field(default_factory=ColabAuthConfig)
    drive_sync: ColabDriveSyncConfig | None = None
    install_extras: list[str] = Field(default_factory=list)


class OtherRuntime(RuntimeBase):
    provider: Literal["other"] = "other"
    adapter: str
    host: str | None = None
    user: str | None = None
    key_env: str | None = None
    remote_runs_dir: str | None = None
    sync_method: str | None = None
    bootstrap_script: str | None = None


RuntimeConfig = Annotated[
    LocalRuntime | KaggleKernelRuntime | GoogleColabRuntime | OtherRuntime,
    Field(discriminator="provider"),
]


class RuntimeRecord(BaseModel):
    """Snapshot written to runs/<id>/runtime.json at pipeline start."""

    runtime_id: str
    provider: str
    mode: Literal["local", "remote"] = "local"
