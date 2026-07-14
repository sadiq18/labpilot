from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from labpilot.workspace.discover import load_project

# Shared with `labpilot.llm.client.create_llm_client()`: if a user switches
# `llm.provider` without also overriding `llm.model`, this is what resolves
# to a real model name for that provider instead of silently sending an
# OpenAI-flavored default (e.g. "gpt-4o-mini") to a different API.
DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash",
}


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    api_key: str = Field(default="", exclude=True, repr=False)


class TrainingConfig(BaseModel):
    cv_folds: int = 5
    random_seed: int = 42


class ProfilerConfig(BaseModel):
    max_rows_sample: int = 100_000
    categorical_cardinality_threshold: int = 50
    max_images_sample: int = 5_000


class DeepBaselineConfig(BaseModel):
    max_epochs: int = 3
    max_train_samples: int = 5_000
    batch_size: int = 16
    learning_rate: float = 2e-5
    unfreeze_last_n_layers: int = 1
    early_stopping_patience: int = 1
    cpu_max_epochs: int = 2
    cpu_max_train_samples: int = 2_000
    cv_folds: int = 3


class KaggleConfig(BaseModel):
    download_unzip: bool = True
    submit_message: str = "labpilot baseline submission"
    cache_dir: Path = Path(".cache/kaggle")
    submission_poll_timeout: int = 120
    submission_poll_interval: int = 5
    kernel_poll_timeout: int = 3600
    kernel_poll_interval: int = 15
    api_token: str = Field(default="", exclude=True, repr=False)
    username: str = Field(default="", exclude=True, repr=False)
    key: str = Field(default="", exclude=True, repr=False)


class RuntimeDefaults(BaseModel):
    runtimes_dir: Path = Path("configs/runtimes")
    default_runtime: str = "local-default"


class PipelineConfig(BaseModel):
    stages: list[str] = Field(default_factory=list)


class ComparatorConfig(BaseModel):
    noise_epsilon: float = 0.001
    max_runtime_increase_pct: float = 50.0


class ExperimentsConfig(BaseModel):
    comparator: ComparatorConfig = Field(default_factory=ComparatorConfig)


class AppConfig(BaseModel):
    runs_dir: Path = Path("runs")
    knowledge_dir: Path = Path("knowledge")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    deep_baseline: DeepBaselineConfig = Field(default_factory=DeepBaselineConfig)
    kaggle: KaggleConfig = Field(default_factory=KaggleConfig)
    runtime: RuntimeDefaults = Field(default_factory=RuntimeDefaults)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kaggle_api_token: str = ""
    kaggle_username: str = ""
    kaggle_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    labpilot_runs_dir: str = "runs"
    labpilot_knowledge_dir: str = "knowledge"
    labpilot_llm_provider: str = ""
    labpilot_llm_model: str = ""
    labpilot_runtimes_dir: str = ""
    labpilot_default_runtime: str = ""


def _package_default_config_path() -> Path:
    return Path("configs/default.yaml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _normalize_paths(raw: dict[str, Any]) -> dict[str, Any]:
    if "runs_dir" in raw:
        raw["runs_dir"] = Path(raw["runs_dir"])
    if "knowledge_dir" in raw:
        raw["knowledge_dir"] = Path(raw["knowledge_dir"])
    runtime = raw.get("runtime")
    if isinstance(runtime, dict) and "runtimes_dir" in runtime:
        runtime["runtimes_dir"] = Path(runtime["runtimes_dir"])
    kaggle = raw.get("kaggle")
    if isinstance(kaggle, dict) and "cache_dir" in kaggle:
        kaggle["cache_dir"] = Path(kaggle["cache_dir"])
    return raw


def _apply_project_overrides(raw: dict[str, Any], project) -> dict[str, Any]:
    project_raw = _load_yaml_dict(project.config_path)
    merged = _deep_merge(raw, project_raw)
    merged["runs_dir"] = project.runs_dir
    runtime = merged.setdefault("runtime", {})
    runtime["runtimes_dir"] = project.runtimes_dir
    runtime["default_runtime"] = project.default_runtime
    kaggle = merged.setdefault("kaggle", {})
    kaggle["cache_dir"] = project.cache_dir
    return merged


def _apply_settings(config: AppConfig, settings: Settings, raw: dict[str, Any]) -> AppConfig:
    config.kaggle.api_token = settings.kaggle_api_token
    config.kaggle.username = settings.kaggle_username
    config.kaggle.key = settings.kaggle_key

    if settings.labpilot_runs_dir != "runs":
        config.runs_dir = Path(settings.labpilot_runs_dir)
    if settings.labpilot_knowledge_dir != "knowledge":
        config.knowledge_dir = Path(settings.labpilot_knowledge_dir)
    if settings.labpilot_runtimes_dir:
        config.runtime.runtimes_dir = Path(settings.labpilot_runtimes_dir)
    if settings.labpilot_default_runtime:
        config.runtime.default_runtime = settings.labpilot_default_runtime

    if settings.labpilot_llm_provider:
        config.llm.provider = settings.labpilot_llm_provider
    elif settings.gemini_api_key.strip() and not settings.openai_api_key.strip():
        config.llm.provider = "gemini"
    elif settings.openai_api_key.strip() and not settings.gemini_api_key.strip():
        config.llm.provider = "openai"
    if settings.labpilot_llm_model:
        config.llm.model = settings.labpilot_llm_model
    elif "model" not in raw.get("llm", {}):
        config.llm.model = DEFAULT_MODEL_BY_PROVIDER.get(config.llm.provider, config.llm.model)

    config.llm.api_key = {
        "openai": settings.openai_api_key,
        "gemini": settings.gemini_api_key,
    }.get(config.llm.provider.strip().lower(), "")

    return config


def load_config(
    path: Path | None = None,
    *,
    project_dir: Path | None = None,
    start_dir: Path | None = None,
) -> AppConfig:
    """Load config with layered precedence:

    1. Package default (`configs/default.yaml`)
    2. Project config (when `project.yaml` is detected or `--project-dir` is set)
    3. Explicit CLI `--config` file
    4. Environment variables (`LABPILOT_*`, credentials)
    """
    layers: list[dict[str, Any]] = []

    package_default = _package_default_config_path()
    if package_default.is_file():
        layers.append(_load_yaml_dict(package_default))

    project = load_project(start=start_dir, project_dir=project_dir)
    if project is not None:
        project_layer = _load_yaml_dict(project.config_path)
        project_layer["runs_dir"] = str(project.runs_dir)
        project_layer.setdefault("runtime", {})
        project_layer["runtime"]["runtimes_dir"] = str(project.runtimes_dir)
        project_layer["runtime"]["default_runtime"] = project.default_runtime
        project_layer.setdefault("kaggle", {})
        project_layer["kaggle"]["cache_dir"] = str(project.cache_dir)
        layers.append(project_layer)

    explicit_path = path or _package_default_config_path()
    skip_explicit = (
        project is not None
        and explicit_path.resolve() == _package_default_config_path().resolve()
        and explicit_path.resolve() != project.config_path.resolve()
    )
    if explicit_path.is_file() and not skip_explicit:
        resolved_explicit = explicit_path.resolve()
        already_loaded = {_package_default_config_path().resolve()}
        if project is not None:
            already_loaded.add(project.config_path.resolve())
        if resolved_explicit not in already_loaded:
            layers.append(_load_yaml_dict(explicit_path))
        elif project is None and resolved_explicit == _package_default_config_path().resolve():
            pass  # already loaded as package default
        elif project is not None and resolved_explicit == project.config_path.resolve():
            pass  # already loaded via project layer
        else:
            layers.append(_load_yaml_dict(explicit_path))

    merged: dict[str, Any] = {}
    for layer in layers:
        merged = _deep_merge(merged, layer)

    merged = _normalize_paths(merged)
    config = AppConfig.model_validate(merged)
    settings = Settings()
    return _apply_settings(config, settings, merged)


def resolve_competitions_dir(
    config: AppConfig,
    competitions_dir: Path | None = None,
    *,
    project_dir: Path | None = None,
    start_dir: Path | None = None,
) -> Path:
    if competitions_dir is not None:
        return competitions_dir
    project = load_project(start=start_dir, project_dir=project_dir)
    if project is not None:
        return project.competitions_dir
    return Path(__file__).resolve().parents[3] / "configs" / "competitions"


def resolve_runtimes_dir(
    config: AppConfig,
    *,
    project_dir: Path | None = None,
    start_dir: Path | None = None,
) -> Path:
    project = load_project(start=start_dir, project_dir=project_dir)
    if project is not None:
        return project.runtimes_dir
    return config.runtime.runtimes_dir
