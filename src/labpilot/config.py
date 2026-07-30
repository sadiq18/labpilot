from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Shared with `labpilot.llm.client.create_llm_client()`: if a user switches
# `llm.provider` without also overriding `llm.model`, this is what resolves
# to a real model name for that provider instead of silently sending an
# OpenAI-flavored default (e.g. "gpt-4o-mini") to a different API.
DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash-lite",
    "ollama": "qwen2.5-coder:14b",
}


class LLMCacheConfig(BaseModel):
    enabled: bool = True
    path: Path = Path(".cache/llm.sqlite")


class TaskProfile(BaseModel):
    """Per-task model routing overrides (planning, coding, summary, …)."""

    model: str | None = None
    provider: str | None = None
    force_local: bool = False
    temperature: float | None = None


class LLMConfig(BaseModel):
    mode: str = "auto"  # auto | local | cloud
    provider: str = "gemini"
    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.3
    api_key: str = Field(default="", exclude=True, repr=False)
    ollama_base_url: str = "http://localhost:11434"
    fallback_model: str = "qwen2.5-coder:14b"
    cache: LLMCacheConfig = Field(default_factory=LLMCacheConfig)
    tasks: dict[str, TaskProfile] = Field(default_factory=dict)


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


class ReflectionConfig(BaseModel):
    max_new_hypotheses: int = 3


class RankingWeightsConfig(BaseModel):
    expected_gain: float = 2.0
    implementation_cost: float = 0.5
    gpu_cost: float = 0.5
    risk: float = 1.0
    novelty: float = 0.5


class RankingConfig(BaseModel):
    default_expected_gain: float = 0.0
    cheap_tags: list[str] = Field(
        default_factory=lambda: [
            "hyperparameter",
            "hyperparams",
            "tune",
            "tuning",
            "loss",
            "scheduler",
            "features",
            "feature-engineering",
            "learning_rate",
            "num_leaves",
            "n_estimators",
            "target_encoding",
            "log_numeric",
        ]
    )
    weights: RankingWeightsConfig = Field(default_factory=RankingWeightsConfig)


class ExperimentsConfig(BaseModel):
    comparator: ComparatorConfig = Field(default_factory=ComparatorConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)


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
    semantic_scholar_api_key: str = ""
    openalex_mailto: str = ""
    openalex_api_key: str = ""
    hf_token: str = ""
    github_token: str = ""
    labpilot_runs_dir: str = "runs"
    labpilot_knowledge_dir: str = "knowledge"
    labpilot_llm_provider: str = ""
    labpilot_llm_model: str = ""
    labpilot_llm_mode: str = ""
    ollama_host: str = ""
    labpilot_runtimes_dir: str = ""
    labpilot_default_runtime: str = ""

    def __init__(self, **values: Any) -> None:
        # Prefer an explicit ``_env_file``; otherwise use workspace-local ``.env``
        # (never the LabPilot package/repo). Tests set ``model_config["env_file"]
        # = None`` to disable file loading entirely.
        if "_env_file" in values:
            env_file = values.pop("_env_file")
        elif self.model_config.get("env_file") is None:
            env_file = None
        else:
            env_file = resolve_env_files()
        super().__init__(_env_file=env_file, **values)


def resolve_env_files() -> tuple[str, ...]:
    """Workspace-local ``.env`` only (competition folder), not the LabPilot clone.

    When a ``labpilot.yaml`` workspace is active, credentials come from
    ``<workspace>/.env``. Otherwise legacy mode uses ``./.env`` under CWD/PWD.
    """
    # Lazy import: avoid circular import at module load (workspace → config).
    from labpilot.workspace import discover_workspace

    workspace = discover_workspace()
    if workspace is not None:
        return (str(workspace.root / ".env"),)

    files: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        files.append(resolved)

    pwd = os.environ.get("PWD")
    if pwd:
        _add(Path(pwd) / ".env")
    _add(Path.cwd() / ".env")
    if files:
        return tuple(str(path) for path in files)
    return (".env",)


def kaggle_credentials_setup_hint() -> str:
    """Plain-English SOP when Kaggle auth fails or credentials are missing."""
    from labpilot.workspace import discover_workspace

    workspace = discover_workspace()
    env_path = (
        workspace.root / ".env"
        if workspace is not None
        else Path.cwd() / ".env"
    )
    return (
        "Kaggle authentication failed.\n"
        "\n"
        "Set up credentials in the competition workspace (not the LabPilot repo):\n"
        f"  1. Create {env_path}\n"
        "  2. Add:  KAGGLE_API_TOKEN=<token>\n"
        "     Get a token: https://www.kaggle.com/settings  → API → Create New Token\n"
        "  3. Join the competition on Kaggle and accept the rules\n"
        "  4. Re-run your command from the workspace directory\n"
        "\n"
        "Optional: save the same token to ~/.kaggle/access_token instead.\n"
        "Legacy: KAGGLE_USERNAME + KAGGLE_KEY in that .env also work.\n"
        "See docs/SOP.md § Credentials."
    )


def _package_default_config_path() -> Path:
    """Repo ``configs/default.yaml`` (not CWD), so competition workspaces can overlay."""
    repo_default = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    if repo_default.is_file():
        return repo_default
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
    llm = raw.get("llm")
    if isinstance(llm, dict):
        cache = llm.get("cache")
        if isinstance(cache, dict) and "path" in cache:
            cache["path"] = Path(cache["path"])
    return raw



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

    if settings.labpilot_llm_mode:
        config.llm.mode = settings.labpilot_llm_mode.strip().lower()
    if settings.ollama_host:
        config.llm.ollama_base_url = settings.ollama_host.strip()

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


def load_config(path: Path | None = None) -> AppConfig:
    """Load config with layered precedence:

    1. Package default (`configs/default.yaml`)
    2. Explicit CLI `--config` file (if different from package default)
    3. Environment variables (`LABPILOT_*`, credentials)
    """
    layers: list[dict[str, Any]] = []

    package_default = _package_default_config_path()
    if package_default.is_file():
        layers.append(_load_yaml_dict(package_default))

    explicit_path = path or package_default
    if explicit_path.is_file():
        resolved_explicit = explicit_path.resolve()
        if resolved_explicit != package_default.resolve():
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
) -> Path:
    if competitions_dir is not None:
        return competitions_dir
    return Path(__file__).resolve().parents[3] / "configs" / "competitions"


def resolve_runtimes_dir(config: AppConfig) -> Path:
    return config.runtime.runtimes_dir
