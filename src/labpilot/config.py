from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    # Shared across runs and keyed by competition slug, so re-running the same
    # competition doesn't re-download identical data every time.
    cache_dir: Path = Path(".cache/kaggle")
    # Kaggle scores submissions asynchronously; after upload we poll
    # `competition_submissions` for up to `submission_poll_timeout` seconds,
    # checking every `submission_poll_interval` seconds, to persist the real
    # public leaderboard score instead of leaving it null.
    submission_poll_timeout: int = 120
    submission_poll_interval: int = 5
    kernel_poll_timeout: int = 3600
    kernel_poll_interval: int = 15
    api_token: str = Field(default="", exclude=True, repr=False)
    username: str = Field(default="", exclude=True, repr=False)
    key: str = Field(default="", exclude=True, repr=False)


class PipelineConfig(BaseModel):
    stages: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    runs_dir: Path = Path("runs")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    deep_baseline: DeepBaselineConfig = Field(default_factory=DeepBaselineConfig)
    kaggle: KaggleConfig = Field(default_factory=KaggleConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kaggle_api_token: str = ""
    kaggle_username: str = ""
    kaggle_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    labpilot_runs_dir: str = "runs"
    labpilot_llm_provider: str = ""
    labpilot_llm_model: str = ""


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or Path("configs/default.yaml")
    raw = {}
    if config_path.exists():
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}

    if "runs_dir" in raw:
        raw["runs_dir"] = Path(raw["runs_dir"])

    config = AppConfig.model_validate(raw)
    settings = Settings()

    config.kaggle.api_token = settings.kaggle_api_token
    config.kaggle.username = settings.kaggle_username
    config.kaggle.key = settings.kaggle_key
    if settings.labpilot_runs_dir != "runs":
        config.runs_dir = Path(settings.labpilot_runs_dir)

    if settings.labpilot_llm_provider:
        config.llm.provider = settings.labpilot_llm_provider
    elif settings.gemini_api_key.strip() and not settings.openai_api_key.strip():
        config.llm.provider = "gemini"
    elif settings.openai_api_key.strip() and not settings.gemini_api_key.strip():
        config.llm.provider = "openai"
    if settings.labpilot_llm_model:
        config.llm.model = settings.labpilot_llm_model
    elif "model" not in raw.get("llm", {}):
        # No explicit model in the config file or env — resolve the right
        # default for whichever provider ended up configured, instead of
        # leaving LLMConfig's dataclass default (an OpenAI model name) in
        # place for a provider it doesn't apply to.
        config.llm.model = DEFAULT_MODEL_BY_PROVIDER.get(config.llm.provider, config.llm.model)

    config.llm.api_key = {
        "openai": settings.openai_api_key,
        "gemini": settings.gemini_api_key,
    }.get(config.llm.provider.strip().lower(), "")

    return config
