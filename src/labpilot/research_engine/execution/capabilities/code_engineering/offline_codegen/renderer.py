import ast
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.research_engine.execution.baseline.registry import BaselineTemplate
from labpilot.research_engine.execution.baseline.selector import BaselineChoice
from labpilot.config import TrainingConfig
from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen.defaults import (
    DEFAULT_TABULAR_MODEL_PARAMS,
)

logger = logging.getLogger(__name__)


class CodeRenderer:
    """Render baseline training code from Jinja2 templates."""

    def __init__(self, training_config: TrainingConfig) -> None:
        self.training_config = training_config

    def render(
        self,
        template: BaselineTemplate,
        choice: BaselineChoice,
        run_dir: Path,
        *,
        model_params: dict[str, Any] | None = None,
        feature_recipes: list[str] | None = None,
        target_encoding_columns: list[str] | None = None,
        log_numeric_columns: list[str] | None = None,
    ) -> Path:
        logger.info("Rendering template '%s' into %s/pipeline", template.name, run_dir)
        env = Environment(
            loader=FileSystemLoader(template.template_dir),
            autoescape=select_autoescape(default=False),
        )

        resolved_params = dict(DEFAULT_TABULAR_MODEL_PARAMS)
        resolved_params["random_state"] = self.training_config.random_seed
        if model_params:
            resolved_params.update(model_params)
        resolved_params.setdefault("random_state", self.training_config.random_seed)

        context = {
            "competition": run_dir.name,
            "choice": choice,
            "cv_folds": self.training_config.cv_folds,
            "random_seed": self.training_config.random_seed,
            "data_dir": str(run_dir / "data" / "raw"),
            "output_dir": str(run_dir),
            "max_images_sample": 5_000,
            "model_params": resolved_params,
            # Partitioned datasets often ship several tables per entity; the
            # most common suffix is the one carrying the target rows.
            "primary_kind": max(
                choice.partition_kinds, key=lambda k: choice.partition_kinds[k]
            )
            if getattr(choice, "partition_kinds", None)
            else "",
            "feature_recipes": feature_recipes or [],
            "target_encoding_columns": target_encoding_columns or [],
            "log_numeric_columns": log_numeric_columns or [],
            "deep": {
                "max_epochs": 3,
                "max_train_samples": 5_000,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "cpu_max_epochs": 2,
                "cpu_max_train_samples": 2_000,
                "cv_folds": 3,
            },
        }

        pipeline_dir = run_dir / "pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        for template_file in template.template_dir.glob("*.j2"):
            rendered_name = template_file.name.removesuffix(".j2")
            rendered = env.get_template(template_file.name).render(**context)
            (pipeline_dir / rendered_name).write_text(rendered)
            logger.debug("Rendered %s", pipeline_dir / rendered_name)

        return pipeline_dir


def validate_python_syntax(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        ast.parse(path.read_text())
    except SyntaxError as exc:
        errors.append(f"{path}: {exc}")
    return errors


def validate_pipeline(pipeline_dir: Path) -> list[str]:
    errors: list[str] = []
    train_script = pipeline_dir / "train.py"
    if not train_script.exists():
        errors.append(f"Missing train.py in {pipeline_dir}")
    else:
        errors.extend(validate_python_syntax(train_script))
    return errors
