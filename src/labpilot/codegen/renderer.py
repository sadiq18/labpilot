import ast
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.baseline.registry import BaselineTemplate
from labpilot.baseline.selector import BaselineChoice
from labpilot.config import TrainingConfig

logger = logging.getLogger(__name__)


class CodeRenderer:
    """Render baseline training code from Jinja2 templates."""

    def __init__(self, training_config: TrainingConfig) -> None:
        self.training_config = training_config

    def render(self, template: BaselineTemplate, choice: BaselineChoice, run_dir: Path) -> Path:
        logger.info("Rendering template '%s' into %s/pipeline", template.name, run_dir)
        env = Environment(
            loader=FileSystemLoader(template.template_dir),
            autoescape=select_autoescape(default=False),
        )

        context = {
            "competition": run_dir.name,
            "choice": choice,
            "cv_folds": self.training_config.cv_folds,
            "random_seed": self.training_config.random_seed,
            "data_dir": str(run_dir / "data" / "raw"),
            "output_dir": str(run_dir),
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
