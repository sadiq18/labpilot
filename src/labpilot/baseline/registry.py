from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineTemplate:
    name: str
    problem_type: str
    template_dir: Path
    description: str
    model_family: str = "lightgbm"


def get_templates_root() -> Path:
    return Path(__file__).resolve().parents[3] / "templates"


def list_templates() -> list[BaselineTemplate]:
    root = get_templates_root()
    templates: list[BaselineTemplate] = []

    registry = {
        "tabular_classification": ("tabular_classification", "LightGBM classifier"),
        "tabular_regression": ("tabular_regression", "LightGBM regressor"),
    }

    for problem_type, (dirname, desc) in registry.items():
        template_dir = root / dirname
        if template_dir.exists():
            templates.append(
                BaselineTemplate(
                    name=dirname,
                    problem_type=problem_type,
                    template_dir=template_dir,
                    description=desc,
                )
            )

    return templates


def get_template(problem_type: str) -> BaselineTemplate | None:
    for template in list_templates():
        if template.problem_type == problem_type:
            return template
    return None
