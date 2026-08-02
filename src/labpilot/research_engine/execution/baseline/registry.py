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
    return (
        Path(__file__).resolve().parents[1]
        / "capabilities"
        / "code_engineering"
        / "templates"
    )


def list_templates() -> list[BaselineTemplate]:
    root = get_templates_root()
    templates: list[BaselineTemplate] = []

    # (problem_type, dirname, description, model_family). A problem type may
    # have several templates; the first one listed is its default, and a
    # specific `template_name` selects the others.
    registry = [
        ("tabular_classification", "tabular_classification", "LightGBM classifier", "lightgbm"),
        ("tabular_regression", "tabular_regression", "LightGBM regressor", "lightgbm"),
        (
            "tabular_regression",
            "tabular_regression_partitioned",
            "Partition-aware LightGBM for predict-forward tasks",
            "lightgbm",
        ),
        (
            "text_classification",
            "text_classification",
            "TF-IDF + Logistic Regression",
            "sklearn",
        ),
        (
            "image_classification",
            "image_classification",
            "ResNet18 features + LightGBM",
            "lightgbm",
        ),
        (
            "text_classification_deep",
            "text_classification_deep",
            "Fine-tuned DistilBERT (transfer learning)",
            "transformers",
        ),
        (
            "image_classification_deep",
            "image_classification_deep",
            "Fine-tuned ResNet18 (transfer learning)",
            "torch",
        ),
    ]

    for problem_type, dirname, desc, family in registry:
        template_dir = root / dirname
        if template_dir.exists():
            templates.append(
                BaselineTemplate(
                    name=dirname,
                    problem_type=problem_type,
                    template_dir=template_dir,
                    description=desc,
                    model_family=family,
                )
            )

    return templates


def get_template(
    problem_type: str, template_name: str | None = None
) -> BaselineTemplate | None:
    for template in list_templates():
        if template_name and template.name == template_name:
            return template
        if template_name is None and template.problem_type == problem_type:
            return template
    return None
