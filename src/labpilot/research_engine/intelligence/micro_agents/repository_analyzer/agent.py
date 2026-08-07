"""Extract transferable ML engineering knowledge from cached repository text."""

from __future__ import annotations

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
from labpilot.research_engine.intelligence.feature_recipes import (
    coerce_feature_recipes,
    heuristic_feature_recipes,
    merge_feature_recipes,
    recipe_technique_names,
)
from labpilot.research_engine.intelligence.repositories.models import RepoKnowledge

_ARCHITECTURE = (
    "efficientnet",
    "resnet",
    "convnext",
    "transformer",
    "unet",
    "yolo",
    "vit",
    "lstm",
    "cnn",
)
_LOSSES = ("focal loss", "cross entropy", "bce", "dice loss", "lovasz", "asymmetric loss")
_AUGMENTATION = (
    "specaugment",
    "mixup",
    "cutmix",
    "albumentations",
    "random crop",
    "time masking",
    "frequency masking",
)
_TRICKS = ("ema", "swa", "amp", "mixed precision", "cosine annealing", "warmup")


class RepositoryAnalyzerAgent(BaseMicroAgent):
    name = "RepositoryAnalyzerAgent"
    output_model = RepoKnowledge

    def system_prompt(self) -> str:
        return (
            "Extract transferable ML engineering knowledge from cached repository "
            "text. Respond ONLY with JSON: {\"repo_id\":str,\"full_name\":str,"
            "\"architecture\":[str],\"loss\":[str],\"augmentation\":[str],"
            "\"training_tricks\":[str],\"interesting_files\":[str],"
            "\"dependencies\":[str],\"techniques\":[str],"
            "\"feature_recipes\":[{\"name\":str,\"description\":str,"
            "\"inputs\":[str],\"outputs\":[str],\"transform\":str}],"
            "\"confidence\":float,"
            "\"grounded_in\":\"readme|code_excerpt|deps|mixed\"}. "
            "For feature engineering, capture concrete new features created "
            "(name, inputs, outputs, transform) — including arithmetic/derived "
            "columns such as new=f1+f2 or new=f1/f2 when present in code/text, "
            "not only encoders/scalers. You decide which creations are worth "
            "recording; omit unsupported inventions. "
            "Do not summarize the repository and do not infer unsupported claims."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return (
            f"Competition: {context.competition}\n"
            f"Repository identity: {context.data.get('full_name', '')}\n"
            f"Cached repository text:\n{context.text}"
        )


def _find_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]
