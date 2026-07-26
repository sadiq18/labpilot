"""Extract transferable ML engineering knowledge from cached repository text."""

from __future__ import annotations

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
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
            "\"dependencies\":[str],\"techniques\":[str],\"confidence\":float,"
            "\"grounded_in\":\"readme|code_excerpt|deps|mixed\"}. "
            "Do not summarize the repository and do not infer unsupported claims."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return (
            f"Competition: {context.competition}\n"
            f"Repository identity: {context.data.get('full_name', '')}\n"
            f"Cached repository text:\n{context.text}"
        )

    def _run_rule_engine(self, context: StructuredContext) -> RepoKnowledge:
        d = context.data
        text = context.text.lower()
        architecture = coerce_str_list(d.get("architecture")) or _find_terms(
            text, _ARCHITECTURE
        )
        losses = coerce_str_list(d.get("loss")) or _find_terms(text, _LOSSES)
        augmentation = coerce_str_list(d.get("augmentation")) or _find_terms(
            text, _AUGMENTATION
        )
        tricks = coerce_str_list(d.get("training_tricks")) or _find_terms(
            text, _TRICKS
        )
        dependencies = coerce_str_list(d.get("dependencies"))
        techniques = list(
            dict.fromkeys(
                [
                    *coerce_str_list(d.get("techniques")),
                    *architecture,
                    *losses,
                    *augmentation,
                    *tricks,
                ]
            )
        )
        files = coerce_str_list(
            d.get("interesting_files") or d.get("files_worth_reading")
        )
        sources = sum(
            bool(value)
            for value in (
                d.get("has_readme"),
                files,
                dependencies,
            )
        )
        grounded = "mixed" if sources > 1 else (
            "code_excerpt" if files else "deps" if dependencies else "readme"
        )
        return RepoKnowledge(
            repo_id=str(d.get("repo_id") or ""),
            full_name=str(d.get("full_name") or ""),
            architecture=architecture,
            loss=losses,
            augmentation=augmentation,
            training_tricks=tricks,
            interesting_files=files,
            dependencies=dependencies,
            techniques=techniques,
            confidence=0.65 if techniques else 0.35,
            grounded_in=grounded,
        )


def _find_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]
