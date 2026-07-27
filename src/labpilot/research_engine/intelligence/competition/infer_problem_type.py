"""Infer ``ProblemType`` from competition text when Kaggle leaves it unknown."""

from __future__ import annotations

import re

from labpilot.research_engine.intelligence.competition.models import ProblemType

# Ordered: first match wins (more specific before generic "classification").
_IMAGE_PATTERNS = (
    r"\bcomputer[- ]?vision\b",
    r"\bimage(?:s)?\b",
    r"\bvideo(?:s)?\b",
    r"\bobject[- ]?detection\b",
    r"\binstanc(?:e|es)?[- ]?segmentation\b",
    r"\bsegmentation\b",
    r"\btracking\b",
    r"\bmicroscopy\b",
    r"\b3d\+?time\b",
    r"\b4d\b",
    r"\bzarr\b",
    r"\btiff?\b",
    r"\bresnet\b",
    r"\bcnn\b",
)
_TEXT_PATTERNS = (
    r"\bnlp\b",
    r"\bnatural language\b",
    r"\btext classification\b",
    r"\bsentiment\b",
    r"\btweet\b",
    r"\btransformer\b",
    r"\bbert\b",
)
_REGRESSION_PATTERNS = (
    r"\bregression\b",
    r"\brmse\b",
    r"\brmsle\b",
    r"\bmae\b",
    r"\bmse\b",
    r"\bmean[- ]?squared(?:[- ]?error)?\b",
    r"\bsquared[- ]?error\b",
)
_CLASSIFICATION_PATTERNS = (
    r"\bclassif(?:y|ication)\b",
    r"\baccuracy\b",
    r"\blog[- ]?loss\b",
    r"\bauc\b",
    r"\bf1\b",
)
# Strong vision signals — incidental "image" mentions / multimodal tags alone
# should not beat an explicit regression metric (e.g. ROGII geology + MSE).
_STRONG_IMAGE_PATTERNS = (
    r"\bcomputer[- ]?vision\b",
    r"\bobject[- ]?detection\b",
    r"\binstanc(?:e|es)?[- ]?segmentation\b",
    r"\bsegmentation\b",
    r"\btracking\b",
    r"\bmicroscopy\b",
    r"\b3d\+?time\b",
    r"\b4d\b",
    r"\bzarr\b",
)


def _blob(*parts: str) -> str:
    return " ".join(p.strip().lower() for p in parts if p and p.strip())


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in patterns)


def infer_problem_type_from_metadata(
    *,
    title: str = "",
    description: str = "",
    tags: list[str] | None = None,
    metric_name: str = "",
    metric_description: str = "",
) -> ProblemType:
    """Best-effort problem type from competition metadata (no dataset required).

    Returns ``UNKNOWN`` when signals are absent or conflicting without a clear
    winner — callers should then consult the dataset profile.
    """
    tag_text = " ".join(tags or [])
    text = _blob(title, description, tag_text, metric_name, metric_description)
    if not text.strip():
        return ProblemType.UNKNOWN

    image = _matches(text, _IMAGE_PATTERNS)
    strong_image = _matches(text, _STRONG_IMAGE_PATTERNS)
    textish = _matches(text, _TEXT_PATTERNS)
    regression = _matches(text, _REGRESSION_PATTERNS)
    classification = _matches(text, _CLASSIFICATION_PATTERNS)

    # Explicit regression metrics beat weak/incidental image signals (PNG
    # previews beside well-log CSVs, "multimodal" tags, etc.).
    if regression and not classification and not strong_image:
        return ProblemType.TABULAR_REGRESSION

    # Vision / video / tracking dominate even if "classification" also appears.
    if image and not textish:
        return ProblemType.IMAGE_CLASSIFICATION
    if textish and not image:
        return ProblemType.TEXT_CLASSIFICATION
    if image and textish:
        # Prefer vision when both fire — multimodal is closer to image baselines.
        return ProblemType.IMAGE_CLASSIFICATION
    if regression and not classification:
        return ProblemType.TABULAR_REGRESSION
    if classification and not regression:
        return ProblemType.TABULAR_CLASSIFICATION
    if regression:
        return ProblemType.TABULAR_REGRESSION
    return ProblemType.UNKNOWN
