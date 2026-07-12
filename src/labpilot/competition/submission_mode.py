"""Derive submission mode and apply rules-page fallbacks for kernel-only comps."""

import logging
import re

from labpilot.competition.models import CompetitionSpec

logger = logging.getLogger(__name__)

_KERNEL_RULE_PATTERNS = (
    re.compile(r"kernels?\s+only", re.I),
    re.compile(r"notebook\s+submission", re.I),
    re.compile(r"code\s+competition", re.I),
    re.compile(r"submit\s+(?:a|your)\s+(?:notebook|kernel)", re.I),
)


def detect_kernel_only_from_rules(text: str) -> bool:
    if not text.strip():
        return False
    return any(pattern.search(text) for pattern in _KERNEL_RULE_PATTERNS)


def apply_submission_mode(spec: CompetitionSpec) -> CompetitionSpec:
    """Set submission_mode from API flag or rules excerpt; fill submissions_url."""
    from labpilot.kaggle.urls import competition_submissions_url

    kernels_only = spec.is_kernels_submissions_only
    if not kernels_only and spec.raw_html and detect_kernel_only_from_rules(spec.raw_html):
        logger.warning(
            "Rules text suggests kernel-only submissions for '%s'; enabling kernel mode.",
            spec.slug,
        )
        kernels_only = True

    mode: str = "kernel" if kernels_only else "csv"
    updates: dict = {
        "submission_mode": mode,
        "is_kernels_submissions_only": kernels_only,
        "submissions_url": spec.submissions_url or competition_submissions_url(spec.slug),
    }
    return spec.model_copy(update=updates)
