"""Kaggle API client + URL helpers (shared platform I/O)."""

from labpilot.accessor.kaggle.client import KaggleClient, KaggleGateway, SubmissionResult
from labpilot.accessor.kaggle.models import CompetitionMetadata
from labpilot.accessor.kaggle.urls import (
    competition_submissions_url,
    kernel_notebook_url,
    parse_kernel_ref,
)

__all__ = [
    "CompetitionMetadata",
    "KaggleClient",
    "KaggleGateway",
    "SubmissionResult",
    "competition_submissions_url",
    "kernel_notebook_url",
    "parse_kernel_ref",
]
