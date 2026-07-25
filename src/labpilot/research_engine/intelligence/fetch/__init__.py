"""Kaggle catalog fetch — kernels + discussions into research artifacts.

Library boundary for ``research fetch`` and a future cron/worker. No Typer/Rich.
"""

from labpilot.research_engine.intelligence.fetch.models import FetchResult
from labpilot.research_engine.intelligence.fetch.service import KaggleFetchService

__all__ = ["FetchResult", "KaggleFetchService"]
