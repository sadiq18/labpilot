import logging
from pathlib import Path
from typing import Any

from labpilot.research_engine.shared.experiments.store import ExperimentRecord, ExperimentStore

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Log experiment params, metrics, and artifact paths."""

    def __init__(self, run_dir: Path) -> None:
        self.store = ExperimentStore(run_dir)

    def log(
        self,
        run_id: str,
        competition: str,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        notes: str = "",
    ) -> Path:
        record = ExperimentRecord(
            run_id=run_id,
            competition=competition,
            metrics=metrics or {},
            params=params or {},
            artifacts=artifacts or [],
            notes=notes,
        )
        path = self.store.save(record)
        logger.info("Logged experiment record for run '%s' to %s", run_id, path)
        return path
