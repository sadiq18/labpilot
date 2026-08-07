import json
import logging
import subprocess
import sys
from pathlib import Path

from labpilot.research_engine.execution.training.environment import (
    child_environment,
    declared_dependencies,
    training_command,
)

logger = logging.getLogger(__name__)


class TrainingRunner:
    """Execute the generated training pipeline as a subprocess."""

    def __init__(self, run_dir: Path) -> None:
        # Workspace root is the process cwd so generated scripts can open
        # ``pipeline/config.yaml`` and write ``metrics.json`` / ``submission.csv``
        # at the competition root (same contract as Code Engineer prompts).
        self.run_dir = Path(run_dir).resolve()
        self.pipeline_dir = self.run_dir / "pipeline"
        self.train_script = self.pipeline_dir / "train.py"

    def run(self, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        if not self.train_script.exists():
            raise FileNotFoundError(f"Training script not found: {self.train_script}")

        cmd = training_command(self.train_script, python=sys.executable)
        deps = declared_dependencies(self.train_script)
        logger.info(
            "Running training script %s (cwd=%s, via=%s%s)",
            self.train_script,
            self.run_dir,
            cmd[0],
            f", deps={deps}" if deps else "",
        )
        result = subprocess.run(
            cmd,
            cwd=self.run_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            # Generated code runs without the operator's provider or Kaggle
            # credentials. It needs data on disk, not API access.
            env=child_environment(),
        )
        logger.info("Training script finished with return code %d", result.returncode)
        return result

    def collect_artifacts(self) -> dict[str, Path]:
        artifacts: dict[str, Path] = {}
        models_dir = self.run_dir / "models"
        oof_path = self.run_dir / "oof.csv"
        metrics_path = self.run_dir / "metrics.json"

        if models_dir.exists():
            artifacts["models"] = models_dir
        if oof_path.exists():
            artifacts["oof"] = oof_path
        if metrics_path.exists():
            artifacts["metrics"] = metrics_path

        return artifacts

    def save_run_log(self, result: subprocess.CompletedProcess[str]) -> Path:
        log_path = self.run_dir / "training.log"
        log_path.write_text(
            json.dumps(
                {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
            )
        )
        return log_path
