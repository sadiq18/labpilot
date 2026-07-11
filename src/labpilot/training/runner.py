import json
import subprocess
import sys
from pathlib import Path


class TrainingRunner:
    """Execute the generated training pipeline as a subprocess."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.pipeline_dir = run_dir / "pipeline"
        self.train_script = self.pipeline_dir / "train.py"

    def run(self, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        if not self.train_script.exists():
            raise FileNotFoundError(f"Training script not found: {self.train_script}")

        return subprocess.run(
            [sys.executable, str(self.train_script)],
            cwd=self.pipeline_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

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
