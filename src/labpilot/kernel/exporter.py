import json
import logging
import re
from pathlib import Path

from labpilot.competition.models import CompetitionSpec

logger = logging.getLogger(__name__)

_LABPILOT_METRIC_IMPORT = "from labpilot.evaluation.metrics import compute_metric"

_COMPUTE_METRIC_STUB = '''
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    roc_auc_score,
)


def compute_metric(y_true, y_pred, metric_name, y_proba=None, *, num_classes=None):
    name = metric_name.lower()
    if name in ("auc", "roc_auc", "roc-auc"):
        if num_classes is not None and num_classes > 2:
            return float(accuracy_score(y_true, y_pred))
        if y_proba is None:
            raise ValueError("AUC requires probability predictions")
        return float(roc_auc_score(y_true, y_proba))
    if name in ("logloss", "log_loss"):
        if num_classes is not None and num_classes > 2:
            return float(accuracy_score(y_true, y_pred))
        if y_proba is None:
            raise ValueError("Log loss requires probability predictions")
        return float(log_loss(y_true, y_proba))
    if name in ("accuracy", "acc"):
        return float(accuracy_score(y_true, y_pred))
    if name == "f1":
        average = "binary" if (num_classes is None or num_classes <= 2) else "macro"
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    if name in ("rmse", "root_mean_squared_error"):
        return float(root_mean_squared_error(y_true, y_pred))
    if name == "mse":
        return float(mean_squared_error(y_true, y_pred))
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name == "rmsle":
        if np.any(y_true < 0) or np.any(y_pred < 0):
            return float(root_mean_squared_error(y_true, y_pred))
        return float(
            root_mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0)))
        )
    raise ValueError(f"Unsupported metric: {metric_name}")
'''


def slugify_kernel_id(title: str, *, max_length: int = 30) -> str:
    """Build a Kaggle-valid kernel slug from a title."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if not slug:
        slug = "labpilot-baseline"
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug or "labpilot-baseline"


def build_kernel_metadata(
    competition: CompetitionSpec,
    *,
    username: str | None = None,
    run_suffix: str | None = None,
) -> tuple[str, dict]:
    """Return (kernel_id, metadata dict) with a valid Kaggle slug."""
    title = competition.title or competition.slug
    slug = slugify_kernel_id(title)
    if run_suffix:
        suffix = slugify_kernel_id(run_suffix, max_length=12)
        slug = slugify_kernel_id(f"{slug}-{suffix}", max_length=50)

    if username:
        kernel_id = f"{username}/{slug}"
    else:
        kernel_id = slug

    metadata = {
        "id": kernel_id,
        "title": f"{title} — LabPilot baseline",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [competition.slug],
        "kernel_sources": [],
    }
    return kernel_id, metadata


def export_kernel(
    run_dir: Path,
    competition: CompetitionSpec,
    *,
    username: str | None = None,
) -> Path:
    """Write `run_dir/kernel/` with metadata and a Kaggle-ready training script."""
    train_path = run_dir / "pipeline" / "train.py"
    if not train_path.is_file():
        raise FileNotFoundError(f"Generated training script not found: {train_path}")

    kernel_dir = run_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    kaggle_input = f"/kaggle/input/{competition.slug}"
    kaggle_working = "/kaggle/working"
    run_py = _adapt_train_script(train_path.read_text(), kaggle_input, kaggle_working)
    (kernel_dir / "run.py").write_text(run_py)

    run_suffix = run_dir.name.split("-")[-1] if run_dir.name else None
    kernel_id, metadata = build_kernel_metadata(
        competition,
        username=username,
        run_suffix=run_suffix,
    )
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Exported kernel artifacts to %s (id=%s)", kernel_dir, kernel_id)
    return kernel_dir


def _adapt_train_script(source: str, data_dir: str, output_dir: str) -> str:
    text = source.replace(_LABPILOT_METRIC_IMPORT, _COMPUTE_METRIC_STUB.strip())
    text = re.sub(
        r"^DATA_DIR = Path\(.+\)$",
        f'DATA_DIR = Path({data_dir!r})',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^OUTPUT_DIR = Path\(.+\)$",
        f'OUTPUT_DIR = Path({output_dir!r})',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    header = (
        '"""LabPilot kernel script — generated from pipeline/train.py."""\n\n'
        "import sys\n"
        "from pathlib import Path\n\n"
        "if __name__ == '__main__' and __package__ is None:\n"
        "    sys.path.insert(0, str(Path(__file__).resolve().parent))\n\n"
    )
    if not text.startswith('"""'):
        text = header + text
    return text
