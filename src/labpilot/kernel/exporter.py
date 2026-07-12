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
    _ = run_suffix  # Stable slug per competition; run id must not change kernel identity.
    slug = slugify_kernel_id(f"{competition.slug[:20]}-labpilot", max_length=50)
    # Kaggle validates that the title slugifies to the kernel slug in metadata id.
    kernel_title = slug.replace("-", " ").title()

    if username:
        kernel_id = f"{username}/{slug}"
    else:
        kernel_id = slug

    metadata = {
        "id": kernel_id,
        "title": kernel_title,
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

    kaggle_input = f"/kaggle/input/competitions/{competition.slug}"
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


_KAGGLE_BOOTSTRAP = '''
def _labpilot_image_search_bases() -> list[Path]:
    """Candidate directories for image files on Kaggle after zip extraction."""
    bases = [DATA_DIR / IMAGE_DIR, OUTPUT_DIR / IMAGE_DIR]
    for rel in ("train", "test", "train/train", "test/test"):
        bases.extend([DATA_DIR / rel, OUTPUT_DIR / rel])
    seen: set[str] = set()
    unique: list[Path] = []
    for base in bases:
        key = str(base)
        if key not in seen:
            seen.add(key)
            unique.append(base)
    return unique


def _labpilot_prepare_kaggle_data() -> None:
    """Extract nested competition zips and validate CSV paths on Kaggle."""
    import os
    import zipfile

    global DATA_DIR

    if not str(DATA_DIR).startswith("/kaggle/input"):
        return

    os.environ.setdefault("TORCH_HOME", str(OUTPUT_DIR / ".torch"))

    input_root = Path("/kaggle/input")
    if not DATA_DIR.is_dir():
        matches: list[Path] = []
        for root in (DATA_DIR, input_root):
            if root.is_dir():
                matches.extend(sorted(root.rglob(TRAIN_FILE)))
        if matches:
            DATA_DIR = matches[0].parent
        elif input_root.is_dir():
            available = sorted(p.name for p in input_root.iterdir())
            raise FileNotFoundError(
                f"Competition data directory not found: {DATA_DIR}. "
                f"Top-level input entries: {available}. "
                "Confirm kernel-metadata.json lists competition_sources and "
                "you have accepted the competition rules on Kaggle."
            )
        else:
            raise FileNotFoundError(
                f"Competition data directory not found: {DATA_DIR}. "
                "Confirm kernel-metadata.json lists competition_sources and "
                "you have accepted the competition rules on Kaggle."
            )

    for archive in sorted(DATA_DIR.glob("*.zip")):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(OUTPUT_DIR)

    for name in (TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE):
        if (DATA_DIR / name).is_file():
            continue
        matches = []
        for root in (DATA_DIR, OUTPUT_DIR, input_root):
            if root.is_dir():
                matches.extend(sorted(root.rglob(name)))
        if matches and name == TRAIN_FILE:
            DATA_DIR = matches[0].parent
            break
        if not matches:
            available = sorted(p.name for p in DATA_DIR.iterdir())
            raise FileNotFoundError(
                f"Required file {name!r} not found under {DATA_DIR} or {OUTPUT_DIR}. "
                f"Available in input: {available}. "
                "Ensure competition_sources includes this competition in kernel-metadata.json."
            )


def _labpilot_resolve_image_path(value: str) -> Path | None:
    candidates = []
    for base in _labpilot_image_search_bases():
        candidates.extend(
            [base / value, base / f"{value}.jpg", base / f"{value}.jpeg", base / f"{value}.png"]
        )
    for path in candidates:
        if path.is_file():
            return path
    return None
'''


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
    text = _inject_kaggle_bootstrap(text)
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


def _inject_kaggle_bootstrap(source: str) -> str:
    """Inject Kaggle zip extraction and multi-root image lookup into train.py."""
    if "def load_data()" not in source:
        return source

    text = source
    if "_labpilot_prepare_kaggle_data" not in text:
        text = text.replace(
            "\n\ndef load_data()",
            f"\n\n{_KAGGLE_BOOTSTRAP.strip()}\n\n\ndef load_data()",
            1,
        )

    resolve_pattern = re.compile(
        r"def resolve_image_path\(value: str\) -> Path \| None:.*?(?=\n\n(?:def |class ))",
        re.DOTALL,
    )
    replacement = (
        "def resolve_image_path(value: str) -> Path | None:\n"
        "    return _labpilot_resolve_image_path(value)\n"
    )
    text, count = resolve_pattern.subn(replacement, text, count=1)
    if count == 0 and "def resolve_image_path" in text:
        logger.warning("Could not patch resolve_image_path for Kaggle bootstrap.")

    if "    _labpilot_prepare_kaggle_data()" not in text:
        text = re.sub(
            r"(def main\(\) -> None:\n)",
            r"\1    _labpilot_prepare_kaggle_data()\n",
            text,
            count=1,
        )
    return text
