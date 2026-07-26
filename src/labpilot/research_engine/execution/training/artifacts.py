from pathlib import Path


def list_model_files(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    return (
        sorted(models_dir.glob("*.joblib"))
        + sorted(models_dir.glob("*.pkl"))
        + sorted(models_dir.glob("*.txt"))
    )


def oof_exists(run_dir: Path) -> bool:
    return (run_dir / "oof.csv").exists()
