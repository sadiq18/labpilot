"""Deterministic, bounded profile of the local competition code."""

from __future__ import annotations

import json
from pathlib import Path

from labpilot.research_engine.intelligence.models import AnalyzeContext
from labpilot.research_engine.intelligence.repositories.models import LocalCodeProfile
from labpilot.research_engine.intelligence.repositories.provider import parse_dependencies

_ARCHITECTURE = ("efficientnet", "resnet", "convnext", "transformer", "unet", "vit", "cnn")
_LOSS = ("focal loss", "cross entropy", "bce", "dice loss", "lovasz", "asymmetric loss")
_AUGMENTATION = ("specaugment", "mixup", "cutmix", "albumentations", "random crop")
_TRICKS = ("ema", "swa", "amp", "mixed precision", "cosine annealing", "warmup")


class LocalCodeProfiler:
    def profile(self, context: AnalyzeContext) -> LocalCodeProfile | None:
        files = _candidate_files(context)
        if not files:
            return None
        texts: dict[str, str] = {}
        for path in files[:30]:
            try:
                texts[str(path)] = path.read_text(encoding="utf-8", errors="replace")[:40_000]
            except OSError:
                continue
        if not texts:
            return None
        joined = "\n".join(texts.values()).lower()
        return LocalCodeProfile(
            architecture=_terms(joined, _ARCHITECTURE),
            loss=_terms(joined, _LOSS),
            augmentation=_terms(joined, _AUGMENTATION),
            training_tricks=_terms(joined, _TRICKS),
            dependencies=parse_dependencies(texts),
            files_scanned=list(texts),
        )


def _candidate_files(context: AnalyzeContext) -> list[Path]:
    names = {
        "train.py",
        "model.py",
        "models.py",
        "loss.py",
        "augment.py",
        "augmentations.py",
        "requirements.txt",
        "pyproject.toml",
        "environment.yml",
    }
    root = context.runs_dir.parent
    candidates = [root / name for name in names if (root / name).is_file()]
    if context.runs_dir.is_dir():
        run_dirs = sorted(
            (path for path in context.runs_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs[:3]:
            meta_path = run_dir / "competition.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    slug = str(meta.get("slug") or meta.get("competition") or "")
                    if slug and slug != context.competition:
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            for path in run_dir.rglob("*"):
                if path.is_file() and (
                    path.name.lower() in names
                    or path.suffix.lower() in {".yaml", ".yml"}
                    and "config" in path.name.lower()
                ):
                    candidates.append(path)
    return list(dict.fromkeys(candidates))


def _terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]
