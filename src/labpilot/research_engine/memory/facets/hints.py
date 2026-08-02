"""Shared needle tables for rules / code / dataset extractors (not a taxonomy product)."""

from __future__ import annotations

# Modality / technique keyword hints — uncertain by design.
MODALITY_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("audio", ("audio", "bird", "sound", "spectrogram", "clef")),
    ("image", ("image", "vision", "cv", "detect", "segment")),
    ("text", ("text", "nlp", "llm", "language", "token")),
    ("tabular", ("tabular", "table", "xgboost", "lightgbm", "catboost")),
    ("time_series", ("time", "forecast", "series", "temporal")),
]

TECHNIQUE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("augmentation", ("augment", "specaugment", "mixup", "cutmix", "ema")),
    ("ensemble", ("ensemble", "stack", "blend")),
    ("imbalance", ("imbalance", "minority", "class_weight", "focal")),
    ("finetune", ("finetune", "fine-tune", "pretrained", "transfer")),
    ("features", ("feature", "embedding", "descriptor")),
]

CODE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("audio", ("librosa", "torchaudio", "melspectrogram", "spectrogram", "soundfile", "wavfile")),
    ("image", ("torchvision", "albumentations", "cv2", "pil.", "efficientnet", "resnet")),
    ("tabular", ("xgboost", "lightgbm", "catboost", "pandas", "sklearn")),
    ("text", ("transformers", "tokenizers", "sentencepiece", "spacy")),
    ("augmentation", ("mixup", "cutmix", "specaugment", "albumentations")),
    ("ensemble", ("votingclassifier", "stacking")),
]

DATASET_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("audio", (".wav", ".flac", ".ogg", ".mp3", "audio/")),
    ("image", (".jpg", ".jpeg", ".png", ".tif", "images/")),
    ("tabular", (".parquet", ".csv", ".feather")),
    ("text", (".txt", ".jsonl", "transcript")),
]

CONF_ONE = 0.45
CONF_TWO = 0.65
CONF_THREE_PLUS = 0.82
CONF_METADATA = 0.88
CONF_TECHNIQUE_FIELD = 0.75
CONF_CODE = 0.8
CONF_DATASET = 0.78
CONF_PAPER = 0.7
CONF_RESULT = 0.72


def confidence_from_hits(n: int) -> float:
    if n >= 3:
        return CONF_THREE_PLUS
    if n == 2:
        return CONF_TWO
    return CONF_ONE
