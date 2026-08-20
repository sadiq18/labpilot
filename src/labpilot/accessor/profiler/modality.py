"""Rule-based modality detection (tabular / text / image) with optional LLM tie-breaker."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

from labpilot.accessor.profiler.schema import ModalityPresence
from labpilot.accessor.profiler.tabular import ColumnProfile, DatasetProfile

if TYPE_CHECKING:
    from labpilot.llm.client import LLMClient

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".npy",
    ".npz",
}
TEXT_COLUMN_HINTS = {"text", "comment", "review", "description", "body", "content", "message"}
MIN_TEXT_AVG_LENGTH = 60


class ModalityResult(BaseModel):
    modality: str = "tabular"
    confidence: str = "high"  # high | ambiguous
    signals: list[str] = Field(default_factory=list)
    #: How many image-like files were seen. Recorded so `presences` can weigh
    #: images against tables without counting the tree a second time.
    image_count: int = 0
    #: True only when a model actually chose between candidates. The no-LLM
    #: fallback is a *default*, not a decision, and the two must not read alike.
    tiebroken: bool = False
    image_dir: str | None = None
    image_column: str | None = None
    text_column: str | None = None


@dataclass(frozen=True)
class _Scan:
    """What one walk of the tree found. Computed once, read by everything.

    `detect` and `presences` each used to walk it — `_detect_image` twice on its
    own, then `_count_csvs` and `_zarr_store` again — so a single profile made
    about five complete passes over a tree that on rogii holds thousands of
    paths, and sorted all of them to find one directory.
    """

    image_dirs: dict[str, int]
    csv_count: int
    zarr_store: str | None

    @property
    def image_count(self) -> int:
        return sum(self.image_dirs.values())

    @property
    def best_image_dir(self) -> str:
        return max(self.image_dirs, key=self.image_dirs.get) if self.image_dirs else ""


class ModalityDetector:
    def scan(self, data_dir: Path) -> _Scan:
        """One pass: image files by directory, table count, first zarr store."""
        image_dirs: dict[str, int] = {}
        csv_count = 0
        zarr_store: str | None = None
        for path in data_dir.rglob("*"):
            if path.is_dir():
                if zarr_store is None and path.name.endswith(".zarr"):
                    zarr_store = str(path.relative_to(data_dir))
                continue
            suffix = path.suffix.lower()
            if suffix == ".csv":
                csv_count += 1
            elif suffix in IMAGE_EXTENSIONS:
                rel_dir = str(path.parent.relative_to(data_dir))
                image_dirs[rel_dir if rel_dir != "." else ""] = (
                    image_dirs.get(rel_dir if rel_dir != "." else "", 0) + 1
                )
        return _Scan(image_dirs=image_dirs, csv_count=csv_count, zarr_store=zarr_store)

    def presences(
        self, data_dir: Path, profile: DatasetProfile, scan: _Scan | None = None
    ) -> list[ModalityPresence]:
        """Every modality this dataset contains, primary first.

        The list `detect` could never return. rogii is 1,546 per-well tables
        *and* a directory of PNG previews; the old decision counted the images,
        preferred tabular — correctly — and then dropped them, so nothing
        downstream could know they were there. Preference and presence are
        different questions, and only one of them has a single answer.

        Order: the primary, then the auxiliaries in a fixed order, so two runs
        over the same bytes list them the same way.
        """
        found: list[ModalityPresence] = []

        scan = scan or self.scan(data_dir)
        images = self._detect_image(data_dir, profile, scan)
        text = self._detect_text(profile)
        csv_count = scan.csv_count
        zarr = scan.zarr_store

        if csv_count:
            found.append(
                ModalityPresence(
                    modality="tabular", role="auxiliary", detail=f"{csv_count} table(s)"
                )
            )
        if images.image_dir is not None or zarr is not None:
            found.append(
                ModalityPresence(
                    modality="image",
                    role="auxiliary",
                    # A zarr store is reachable here for the first time: the CSV
                    # preference used to return before the branch that looked
                    # for one, and every zarr competition ships a
                    # `sample_submission.csv`.
                    detail=(
                        f"zarr store {zarr}"
                        if zarr
                        else f"{images.image_count} image file(s) in {images.image_dir or '.'}"
                    ),
                    image_dir=zarr or images.image_dir,
                    image_column=images.image_column,
                )
            )
        if text.text_column is not None:
            found.append(
                ModalityPresence(
                    modality="text",
                    role="auxiliary",
                    detail=f"text column {text.text_column!r}",
                    text_column=text.text_column,
                )
            )
        if not found:
            return [
                ModalityPresence(
                    modality="environment",
                    role="primary",
                    detail="no tables, images or text — an environment to act in",
                )
            ]

        primary = self._primary_of(found, scan=scan)
        ordered = [presence for presence in found if presence.modality == primary]
        ordered += [presence for presence in found if presence.modality != primary]
        return [
            presence.model_copy(update={"role": "primary" if index == 0 else "auxiliary"})
            for index, presence in enumerate(ordered)
        ]

    @staticmethod
    def _primary_of(found: list[ModalityPresence], *, scan: _Scan) -> str:
        """Which modality carries the signal. The old rules, stated once.

        Tables win a tie with images because a per-entity layout with previews
        is a tabular problem; images win when there are more of them than tables,
        which is what an image competition looks like.

        **A zarr store wins outright.** The volume *is* the dataset and the CSVs
        beside it are the submission template — the CSV preference used to fire
        first, so a zarr competition came out `tabular` and the branch that would
        have said otherwise was unreachable. Making the store visible without
        letting it decide would have left that outcome exactly as it was.
        """
        kinds = {presence.modality for presence in found}
        if "image" in kinds and (
            scan.zarr_store is not None or not scan.csv_count or scan.image_count > scan.csv_count
        ):
            return "image"
        if "tabular" in kinds:
            return "tabular"
        return next(iter(sorted(kinds)))

    def detect(
        self,
        data_dir: Path,
        profile: DatasetProfile,
        llm_client: "LLMClient | None" = None,
        competition_title: str = "",
        competition_description: str = "",
    ) -> ModalityResult:
        image_signal = self._detect_image(data_dir, profile)
        text_signal = self._detect_text(profile)

        if image_signal.modality == "image" and text_signal.modality != "text":
            return image_signal
        if text_signal.modality == "text" and image_signal.modality != "image":
            return text_signal

        if image_signal.modality == "image" and text_signal.modality == "text":
            ambiguous = ModalityResult(
                modality="tabular",
                confidence="ambiguous",
                signals=image_signal.signals + text_signal.signals,
                image_dir=image_signal.image_dir,
                image_column=image_signal.image_column,
                text_column=text_signal.text_column,
            )
            return self._llm_tiebreak(
                ambiguous, competition_title, competition_description, profile, llm_client
            )

        if image_signal.confidence == "ambiguous":
            return self._llm_tiebreak(
                image_signal, competition_title, competition_description, profile, llm_client
            )
        if text_signal.confidence == "ambiguous":
            return self._llm_tiebreak(
                text_signal, competition_title, competition_description, profile, llm_client
            )

        return ModalityResult(modality="tabular", confidence="high", signals=["default_tabular"])

    def _detect_image(
        self, data_dir: Path, profile: DatasetProfile, scan: _Scan | None = None
    ) -> ModalityResult:
        scan = scan or self.scan(data_dir)
        dir_counts = scan.image_dirs

        if not dir_counts and scan.zarr_store is None:
            return ModalityResult(modality="tabular", confidence="high")

        best_dir = scan.best_image_dir
        image_count = scan.image_count
        csv_count = scan.csv_count
        # Multimodal geology-style layouts: many per-well CSVs plus PNG
        # previews. Prefer tabular so baselines use the structured logs — and
        # still say where the images are. Returning `image_dir=None` here is
        # what made the previews invisible to everything downstream, which is
        # not what "prefer tabular" should mean.
        if csv_count > 0 and csv_count >= max(image_count, 1):
            return ModalityResult(
                modality="tabular",
                confidence="ambiguous",
                signals=[
                    f"csv_files={csv_count}",
                    f"image_files={image_count}",
                    "prefer_tabular_over_auxiliary_images",
                ],
                image_dir=best_dir or None,
                image_count=image_count,
            )

        if scan.zarr_store is not None and not dir_counts:
            return ModalityResult(
                modality="image",
                confidence="high",
                signals=[f"zarr_store={scan.zarr_store}"],
                image_dir=scan.zarr_store,
            )

        image_dir_path = data_dir / best_dir if best_dir else data_dir
        image_column = self._match_filename_column(data_dir, profile, image_dir_path)
        if image_column is None:
            # Scientific / tracking datasets often have images without a CSV
            # filename column — still treat as image modality.
            return ModalityResult(
                modality="image",
                confidence="high",
                signals=[
                    f"images_in={best_dir or '.'}",
                    "no_filename_column",
                ],
                image_dir=best_dir or ".",
                image_count=image_count,
            )

        return ModalityResult(
            modality="image",
            confidence="high",
            signals=[f"image_dir={best_dir or '.'}", f"image_column={image_column}"],
            image_dir=best_dir or ".",
            image_column=image_column,
            image_count=image_count,
        )

    def _match_filename_column(
        self, data_dir: Path, profile: DatasetProfile, image_dir: Path
    ) -> str | None:
        if not profile.train_file:
            return None
        train_df = pd.read_csv(data_dir / profile.train_file, nrows=200)
        image_files = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not image_files:
            return None

        sample_names = {p.name for p in image_files}
        sample_stems = {p.stem for p in image_files}
        skip = {profile.target_column}
        candidates: list[str] = []

        for col in train_df.columns:
            if col in skip:
                continue
            values = train_df[col].astype(str).dropna()
            if values.empty:
                continue
            sample = values.head(50)
            hits = sum(
                1 for value in sample if value in sample_names or Path(value).stem in sample_stems
            )
            if hits >= max(3, len(sample) * 0.5):
                candidates.append(col)

        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            return candidates[0]
        return None

    def _detect_text(self, profile: DatasetProfile) -> ModalityResult:
        skip = {profile.target_column, profile.id_column}
        candidates: list[tuple[str, float]] = []

        for column in profile.columns:
            if column.name in skip or column.is_numeric:
                continue
            avg_len = column.stats.get("avg_length", 0.0)
            if avg_len >= MIN_TEXT_AVG_LENGTH and column.unique_count > 10:
                candidates.append((column.name, avg_len))
            elif column.name.lower() in TEXT_COLUMN_HINTS:
                candidates.append((column.name, avg_len or float(MIN_TEXT_AVG_LENGTH)))

        if len(candidates) == 1:
            name, _ = candidates[0]
            return ModalityResult(
                modality="text",
                confidence="high",
                signals=[f"text_column={name}"],
                text_column=name,
            )
        if len(candidates) > 1:
            return ModalityResult(
                modality="text",
                confidence="ambiguous",
                signals=[f"text_candidates={[c[0] for c in candidates]}"],
                text_column=candidates[0][0],
            )
        return ModalityResult(modality="tabular", confidence="high")

    def enrich_column_stats(self, df: pd.DataFrame, columns: list[ColumnProfile]) -> None:
        for column in columns:
            if column.is_numeric or column.name not in df.columns:
                continue
            series = df[column.name].astype(str).dropna()
            if len(series) == 0:
                continue
            column.stats["avg_length"] = float(series.str.len().mean())

    def _llm_tiebreak(
        self,
        result: ModalityResult,
        title: str,
        description: str,
        profile: DatasetProfile,
        llm_client: "LLMClient | None",
    ) -> ModalityResult:
        if llm_client is None:
            if result.image_column and not result.text_column:
                fallback = "image"
            elif result.text_column and not result.image_column:
                fallback = "text"
            else:
                fallback = "tabular"
            # `ambiguous`, not `high`. This branch is the one production always
            # takes — `_ensure_profile` passes no client — so the certainty this
            # used to stamp was the *absence* of a tie-breaker reported as the
            # presence of an answer.
            return ModalityResult(
                modality=fallback,
                confidence="ambiguous",
                signals=result.signals + ["llm_unavailable"],
                image_dir=result.image_dir,
                image_column=result.image_column,
                text_column=result.text_column,
                image_count=result.image_count,
            )

        system = "Pick exactly one modality: tabular, text, or image. Reply with one word only."
        user = (
            f"Title: {title}\nDescription: {description}\n"
            f"Rows: {profile.row_count}, Columns: {profile.column_count}\n"
            f"Signals: {', '.join(result.signals)}\n"
            "Reply with only: tabular, text, or image."
        )
        try:
            response = llm_client.complete(system, user).strip().lower()
        except Exception:
            logger.warning("LLM modality tie-breaker failed; using tabular.", exc_info=True)
            return ModalityResult(
                modality="tabular",
                confidence="ambiguous",
                signals=result.signals + ["llm_tiebreak_failed"],
            )

        token = re.split(r"[\s,.]+", response)[0] if response else ""
        if token not in {"tabular", "text", "image"}:
            # An unusable reply is a tie nothing resolved. Reporting `high` here
            # was the same laundering the no-client branch above used to do.
            return ModalityResult(
                modality="tabular",
                confidence="ambiguous",
                signals=result.signals + ["llm_tiebreak_unusable"],
            )

        return result.model_copy(
            update={"modality": token, "confidence": "high", "tiebroken": True}
        )
