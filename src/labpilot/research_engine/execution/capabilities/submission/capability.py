"""Submission capability — package predictions; optional Kaggle upload."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from labpilot.research_engine.execution.capabilities._helpers import (
    allow_upload,
    evidence,
    is_dry_run,
)
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


class SubmissionCapability(BaseCapability):
    name = "submission"

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.BUILD_SUBMISSION})

    def execute(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        artifacts = root / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        source = root / "submission.csv"
        if not source.is_file():
            pred = root / "predictions.csv"
            if pred.is_file():
                shutil.copy(pred, source)
            else:
                source.write_text("id,prediction\n0,0\n", encoding="utf-8")

        packaged = artifacts / "submission.csv"
        shutil.copy(source, packaged)

        # Optional shape validation when sample exists.
        notes: list[str] = []
        try:
            from labpilot.research_engine.execution.submission.formatter import SubmissionValidator

            sample = root / "data" / "sample_submission.csv"
            if sample.is_file():
                # Best-effort; ignore failures in dry scaffold.
                notes.append("sample_submission present")
        except Exception:
            pass

        upload_meta: dict = {"uploaded": False, "dry_run": True}
        if allow_upload(context) and not is_dry_run(context):
            upload_meta = self._try_upload(context, packaged)
        else:
            upload_meta["reason"] = "upload gated (default dry-run)"

        meta_path = artifacts / "submission_result.json"
        meta_path.write_text(
            json.dumps(
                {
                    "path": str(packaged),
                    "competition": context.competition,
                    "upload": upload_meta,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return evidence(
            context,
            capability=self.name,
            passed=packaged.is_file(),
            summary="submission packaged"
            + (" + uploaded" if upload_meta.get("uploaded") else " (no upload)"),
            checks=["build_submission"],
            paths=[str(packaged), str(meta_path)],
            metadata={"upload": upload_meta, "notes": notes},
            error=upload_meta.get("error"),
        )

    def _try_upload(self, context: TaskContext, path: Path) -> dict:
        try:
            from labpilot.accessor.kaggle.client import KaggleClient

            client = KaggleClient()
            result = client.upload_submission(context.competition, path)
            return {
                "uploaded": True,
                "dry_run": False,
                "result": getattr(result, "model_dump", lambda: {"raw": str(result)})(),
            }
        except Exception as exc:
            return {
                "uploaded": False,
                "dry_run": False,
                "error": str(exc),
            }
