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
from labpilot.research_engine.execution.outcome import (
    package_execution_submission,
    submission_result_path,
)
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
        execution_id = context.execution.id

        # Ensure root submission.csv exists for packaging.
        source = root / "submission.csv"
        if not source.is_file():
            pred = root / "predictions.csv"
            if pred.is_file():
                shutil.copy(pred, source)
            elif is_dry_run(context):
                # A dry run is checking the wiring, and the placeholder is the
                # wiring. A real run is not.
                source.write_text("id,prediction\n0,0\n", encoding="utf-8")
            else:
                # This wrote `id,prediction\n0,0` and then reported
                # `passed=packaged.is_file()` — a verdict about a file it had
                # just fabricated. The step promised "a submission was built"
                # and tested "did I write something", so a workspace with no
                # model, no predictions and no data passed it. M20 finding,
                # 2026-08-09.
                return evidence(
                    context,
                    capability=self.name,
                    passed=False,
                    summary="nothing to submit",
                    checks=["build_submission"],
                    error=(
                        "no submission.csv and no predictions.csv in the workspace. "
                        "Writing a placeholder row would package a file that "
                        "predicts nothing and report it as a submission."
                    ),
                )

        packaged = package_execution_submission(root, execution_id)

        notes: list[str] = []
        try:
            from labpilot.research_engine.execution.submission.formatter import (
                SubmissionValidator,
            )

            sample = root / "data" / "sample_submission.csv"
            if sample.is_file():
                notes.append("sample_submission present")
                _ = SubmissionValidator
        except Exception:
            pass

        upload_meta: dict = {"uploaded": False, "dry_run": True}
        if allow_upload(context) and not is_dry_run(context):
            upload_meta = self._try_upload(context, packaged)
        else:
            upload_meta["reason"] = "upload gated (default dry-run)"

        public_score = None
        if isinstance(upload_meta.get("public_score"), (int, float)):
            public_score = float(upload_meta["public_score"])
        result_blob = upload_meta.get("result") or {}
        if public_score is None and isinstance(result_blob, dict):
            if isinstance(result_blob.get("public_score"), (int, float)):
                public_score = float(result_blob["public_score"])

        meta_path = submission_result_path(root, execution_id)
        payload = {
            "execution_id": execution_id,
            "path": str(packaged),
            "competition": context.competition,
            "public_score": public_score,
            "status": upload_meta.get("status")
            or ("scored" if public_score is not None else "packaged"),
            "submissions_url": upload_meta.get("submissions_url")
            or (result_blob.get("submissions_url") if isinstance(result_blob, dict) else None),
            "upload": upload_meta,
        }
        meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        # Convenience latest copy.
        (artifacts / "submission_result.json").write_text(
            meta_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

        return evidence(
            context,
            capability=self.name,
            passed=packaged.is_file(),
            summary="submission packaged"
            + (" + uploaded" if upload_meta.get("uploaded") else " (no upload)"),
            checks=["build_submission"],
            paths=[str(packaged), str(meta_path)],
            metadata={
                "upload": upload_meta,
                "notes": notes,
                "execution_id": execution_id,
                "submission_path": str(packaged),
            },
            error=upload_meta.get("error"),
        )

    def _try_upload(self, context: TaskContext, path: Path) -> dict:
        try:
            from labpilot.accessor.kaggle.client import KaggleClient
            from labpilot.config import KaggleConfig

            kaggle_cfg = context.constraints.get("kaggle")
            if isinstance(kaggle_cfg, KaggleConfig):
                client = KaggleClient(kaggle_cfg)
            elif isinstance(kaggle_cfg, dict):
                client = KaggleClient(KaggleConfig.model_validate(kaggle_cfg))
            else:
                client = KaggleClient(KaggleConfig())
            result = client.upload_submission(
                context.competition,
                path,
                message=f"labpilot {context.execution.id}",
            )
            dumped = result.model_dump(mode="json")
            return {
                "uploaded": True,
                "dry_run": False,
                "public_score": result.public_score,
                "status": result.status,
                "message": result.message,
                "submissions_url": result.submissions_url,
                "result": dumped,
            }
        except Exception as exc:
            return {
                "uploaded": False,
                "dry_run": False,
                "error": str(exc),
                "status": "error",
            }
