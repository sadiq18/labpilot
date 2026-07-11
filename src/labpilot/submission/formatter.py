from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class SubmissionValidation(BaseModel):
    valid: bool
    row_count: int
    columns: list[str]
    errors: list[str] = []


class SubmissionFormatter:
    """Format model predictions into a Kaggle submission file."""

    def format(
        self,
        predictions: pd.DataFrame,
        id_column: str,
        target_column: str,
        output_path: Path,
    ) -> Path:
        submission = predictions[[id_column, target_column]].copy()
        submission.to_csv(output_path, index=False)
        return output_path


class SubmissionValidator:
    """Validate submission file format before upload."""

    def validate(self, submission_path: Path, expected_rows: int | None = None) -> SubmissionValidation:
        errors: list[str] = []
        df = pd.read_csv(submission_path)

        if len(df.columns) < 2:
            errors.append("Submission must have at least 2 columns (id + prediction).")

        if expected_rows is not None and len(df) != expected_rows:
            errors.append(f"Expected {expected_rows} rows, got {len(df)}.")

        if df.isnull().any().any():
            errors.append("Submission contains null values.")

        return SubmissionValidation(
            valid=len(errors) == 0,
            row_count=len(df),
            columns=list(df.columns),
            errors=errors,
        )
