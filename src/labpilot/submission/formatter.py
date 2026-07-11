from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field


class SubmissionValidation(BaseModel):
    valid: bool
    row_count: int
    columns: list[str]
    errors: list[str] = Field(default_factory=list)


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

    def validate(
        self,
        submission_path: Path,
        expected_rows: int | None = None,
        expected_columns: list[str] | None = None,
        target_column: str | None = None,
        require_integer_target: bool = False,
    ) -> SubmissionValidation:
        errors: list[str] = []
        df = pd.read_csv(submission_path)

        if len(df.columns) < 2:
            errors.append("Submission must have at least 2 columns (id + prediction).")

        if expected_rows is not None and len(df) != expected_rows:
            errors.append(f"Expected {expected_rows} rows, got {len(df)}.")

        if expected_columns is not None and list(df.columns) != expected_columns:
            errors.append(f"Expected columns {expected_columns}, got {list(df.columns)}.")

        if df.isnull().any().any():
            errors.append("Submission contains null values.")

        if require_integer_target and target_column:
            if target_column not in df:
                errors.append(f"Target column '{target_column}' is missing.")
            else:
                numeric = pd.to_numeric(df[target_column], errors="coerce")
                if numeric.isna().any() or not (numeric % 1 == 0).all():
                    errors.append(f"Target column '{target_column}' must contain integer labels.")

        return SubmissionValidation(
            valid=len(errors) == 0,
            row_count=len(df),
            columns=list(df.columns),
            errors=errors,
        )
