import json
from pathlib import Path

from labpilot.accessor.common.derived import derived_note
from labpilot.accessor.profiler.tabular import DatasetProfile


def write_profile(run_dir: Path, profile: DatasetProfile) -> tuple[Path, Path]:
    json_path = run_dir / "profile.json"
    md_path = run_dir / "profile.md"

    json_path.write_text(profile.model_dump_json(indent=2))
    md_path.write_text(
        derived_note(
            source_of_record="profile.json",
            warning="Regenerated only when profiling reruns; the data may have changed.",
        )
        + "\n\n"
        + render_markdown(profile)
    )

    return json_path, md_path


def render_markdown(profile: DatasetProfile) -> str:
    """Markdown view over a DatasetProfile.

    Stamped: `profile.json` is written from the same call and is what every
    consumer reads, while this is regenerated only when profiling reruns. M20
    criterion 4.
    """
    lines = [
        f"# Dataset Profile: {profile.competition}",
        "",
        f"- **Rows:** {profile.row_count:,}",
        f"- **Columns:** {profile.column_count}",
        f"- **Files:** {', '.join(profile.files) if profile.files else 'none'}",
        "",
        "## Columns",
        "",
        "| Column | Dtype | Nulls | Unique |",
        "|--------|-------|-------|--------|",
    ]

    for col in profile.columns:
        lines.append(f"| {col.name} | {col.dtype} | {col.null_pct}% | {col.unique_count} |")

    if profile.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in profile.warnings)

    return "\n".join(lines) + "\n"


def load_profile(run_dir: Path) -> DatasetProfile:
    return DatasetProfile.model_validate(json.loads((run_dir / "profile.json").read_text()))
