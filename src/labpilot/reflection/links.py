"""Deterministic submission link footer for reflection.md."""

from labpilot.accessor.kaggle.client import SubmissionResult


def render_submission_links(submission: SubmissionResult) -> str:
    lines = ["## Submission links", ""]
    if submission.submissions_url:
        lines.append(f"- [Competition submissions]({submission.submissions_url})")
    if submission.kernel_url:
        lines.append(f"- [Kernel notebook]({submission.kernel_url}) *(kernel-only)*")
    elif submission.kernel_slug:
        lines.append(f"- Kernel: `{submission.kernel_slug}`")
    if submission.kernel_version is not None:
        lines.append(f"- Kernel version: {submission.kernel_version}")
    if submission.kernel_run_status:
        lines.append(f"- Kernel run status: {submission.kernel_run_status}")
    lines.append("")
    return "\n".join(lines)
