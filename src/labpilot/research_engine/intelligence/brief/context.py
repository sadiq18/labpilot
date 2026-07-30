"""Render deterministic competition context for brief.md and LLM prompts."""

from labpilot.research_engine.intelligence.competition.models import CompetitionSpec

_RULES_EXCERPT_MAX_CHARS = 2000


def render_competition_context(competition: CompetitionSpec) -> str:
    """Build the `## Competition Context` markdown block from structured metadata."""
    lines = ["## Competition Context", ""]

    if competition.title:
        lines.append(f"**Title:** {competition.title}")
    lines.append(f"**Slug:** {competition.slug}")

    metric = competition.evaluation_metric
    if metric:
        lines.append(f"**Metric:** {metric.description or metric.name}")
        if metric.key:
            lines.append(f"**Metric key:** {metric.key}")
        lines.append(f"**Direction:** {metric.direction}")
    else:
        lines.append("**Metric:** unknown")

    if competition.deadline:
        lines.append(f"**Deadline:** {competition.deadline}")
    else:
        lines.append("**Deadline:** unknown")

    if competition.submissions_disabled:
        lines.append("**Submissions:** disabled")
    else:
        lines.append("**Submissions:** enabled")

    if competition.max_daily_submissions is not None:
        lines.append(f"**Daily submission limit:** {competition.max_daily_submissions}")

    if competition.is_kernels_submissions_only:
        lines.append("**Kernels-only submissions:** yes")

    lines.append(f"**Submission mode:** {competition.submission_mode}")
    if competition.submission_mode == "kernel":
        lines.append(f"**Kernel output file:** {competition.kernel_output_file}")
    if competition.submissions_url:
        lines.append(f"**Submissions page:** {competition.submissions_url}")

    if competition.tags:
        lines.append(f"**Tags:** {', '.join(competition.tags)}")

    if competition.raw_html:
        excerpt = competition.raw_html.strip()
        if len(excerpt) > _RULES_EXCERPT_MAX_CHARS:
            excerpt = excerpt[:_RULES_EXCERPT_MAX_CHARS].rsplit(" ", 1)[0].rstrip(".,;:")
            if excerpt and excerpt[-1] not in ".!?":
                excerpt = f"{excerpt}."
        lines.extend(["", "### Rules excerpt", "", excerpt])

    lines.append("")
    return "\n".join(lines)
