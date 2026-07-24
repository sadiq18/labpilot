"""``CompetitionPageAnalyzerAgent`` — typed extract from overview + rules.

Plan 5b: LLM fills :class:`CompetitionPageExtract` when configured; otherwise
(and on soft-fail) the deterministic ``rule_engine`` path uses the same schema.
Never invents policy — inconclusive fields stay null/empty for the caller to
mark ``unavailable``.
"""

from __future__ import annotations

import re

from labpilot.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.intelligence.micro_agents.artifacts import CompetitionPageExtract

_DISALLOW_EXTERNAL = (
    "external data is not permitted",
    "no external data",
    "external data are not allowed",
    "external datasets are not allowed",
    "cannot use external data",
    "external data is prohibited",
)
_ALLOW_EXTERNAL = (
    "external data is allowed",
    "external data are allowed",
    "external data is permitted",
    "you may use external data",
    "external data allowed",
    "external datasets are allowed",
)
_PRETRAINED = ("pretrained", "pre-trained", "transfer learning", "imagenet", "weights")
_NO_INTERNET = (
    "no internet",
    "internet disabled",
    "without internet",
    "internet access is disabled",
    "offline",
)
_YES_INTERNET = (
    "internet enabled",
    "internet access is enabled",
    "with internet",
    "internet allowed",
)


class CompetitionPageAnalyzerAgent(BaseMicroAgent):
    name = "CompetitionPageAnalyzerAgent"
    output_model = CompetitionPageExtract

    def system_prompt(self) -> str:
        return (
            "You extract structured competition-contract fields from Kaggle "
            "overview and rules text. Respond ONLY with a JSON object matching "
            "this schema (use null for unknown booleans, empty string for "
            "unknown text — never invent policy):\n"
            "{"
            '"external_data_allowed": bool|null, '
            '"pretrained_weights_allowed": bool|null, '
            '"external_data_notes": str, '
            '"runtime_notes": str, '
            '"hardware_notes": str, '
            '"internet_allowed": bool|null, '
            '"inference_notes": str, '
            '"evaluation_formula": str, '
            '"evaluation_description": str, '
            '"submission_format": str, '
            '"submission_columns_notes": str, '
            '"sample_submission_notes": str, '
            '"overview_summary": str, '
            '"other_notes": str'
            "}. Extract evaluation formula/description, submission file format, "
            "external-data and inference/runtime limits when stated."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Competition pages:\n{context.text}"

    def _run_rule_engine(self, context: StructuredContext) -> CompetitionPageExtract:
        text = context.text or ""
        if not text.strip():
            return CompetitionPageExtract()

        # Prefer pre-parsed signals when callers inject them (tests / upstream).
        d = context.data
        if d.get("use_data_signals"):
            return CompetitionPageExtract(
                external_data_allowed=_as_optional_bool(d.get("external_data_allowed")),
                pretrained_weights_allowed=_as_optional_bool(
                    d.get("pretrained_weights_allowed")
                ),
                external_data_notes=str(d.get("external_data_notes", "")),
                runtime_notes=str(d.get("runtime_notes", "")),
                hardware_notes=str(d.get("hardware_notes", "")),
                internet_allowed=_as_optional_bool(d.get("internet_allowed")),
                inference_notes=str(d.get("inference_notes", "")),
                evaluation_formula=str(d.get("evaluation_formula", "")),
                evaluation_description=str(d.get("evaluation_description", "")),
                submission_format=str(d.get("submission_format", "")),
                submission_columns_notes=str(d.get("submission_columns_notes", "")),
                sample_submission_notes=str(d.get("sample_submission_notes", "")),
                overview_summary=str(d.get("overview_summary", "")),
                other_notes=str(d.get("other_notes", "")),
            )

        lower = text.lower()
        external_allowed = _match_bool(lower, _ALLOW_EXTERNAL, _DISALLOW_EXTERNAL)
        pretrained: bool | None = None
        if external_allowed is False:
            pretrained = False
        elif external_allowed is True and any(h in lower for h in _PRETRAINED):
            pretrained = True

        internet = _match_bool(lower, _YES_INTERNET, _NO_INTERNET)
        evaluation_section = _section_after_heading(
            text, ("evaluation", "metric", "scoring")
        )
        submission_section = _section_after_heading(
            text, ("submission", "submit", "file format")
        )
        runtime_section = _section_after_heading(
            text, ("runtime", "inference", "kernel", "resource", "hardware", "gpu")
        )
        overview_section = _section_after_heading(
            text, ("overview", "description", "summary")
        ) or _first_paragraph(text)

        formula = _pick_formula(evaluation_section) or _pick_formula(text)
        submission_format = _guess_submission_format(submission_section or text)

        return CompetitionPageExtract(
            external_data_allowed=external_allowed,
            pretrained_weights_allowed=pretrained,
            external_data_notes=_clip(
                _sentence_with(lower, text, "external data") if external_allowed is not None else ""
            ),
            runtime_notes=_clip(runtime_section),
            hardware_notes=_clip(
                _sentence_with(lower, text, "gpu") or _sentence_with(lower, text, "cpu")
            ),
            internet_allowed=internet,
            inference_notes=_clip(
                _sentence_with(lower, text, "internet") if internet is not None else ""
            ),
            evaluation_formula=_clip(formula),
            evaluation_description=_clip(evaluation_section),
            submission_format=submission_format,
            submission_columns_notes=_clip(
                _sentence_with(lower, text, "column")
                or _sentence_with(lower, text, "header")
            ),
            sample_submission_notes=_clip(_sentence_with(lower, text, "sample submission")),
            overview_summary=_clip(overview_section, limit=500),
            other_notes="",
        )


def _as_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _match_bool(lower: str, yes: tuple[str, ...], no: tuple[str, ...]) -> bool | None:
    if any(hit in lower for hit in no):
        return False
    if any(hit in lower for hit in yes):
        return True
    return None


def _section_after_heading(text: str, headings: tuple[str, ...]) -> str:
    lines = text.splitlines()
    start: int | None = None
    for idx, line in enumerate(lines):
        normalized = re.sub(r"^#+\s*", "", line).strip().lower()
        if any(normalized == h or normalized.startswith(f"{h} ") for h in headings):
            start = idx + 1
            break
    if start is None:
        return ""
    chunk: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("#") and chunk:
            break
        # Next all-caps short heading
        if (
            chunk
            and stripped
            and stripped.isupper()
            and len(stripped.split()) <= 6
            and not stripped.endswith(".")
        ):
            break
        chunk.append(line)
        if sum(len(c) for c in chunk) > 1200:
            break
    return "\n".join(chunk).strip()


def _first_paragraph(text: str) -> str:
    for block in re.split(r"\n\s*\n", text.strip()):
        cleaned = block.strip()
        if len(cleaned) >= 40:
            return cleaned
    return text.strip()[:500]


def _pick_formula(text: str) -> str:
    if not text:
        return ""
    # Prefer fenced / inline math-ish or "score =" lines.
    for pattern in (
        r"\$\$[\s\S]{3,400}?\$\$",
        r"\$[^$\n]{3,200}\$",
        r"(?i)(?:score|metric|f1|map|iou)\s*[=:]\s*[^\n]{3,200}",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return ""


def _guess_submission_format(text: str) -> str:
    lower = text.lower()
    if "submission.csv" in lower or re.search(r"\bcsv\b", lower):
        return "csv"
    if "kernel" in lower and "submit" in lower:
        return "kernel"
    if ".parquet" in lower:
        return "parquet"
    return ""


def _sentence_with(lower: str, original: str, needle: str) -> str:
    if needle not in lower:
        return ""
    # Map back roughly via splitting original on sentence boundaries.
    for sentence in re.split(r"(?<=[.!?])\s+", original):
        if needle in sentence.lower():
            return sentence.strip()
    return ""


def _clip(value: str, *, limit: int = 800) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
