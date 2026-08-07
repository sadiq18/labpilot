"""``PaperAnalyzerAgent`` — structured extraction from a single paper.

Flagship "Yes" pattern (design §2.4). The LLM is an information extractor, not
a chatbot: never "summarize this paper", always structured JSON validated
against :class:`PaperKnowledge`.
"""

from __future__ import annotations

import re

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
from labpilot.research_engine.intelligence.feature_recipes import (
    coerce_feature_recipes,
    heuristic_feature_recipes,
    merge_feature_recipes,
    recipe_technique_names,
)
from labpilot.research_engine.intelligence.literature.models import PaperKnowledge

_METHODISH = (
    "propose",
    "introduce",
    "we use",
    "we apply",
    "architecture",
    "augmentation",
    "loss",
    "training",
    "fine-tun",
    "mask",
    "attention",
    "transformer",
    "cnn",
    "resnet",
)
_LIMITISH = (
    "limitation",
    "fail",
    "cannot",
    "unable",
    "only work",
    "assumption",
    "expensive",
    "computationally",
    "does not",
    "struggle",
)


class PaperAnalyzerAgent(BaseMicroAgent):
    name = "PaperAnalyzerAgent"
    output_model = PaperKnowledge
    llm_max_attempts = 3
    llm_retry_delay_seconds = 20.0

    def system_prompt(self) -> str:
        return (
            "You extract structured research knowledge from an ML paper. "
            "Respond ONLY with a JSON object matching this schema "
            "(never write an essay summary):\n"
            "{"
            '"paper_id": str, "title": str, '
            '"contributions": [str], "methods": [str], "limitations": [str], '
            '"ideas_worth_testing": [str], "techniques": [str], '
            '"feature_recipes": [{"name": str, "description": str, '
            '"inputs": [str], "outputs": [str], "transform": str}], '
            '"datasets_used": [str], "benchmarks": [str], "code_urls": [str], '
            '"confidence": float, '
            '"grounded_in": "abstract"|"pdf_excerpt"|"metadata"'
            "}. "
            "Extract what is claimed as new, how it works, what breaks, and "
            "ideas worth testing on the competition named in the context. "
            "When the paper creates new features, fill feature_recipes with "
            "concrete name/inputs/outputs/transform — including arithmetic or "
            "derived columns (e.g. new=f1+f2, new=f1/f2, interactions, ratios), "
            "not only encoders/scalers. You decide which creations are worth "
            "recording; leave feature_recipes empty if none are grounded. "
            "Do not summarize sections."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        competition = context.competition or "(unknown competition)"
        return (
            f"Competition: {competition}\n"
            f"Paper text (abstract/metadata):\n{context.text}"
        )


def _heuristic_extract(
    text: str,
    *,
    techniques: list[str],
    ideas: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Thin abstract heuristics when no pre-parsed signals exist."""
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) >= 40
    ]
    contributions: list[str] = []
    methods: list[str] = []
    limitations: list[str] = []
    for sentence in sentences[:12]:
        lower = sentence.lower()
        if any(h in lower for h in _LIMITISH) and len(limitations) < 3:
            limitations.append(_clip(sentence))
        elif any(h in lower for h in ("we propose", "we introduce", "our contribution", "novel")):
            if len(contributions) < 3:
                contributions.append(_clip(sentence))
        elif any(h in lower for h in _METHODISH) and len(methods) < 4:
            methods.append(_clip(sentence))
    if not contributions and sentences:
        contributions = [_clip(sentences[0])]
    if not ideas and methods:
        ideas = [f"Try adapting: {_clip(methods[0], 120)}"]
    # Pull TitleCase / CamelCase-ish tokens as technique tags.
    tags = techniques or _guess_technique_tags(text)
    return contributions, methods, limitations, ideas, tags


def _guess_technique_tags(text: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+)\b", text):
        token = match.group(1)
        if token.lower() in {"http", "https", "arxiv"}:
            continue
        if token not in found:
            found.append(token)
        if len(found) >= 5:
            break
    return found


def _clip(value: str, limit: int = 240) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
