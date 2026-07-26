"""``ForumAnalyzerAgent`` — mine structured signals from a discussion thread.

Discussion analysis is **not** a Phase 1 default analyzer, but ``research fetch``
uses this agent to enrich discussion artifacts. ``rule_engine`` applies light
keyword heuristics when the LLM is unavailable; LLM sharpens when present.
Emits :class:`ForumExtract`.
"""

from __future__ import annotations

import re

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
from labpilot.research_engine.intelligence.micro_agents.artifacts import ForumExtract

_MISTAKE = (
    r"\b(?:mistake|wrong|don't|do not|avoid|trap|pitfall|leak(?:age)?|"
    r"overfit|target.?leak)\b"
)
_DISCOVERY = (
    r"\b(?:found that|works well|improved|boost|gain|trick|tip|"
    r"surprisingly|key insight)\b"
)
_DATASET_BUG = (
    r"\b(?:dataset bug|label(?:ling)? error|corrupt(?:ed)?|"
    r"missing (?:files|labels)|bad annotation|data issue)\b"
)
_LB = (
    r"\b(?:shake.?up|leaderboard|public lb|private lb|probe|"
    r"overfit (?:to )?public)\b"
)
_OOD = (
    r"\b(?:o\.?o\.?d\.?|out[- ]of[- ]distribution|distribution shift|"
    r"domain shift|unseen)\b"
)


class ForumAnalyzerAgent(BaseMicroAgent):
    name = "ForumAnalyzerAgent"
    output_model = ForumExtract

    def system_prompt(self) -> str:
        return (
            "You extract actionable signals from a Kaggle discussion thread. "
            'Respond ONLY with JSON: {"mistakes": [str], "discoveries": [str], '
            '"dataset_bugs": [str], "lb_shakeups": [str], "ood_notes": [str]}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Discussion text:\n{context.text}"

    def _run_rule_engine(self, context: StructuredContext) -> ForumExtract:
        d = context.data
        pre = ForumExtract(
            mistakes=coerce_str_list(d.get("mistakes")),
            discoveries=coerce_str_list(d.get("discoveries")),
            dataset_bugs=coerce_str_list(d.get("dataset_bugs")),
            lb_shakeups=coerce_str_list(d.get("lb_shakeups")),
            ood_notes=coerce_str_list(d.get("ood_notes")),
        )
        if any(
            (
                pre.mistakes,
                pre.discoveries,
                pre.dataset_bugs,
                pre.lb_shakeups,
                pre.ood_notes,
            )
        ):
            return pre
        return _heuristic_extract(context.text or "")


def _heuristic_extract(text: str) -> ForumExtract:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    mistakes: list[str] = []
    discoveries: list[str] = []
    dataset_bugs: list[str] = []
    lb_shakeups: list[str] = []
    ood_notes: list[str] = []
    for line in lines:
        lower = line.lower()
        snippet = line[:180]
        if re.search(_DATASET_BUG, lower) and len(dataset_bugs) < 5:
            dataset_bugs.append(snippet)
        elif re.search(_OOD, lower) and len(ood_notes) < 5:
            ood_notes.append(snippet)
        elif re.search(_LB, lower) and len(lb_shakeups) < 5:
            lb_shakeups.append(snippet)
        elif re.search(_MISTAKE, lower) and len(mistakes) < 5:
            mistakes.append(snippet)
        elif re.search(_DISCOVERY, lower) and len(discoveries) < 5:
            discoveries.append(snippet)
    return ForumExtract(
        mistakes=mistakes,
        discoveries=discoveries,
        dataset_bugs=dataset_bugs,
        lb_shakeups=lb_shakeups,
        ood_notes=ood_notes,
    )
