"""Stage 5 — Context Compression + hierarchical memory budgets.

Character safety cap ``C = T × 3.5``. Truncation drops lowest-priority sections
first (secondary relational context → experiments/failures → technique evidence);
intent / L1 core is kept last.
"""

from __future__ import annotations

import json
from typing import Any

from labpilot.research_engine.intelligence.retrieval.models import (
    L1_CHAR_BUDGET,
    L2_CHAR_BUDGET,
    TOTAL_CHAR_BUDGET,
    RetrievalIntent,
    SymbolicBundle,
    TechniqueCard,
)


def compress_bundle(
    bundle: SymbolicBundle,
    *,
    intent: RetrievalIntent,
    competition: dict[str, Any] | None = None,
    constraints: list[str] | None = None,
) -> tuple[list[TechniqueCard], dict[str, Any], str, dict[str, int]]:
    """Build technique cards and a budgeted brief string.

    Returns ``(cards, research_context_fields, brief, budget_usage)``.
    """
    competition = competition or {}
    constraints = list(constraints or [])
    cards = _technique_cards(bundle)
    l1 = _render_l1(intent, competition)
    l2 = _render_l2(cards, bundle)
    l3_experiments = _render_experiments(bundle)
    l3_failures = _render_failures(bundle)
    secondary = _render_secondary(bundle)

    sections = {
        "l1": _fit(l1, L1_CHAR_BUDGET),
        "techniques": _fit(l2, L2_CHAR_BUDGET),
        "experiments": l3_experiments,
        "failures": l3_failures,
        "secondary": secondary,
    }
    brief, kept = _assemble_brief(sections, intent)
    budget = {
        "l1_chars": len(sections["l1"]),
        "l2_chars": len(sections["techniques"]),
        "l3_chars": len(sections["experiments"])
        + len(sections["failures"])
        + len(sections["secondary"]),
        "total_chars": len(brief),
        "total_budget": TOTAL_CHAR_BUDGET,
        "dropped": len([name for name, kept_flag in kept.items() if not kept_flag]),
    }

    fields = {
        "competition": {
            "slug": competition.get("slug") or intent.dataset or "",
            "task": intent.task,
            "metric": intent.metric,
            "domain": intent.domain,
            "pipeline": list(intent.current_pipeline),
            **{k: v for k, v in competition.items() if k not in {"slug"}},
        },
        "techniques": [card.model_dump(mode="json") for card in cards],
        "papers": [hit.model_dump(mode="json") for hit in bundle.papers],
        "experiments": [hit.model_dump(mode="json") for hit in bundle.experiments],
        "repositories": [hit.model_dump(mode="json") for hit in bundle.repositories],
        "failures": [hit.model_dump(mode="json") for hit in bundle.failures],
        "constraints": constraints,
        "question": intent.question or intent.goal or "",
    }
    return cards, fields, brief, budget


def _technique_cards(bundle: SymbolicBundle) -> list[TechniqueCard]:
    papers_by_tech: dict[str, list[str]] = {}
    exps_by_tech: dict[str, list[str]] = {}
    repos_by_tech: dict[str, list[str]] = {}
    labels_by_tech: dict[str, list[str]] = {}
    fail_by_tech: dict[str, list[str]] = {}

    for hit in bundle.papers:
        for tid in hit.knowledge_ids:
            papers_by_tech.setdefault(tid, []).append(hit.document_id or hit.label)
            labels_by_tech.setdefault(tid, []).append(hit.label)
    for hit in bundle.experiments:
        for tid in hit.knowledge_ids:
            exps_by_tech.setdefault(tid, []).append(hit.document_id or hit.label)
            labels_by_tech.setdefault(tid, []).append(hit.label)
    for hit in bundle.repositories:
        for tid in hit.knowledge_ids:
            repos_by_tech.setdefault(tid, []).append(hit.document_id or hit.label)
            labels_by_tech.setdefault(tid, []).append(hit.label)
    for hit in bundle.failures:
        for tid in hit.knowledge_ids:
            fail_by_tech.setdefault(tid, []).append(hit.summary or hit.label)

    cards: list[TechniqueCard] = []
    for row in bundle.techniques:
        tid = str(row["id"])
        meta = row.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        meta = meta if isinstance(meta, dict) else {}
        belief = str(meta.get("belief_status") or "")
        evidence_labels = list(dict.fromkeys(labels_by_tech.get(tid, [])))[:6]
        cards.append(
            TechniqueCard(
                id=tid,
                name=str(row.get("name") or tid),
                confidence=float(row.get("confidence") or row.get("_score") or 0.5),
                category=str(row.get("category") or ""),
                domain=str(row.get("domain") or ""),
                evidence_labels=evidence_labels,
                paper_ids=list(dict.fromkeys(papers_by_tech.get(tid, []))),
                experiment_ids=list(dict.fromkeys(exps_by_tech.get(tid, []))),
                repository_ids=list(dict.fromkeys(repos_by_tech.get(tid, []))),
                benefits=str(row.get("summary") or "")[:160],
                known_issues=str(row.get("known_issues") or "")[:160],
                belief_status=belief,
                failure_notes=list(dict.fromkeys(fail_by_tech.get(tid, [])))[:3],
            )
        )
    return cards


def _render_l1(intent: RetrievalIntent, competition: dict[str, Any]) -> str:
    slug = competition.get("slug") or intent.dataset or "unknown"
    pipeline = " · ".join(intent.current_pipeline) or "(unknown)"
    lines = [
        "Current Competition",
        f"    {slug}",
        "",
        "Current Pipeline",
        f"    {pipeline}",
    ]
    if intent.metric or intent.goal:
        lines.extend(["", "Current Goal", f"    {intent.goal or intent.metric}"])
    return "\n".join(lines)


def _render_l2(cards: list[TechniqueCard], bundle: SymbolicBundle) -> str:
    if not cards:
        return "Relevant Knowledge\n    (none)"
    blocks = ["Relevant Knowledge"]
    for card in cards:
        for line in card.render().splitlines():
            blocks.append(f"    {line}")
        blocks.append("")
    return "\n".join(blocks).rstrip()


def _render_experiments(bundle: SymbolicBundle) -> str:
    if not bundle.experiments:
        return ""
    lines = ["Local Experiments"]
    for hit in bundle.experiments:
        lines.append(f"    {hit.label}: {hit.summary or hit.why}")
    return "\n".join(lines)


def _render_failures(bundle: SymbolicBundle) -> str:
    if not bundle.failures:
        return ""
    lines = ["Relevant Failures"]
    for hit in bundle.failures:
        lines.append(f"    {hit.label}: {hit.summary or hit.why}")
    return "\n".join(lines)


def _render_secondary(bundle: SymbolicBundle) -> str:
    parts: list[str] = []
    if bundle.repositories:
        lines = ["Repositories"]
        for hit in bundle.repositories:
            lines.append(f"    {hit.label}: {hit.summary or hit.why}")
        parts.append("\n".join(lines))
    if bundle.papers:
        lines = ["Papers (labels)"]
        for hit in bundle.papers:
            lines.append(f"    {hit.label}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _fit(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 1)].rstrip() + "…"


def _assemble_brief(
    sections: dict[str, str], intent: RetrievalIntent
) -> tuple[str, dict[str, bool]]:
    """Priority keep order: l1 → techniques → experiments/failures → secondary."""
    question = intent.question or intent.goal or ""
    header = f"Question\n    {question}".strip() if question else ""

    # Drop lowest priority first until under TOTAL_CHAR_BUDGET.
    drop_order = ["secondary", "failures", "experiments"]
    active = dict(sections)
    kept = {name: bool(text.strip()) for name, text in active.items()}

    def render() -> str:
        parts = [header, active.get("l1", ""), active.get("techniques", "")]
        for name in ("experiments", "failures", "secondary"):
            if kept.get(name) and active.get(name):
                parts.append(active[name])
        return "\n\n".join(part for part in parts if part and part.strip())

    brief = render()
    for name in drop_order:
        if len(brief) <= TOTAL_CHAR_BUDGET:
            break
        if kept.get(name):
            kept[name] = False
            brief = render()

    # If still over, trim techniques then l1 (last resorts).
    if len(brief) > TOTAL_CHAR_BUDGET and active.get("techniques"):
        overflow = len(brief) - TOTAL_CHAR_BUDGET
        active["techniques"] = _fit(
            active["techniques"], max(80, len(active["techniques"]) - overflow)
        )
        brief = render()
    if len(brief) > TOTAL_CHAR_BUDGET:
        brief = _fit(brief, TOTAL_CHAR_BUDGET)
    return brief, kept
