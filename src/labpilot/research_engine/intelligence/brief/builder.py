"""Assemble a Research Brief from AnalysisReport + KnowledgeStore."""

from __future__ import annotations

from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext, run_or_none
from labpilot.research_engine.intelligence.brief.models import ResearchBrief
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.micro_agents.research_brief.agent import (
    ResearchBriefAgent,
)
from labpilot.research_engine.intelligence.models import (
    AnalysisReport,
    ResearchArtifactType,
)


def build_research_brief(
    report: AnalysisReport,
    store: KnowledgeStore,
    *,
    llm_client: object | None = None,
) -> ResearchBrief:
    """Deterministic section assembly + optional Micro Agent narrative."""
    competition = report.competition if isinstance(report.competition, dict) else {}
    dataset = _dataset_overview(report)
    rules = _rules_and_metric(competition)
    papers = _paper_lines(report)
    similar = _similar_competition_lines(report)
    repos = _repository_lines(report)
    techniques = _winning_technique_lines(report, competition)
    beliefs = _belief_lines(store)
    hypotheses = _hypothesis_lines(report)
    risks = _risk_lines(report, competition)
    suggested = _suggested_lines(report)

    structured = {
        "title": str(competition.get("title") or competition.get("slug") or ""),
        "slug": str(competition.get("slug") or store.competition),
        "problem_type": competition.get("problem_type"),
        "metric": competition.get("metric"),
        "dataset_overview": dataset,
        "rules_and_metric": rules,
        "papers": papers[:8],
        "similar_competitions": similar[:6],
        "repositories": repos[:8],
        "winning_techniques": techniques[:10],
        "beliefs": beliefs[:10],
        "top_hypotheses": hypotheses[:5],
        "known_risks": risks[:10],
        "suggested_experiments": suggested[:5],
    }
    agent = ResearchBriefAgent(llm_client=llm_client)
    narrative = run_or_none(
        agent,
        StructuredContext(
            competition=store.competition,
            question="Write a researcher briefing before experimentation",
            text=_context_text(structured),
            data=structured,
        ),
    )
    generated_by = agent.last_generated_by if narrative is not None else "template_fallback"
    problem = str(getattr(narrative, "problem_summary", "") or "").strip()
    if not problem:
        problem = _fallback_problem_summary(competition, dataset)
    narrative_risks = [
        str(item).strip()
        for item in (getattr(narrative, "key_risks", None) or [])
        if str(item).strip()
    ]
    known_risks = list(dict.fromkeys([*narrative_risks, *risks]))
    focus = str(getattr(narrative, "recommended_focus", "") or "").strip()
    suggested_out = list(suggested)
    if focus and focus not in suggested_out:
        suggested_out = [focus, *suggested_out][:8]

    notes: list[str] = []
    if not dataset:
        notes.append("dataset overview unavailable (no profile artifact yet)")
    if not papers:
        notes.append("no related papers in report")

    return ResearchBrief(
        problem_summary=problem,
        dataset_overview=dataset,
        rules_and_metric=rules,
        related_papers=papers,
        similar_competitions=similar,
        repositories=repos,
        winning_techniques=techniques,
        beliefs=beliefs,
        top_hypotheses=hypotheses,
        known_risks=known_risks,
        suggested_experiments=suggested_out,
        generated_by=generated_by,
        notes=notes,
    )


def _context_text(structured: dict[str, Any]) -> str:
    lines = [
        f"Competition: {structured.get('title')} ({structured.get('slug')})",
        f"Problem type: {structured.get('problem_type')}",
        f"Dataset: {structured.get('dataset_overview')}",
        f"Rules/metric: {structured.get('rules_and_metric')}",
        "Papers: " + "; ".join(structured.get("papers") or []),
        "Repos: " + "; ".join(structured.get("repositories") or []),
        "Techniques: " + "; ".join(structured.get("winning_techniques") or []),
        "Beliefs: " + "; ".join(structured.get("beliefs") or []),
        "Hypotheses: " + "; ".join(structured.get("top_hypotheses") or []),
        "Risks: " + "; ".join(structured.get("known_risks") or []),
    ]
    return "\n".join(lines)[:6000]


def _fallback_problem_summary(competition: dict[str, Any], dataset: str) -> str:
    title = str(competition.get("title") or competition.get("slug") or "Competition")
    problem = str(competition.get("problem_type") or "unknown problem type")
    metric = competition.get("metric") or {}
    metric_name = ""
    if isinstance(metric, dict):
        metric_name = str(metric.get("name") or metric.get("label") or "")
    bits = [f"{title} — {problem}"]
    if metric_name:
        bits.append(f"metric={metric_name}")
    if dataset:
        bits.append(dataset)
    return ". ".join(bits) + "."


def _dataset_overview(report: AnalysisReport) -> str:
    for artifact in report.artifacts:
        if artifact.type is not ResearchArtifactType.DATASET:
            continue
        meta = artifact.metadata or {}
        parts = [
            artifact.summary
            or f"{meta.get('modality', 'dataset')} — "
            f"{meta.get('row_count', '?')} rows / {meta.get('column_count', '?')} cols"
        ]
        target = meta.get("target_column")
        if target:
            parts.append(f"target={target}")
        null_heavy = meta.get("null_heavy_columns") or []
        if null_heavy:
            parts.append(f"null-heavy: {', '.join(str(c) for c in null_heavy[:5])}")
        warnings = meta.get("warnings") or []
        if warnings:
            parts.append(f"warnings: {'; '.join(str(w) for w in warnings[:3])}")
        return "; ".join(parts)
    return ""


def _rules_and_metric(competition: dict[str, Any]) -> str:
    parts: list[str] = []
    metric = competition.get("metric")
    if isinstance(metric, dict) and metric:
        name = metric.get("name") or metric.get("label") or ""
        direction = metric.get("direction") or ""
        parts.append(f"metric={name}{f' ({direction})' if direction else ''}".strip())
    elif metric:
        parts.append(f"metric={metric}")
    rules = str(competition.get("rules_excerpt") or "").strip()
    if rules:
        parts.append(rules[:400])
    external = competition.get("external_data") or {}
    if isinstance(external, dict) and external:
        allowed = external.get("allowed")
        if allowed is not None:
            parts.append(f"external_data_allowed={allowed}")
    inference = competition.get("inference_limits") or {}
    if isinstance(inference, dict):
        runtime = str(inference.get("runtime_notes") or "").strip()
        if runtime:
            parts.append(f"runtime: {runtime[:200]}")
    submission = competition.get("submission") or {}
    if isinstance(submission, dict):
        fmt = str(submission.get("format") or submission.get("notes") or "").strip()
        if fmt:
            parts.append(f"submission: {fmt[:200]}")
    return " | ".join(parts)


def _paper_lines(report: AnalysisReport) -> list[str]:
    lines: list[str] = []
    for paper in report.papers:
        title = str(paper.get("title") or paper.get("id") or "").strip()
        if not title:
            continue
        techniques = paper.get("techniques") or []
        if techniques:
            lines.append(f"{title} [{', '.join(str(t) for t in techniques[:4])}]")
        else:
            lines.append(title)
    return lines


def _similar_competition_lines(report: AnalysisReport) -> list[str]:
    lines: list[str] = []
    for item in report.related_competitions:
        slug = str(item.get("slug") or item.get("title") or "").strip()
        if not slug:
            continue
        relation = str(item.get("relation") or "").strip()
        lines.append(f"{slug}" + (f" ({relation})" if relation else ""))
    return lines


def _repository_lines(report: AnalysisReport) -> list[str]:
    lines: list[str] = []
    for repo in report.repositories:
        title = str(repo.get("title") or repo.get("id") or "").strip()
        if not title:
            continue
        techniques = repo.get("techniques") or []
        if techniques:
            lines.append(f"{title} [{', '.join(str(t) for t in techniques[:4])}]")
        else:
            lines.append(title)
    for transfer in report.transfer_opportunities:
        summary = str(transfer.get("summary") or transfer.get("hypothesis_hint") or "").strip()
        if summary:
            lines.append(f"transfer: {summary}")
    return list(dict.fromkeys(lines))


def _winning_technique_lines(
    report: AnalysisReport, competition: dict[str, Any]
) -> list[str]:
    lines: list[str] = []
    winning = competition.get("winning_solutions") or {}
    if isinstance(winning, dict):
        status = str(winning.get("status") or "")
        if status and status != "ok":
            lines.append(f"winning_solutions: {status}")
        for item in winning.get("items") or []:
            if isinstance(item, dict):
                label = str(item.get("title") or item.get("summary") or "").strip()
                if label:
                    lines.append(label)
    buckets = report.techniques
    for label, values in (
        ("external", buckets.external_recommendations),
        ("validated", buckets.locally_validated),
        ("unverified", buckets.unverified),
    ):
        for technique in values:
            lines.append(f"{technique} ({label})")
    return list(dict.fromkeys(lines))


def _belief_lines(store: KnowledgeStore) -> list[str]:
    lines: list[str] = []
    for belief in store.list_beliefs():
        technique = str(belief.get("technique") or "").strip()
        if not technique:
            continue
        status = str(belief.get("status") or "suggested")
        effect = str(belief.get("effect") or "")
        conf = belief.get("confidence")
        bit = f"{technique} [{status}"
        if effect and effect != "unknown":
            bit += f", {effect}"
        if conf is not None:
            try:
                bit += f", conf={float(conf):.2f}"
            except (TypeError, ValueError):
                pass
        bit += "]"
        lines.append(bit)
    return lines


def _hypothesis_lines(report: AnalysisReport) -> list[str]:
    lines: list[str] = []
    for card in report.hypothesis_recommendations[:5]:
        title = str(card.get("title") or "").strip()
        hyp_id = str(card.get("hypothesis_id") or "").strip()
        if not title:
            continue
        lines.append(f"{hyp_id}: {title}" if hyp_id else title)
    return lines


def _suggested_lines(report: AnalysisReport) -> list[str]:
    lines: list[str] = []
    for card in report.suggested_experiments[:8]:
        title = str(card.get("title") or "").strip()
        if title:
            lines.append(title)
    return lines


def _risk_lines(report: AnalysisReport, competition: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    for artifact in report.artifacts:
        if artifact.type is ResearchArtifactType.EXPERIMENT:
            meta = artifact.metadata or {}
            for key in ("failures", "failed_techniques", "known_failures"):
                for item in meta.get(key) or []:
                    risks.append(str(item))
            summary = (artifact.summary or "").lower()
            if any(token in summary for token in ("hurt", "fail", "regress", "worse")):
                risks.append(artifact.summary)
        forum = (artifact.metadata or {}).get("forum_extract") or {}
        if isinstance(forum, dict):
            for key in ("mistakes", "dataset_bugs", "lb_shakeups", "ood_notes"):
                for item in forum.get(key) or []:
                    risks.append(f"{key}: {item}")
    for unit in report.knowledge_units:
        issues = unit.get("known_issues") or unit.get("metadata", {}).get("known_issues")
        if issues:
            name = unit.get("name") or unit.get("id") or "technique"
            risks.append(f"{name}: {issues}")
    external = competition.get("external_data") or {}
    if isinstance(external, dict) and external.get("allowed") is False:
        risks.append("External data / pretrained weights restricted by competition rules")
    inference = competition.get("inference_limits") or {}
    if isinstance(inference, dict):
        for key in ("runtime_notes", "hardware_notes"):
            note = str(inference.get(key) or "").strip()
            if note:
                risks.append(note)
    return list(dict.fromkeys(str(r).strip() for r in risks if str(r).strip()))
