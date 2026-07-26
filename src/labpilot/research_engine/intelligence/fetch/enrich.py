"""Micro Agent enrichment for fetched Kaggle kernels and discussions."""

from __future__ import annotations

from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.intelligence.micro_agents.forum_analyzer import (
    ForumAnalyzerAgent,
)
from labpilot.research_engine.intelligence.micro_agents.repository_analyzer import (
    RepositoryAnalyzerAgent,
)
from labpilot.research_engine.intelligence.models import ResearchArtifact
from labpilot.research_engine.intelligence.repositories.models import RepoKnowledge


def enrich_kernel_artifact(
    artifact: ResearchArtifact,
    *,
    competition: str,
    source_text: str,
    llm_client: object | None = None,
) -> tuple[ResearchArtifact, str]:
    """Run ``RepositoryAnalyzerAgent``; return (artifact, extraction_source)."""
    agent = RepositoryAnalyzerAgent(llm_client=llm_client)
    meta = dict(artifact.metadata)
    try:
        card: RepoKnowledge = agent.run(
            StructuredContext(
                competition=competition,
                text=(source_text or "")[:120_000],
                data={
                    "repo_id": artifact.id,
                    "full_name": meta.get("ref") or artifact.title,
                    "interesting_files": meta.get("files") or [],
                    "has_readme": bool(source_text),
                },
            )
        )
    except Exception:
        meta["extraction_source"] = "rule_engine"
        return artifact.model_copy(update={"metadata": meta}), "rule_engine"

    source = "llm" if agent.last_used_llm else "rule_engine"
    techniques = list(dict.fromkeys([*artifact.techniques, *card.techniques]))
    models = list(dict.fromkeys([*artifact.models, *card.architecture]))
    claims = list(
        dict.fromkeys(
            [
                *artifact.claims,
                *[f"loss:{item}" for item in card.loss],
                *[f"aug:{item}" for item in card.augmentation],
                *[f"trick:{item}" for item in card.training_tricks],
            ]
        )
    )
    summary = artifact.summary
    if card.techniques and not summary:
        summary = f"Techniques: {', '.join(card.techniques[:8])}"
    elif card.techniques:
        summary = f"{summary} | techniques: {', '.join(card.techniques[:6])}"
    meta["extraction_source"] = source
    meta["repo_knowledge"] = card.model_dump(mode="json")
    updated = artifact.model_copy(
        update={
            "techniques": techniques,
            "models": models,
            "claims": claims,
            "summary": summary,
            "confidence": max(artifact.confidence, float(card.confidence or 0.5)),
            "metadata": meta,
        }
    )
    return updated, source


def enrich_discussion_artifact(
    artifact: ResearchArtifact,
    *,
    competition: str,
    thread_text: str,
    llm_client: object | None = None,
) -> tuple[ResearchArtifact, str]:
    """Run ``ForumAnalyzerAgent``; attach ``forum_extract`` metadata."""
    agent = ForumAnalyzerAgent(llm_client=llm_client)
    meta = dict(artifact.metadata)
    try:
        extract = agent.run(
            StructuredContext(
                competition=competition,
                text=(thread_text or "")[:80_000],
                data={},
            )
        )
    except Exception:
        meta["extraction_source"] = "rule_engine"
        return artifact.model_copy(update={"metadata": meta}), "rule_engine"

    source = "llm" if agent.last_used_llm else "rule_engine"
    payload = extract.model_dump(mode="json")
    meta["forum_extract"] = payload
    meta["extraction_source"] = source
    bits: list[str] = []
    for key, label in (
        ("discoveries", "discovery"),
        ("mistakes", "mistake"),
        ("dataset_bugs", "dataset bug"),
        ("lb_shakeups", "LB shakeup"),
        ("ood_notes", "OOD"),
    ):
        for item in payload.get(key) or []:
            bits.append(f"{label}: {item}")
    summary = artifact.summary
    if bits:
        joined = "; ".join(bits[:6])
        summary = f"{summary} | {joined}" if summary else joined
    updated = artifact.model_copy(
        update={
            "summary": summary,
            "metadata": meta,
            "confidence": max(artifact.confidence, 0.55 if bits else artifact.confidence),
        }
    )
    return updated, source


def collect_kernel_source_text(pull_dir: Any) -> tuple[str, list[str]]:
    """Read pulled kernel files into a single text blob for the Micro Agent."""
    from pathlib import Path

    root = Path(pull_dir)
    parts: list[str] = []
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "kernel-metadata.json":
            continue
        if path.suffix.lower() not in {".py", ".ipynb", ".r", ".md", ".txt", ".rmd"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        files.append(rel)
        parts.append(f"\nFILE {rel}:\n{text}")
    return "\n".join(parts), files


def thread_text_from_messages(
    title: str, messages: list[dict[str, Any]]
) -> str:
    chunks = [f"Title: {title}"]
    for msg in messages:
        if msg.get("is_deleted"):
            continue
        author = msg.get("author_name") or "unknown"
        depth = int(msg.get("depth") or 0)
        indent = "  " * depth
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        chunks.append(f"{indent}{author}: {content}")
    return "\n\n".join(chunks)
