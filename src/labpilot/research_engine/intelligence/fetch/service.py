"""``KaggleFetchService`` — pull kernels/discussions into RawStore + research_artifacts."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from labpilot.config import KaggleConfig, Settings
from labpilot.kaggle.client import KaggleClient
from labpilot.kaggle.urls import kernel_notebook_url, parse_kernel_ref
from labpilot.research_engine.intelligence.fetch.enrich import (
    collect_kernel_source_text,
    enrich_discussion_artifact,
    enrich_kernel_artifact,
    thread_text_from_messages,
)
from labpilot.research_engine.intelligence.fetch.models import FetchResult
from labpilot.research_engine.intelligence.knowledge.sources import RawStore
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    ResearchArtifact,
    ResearchArtifactType,
)

logger = logging.getLogger("labpilot.research_engine.intelligence.fetch")

SourceName = Literal["discussions", "kernels"]
KernelSort = Literal["voteCount", "scoreDescending"]
DiscussionSort = Literal["top", "hot", "new", "recent", "active", "relevance"]

_DEFAULT_PAGE_SIZE = 20
_MAX_PAGES = 25


class KaggleFetchService:
    """Cron/CLI-ready fetcher. Soft-fails per source; never invents empty success."""

    def __init__(
        self,
        *,
        llm_client: object | None = None,
        kaggle: KaggleClient | None = None,
    ) -> None:
        self.llm_client = llm_client
        self._kaggle = kaggle

    def _client(self) -> KaggleClient:
        if self._kaggle is not None:
            return self._kaggle
        settings = Settings()
        config = KaggleConfig(
            api_token=settings.kaggle_api_token or "",
            username=settings.kaggle_username or "",
            key=settings.kaggle_key or "",
        )
        return KaggleClient(config)

    def fetch(
        self,
        competition: str,
        *,
        sources: set[SourceName],
        kernel_sort: KernelSort = "voteCount",
        discussion_sort: DiscussionSort = "top",
        limit: int = 20,
        refresh: bool = False,
        knowledge_dir: Path,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int = _MAX_PAGES,
    ) -> FetchResult:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if not sources:
            raise ValueError("sources must be non-empty")

        result = FetchResult(competition=competition, sources=sorted(sources))
        client = self._client()

        with KnowledgeStore(knowledge_dir, competition) as store:
            raw = RawStore(knowledge_dir, competition)
            if "kernels" in sources:
                self._fetch_kernels(
                    client,
                    store,
                    raw,
                    competition,
                    sort_by=kernel_sort,
                    limit=limit,
                    refresh=refresh,
                    page_size=page_size,
                    max_pages=max_pages,
                    result=result,
                )
            if "discussions" in sources:
                self._fetch_discussions(
                    client,
                    store,
                    raw,
                    competition,
                    sort_by=discussion_sort,
                    limit=limit,
                    refresh=refresh,
                    max_pages=max_pages,
                    result=result,
                )
        return result

    def _fetch_kernels(
        self,
        client: KaggleClient,
        store: KnowledgeStore,
        raw: RawStore,
        competition: str,
        *,
        sort_by: str,
        limit: int,
        refresh: bool,
        page_size: int,
        max_pages: int,
        result: FetchResult,
    ) -> None:
        written = 0
        try:
            for page in range(1, max_pages + 1):
                result.pages_scanned += 1
                try:
                    rows = client.list_kernels(
                        competition,
                        sort_by=sort_by,
                        page=page,
                        page_size=page_size,
                    )
                except Exception as exc:
                    result.notes.append(f"[kernels] list unavailable: {exc}")
                    return
                if not rows:
                    break
                for row in rows:
                    result.fetched += 1
                    ref = str(row.get("ref") or "").strip()
                    if not ref:
                        continue
                    artifact_id = f"kaggle-kernel:{ref}"
                    existing = store.get_artifact(artifact_id)
                    if existing is not None and not refresh:
                        result.skipped_existing += 1
                        continue
                    try:
                        artifact = self._ingest_kernel(
                            client,
                            store,
                            raw,
                            competition,
                            row=row,
                            sort_by=sort_by,
                            refresh=refresh,
                            result=result,
                        )
                    except Exception as exc:
                        logger.warning("Kernel ingest failed for %s: %s", ref, exc)
                        result.notes.append(f"[kernels] skip {ref}: {exc}")
                        continue
                    result.written += 1
                    result.artifact_ids.append(artifact.id)
                    written += 1
                    if written >= limit:
                        result.notes.append(
                            f"[kernels] reached unique limit={limit} "
                            f"(sort={sort_by})."
                        )
                        return
            result.notes.append(
                f"[kernels] exhausted pages (written={written}, limit={limit})."
            )
        except Exception as exc:
            result.notes.append(f"[kernels] failed: {exc}")

    def _ingest_kernel(
        self,
        client: KaggleClient,
        store: KnowledgeStore,
        raw: RawStore,
        competition: str,
        *,
        row: dict[str, Any],
        sort_by: str,
        refresh: bool,
        result: FetchResult,
    ) -> ResearchArtifact:
        ref = str(row["ref"])
        artifact_id = f"kaggle-kernel:{ref}"
        tmp = Path(tempfile.mkdtemp(prefix="labpilot-kernel-"))
        try:
            client.pull_kernel(ref, tmp, metadata=True)
            source_text, files = collect_kernel_source_text(tmp)
            payload = {
                "catalog": row,
                "files": files,
                "source_text_excerpt": source_text[:50_000],
            }
            # Also stash pulled blobs for re-extract.
            for path in tmp.rglob("*"):
                if path.is_file() and path.suffix.lower() in {
                    ".py",
                    ".ipynb",
                    ".r",
                    ".rmd",
                    ".json",
                }:
                    rel = path.relative_to(tmp).as_posix().replace("/", "__")
                    raw.write(
                        "kernels",
                        f"{_safe(ref)}__{rel}",
                        path.read_bytes(),
                        refresh=refresh,
                        ext=path.suffix.lstrip(".") or "bin",
                    )
            raw.write(
                "kernels",
                _safe(ref),
                json.dumps(payload, indent=2),
                refresh=refresh,
                ext="json",
            )
            try:
                owner, slug = parse_kernel_ref(ref)
                url = kernel_notebook_url(owner, slug)
            except Exception:
                url = f"https://www.kaggle.com/code/{ref}"
            artifact = ResearchArtifact(
                id=artifact_id,
                type=ResearchArtifactType.REPOSITORY,
                source="kaggle",
                title=str(row.get("title") or ref),
                summary=str(row.get("title") or ref),
                competition_slug=competition,
                confidence=0.55,
                metadata={
                    "kind": "kaggle_kernel",
                    "ref": ref,
                    "owner": row.get("author"),
                    "slug": row.get("slug"),
                    "votes": row.get("total_votes"),
                    "public_score": row.get("public_score"),
                    "sort": sort_by,
                    "url": url,
                    "language": row.get("language"),
                    "kernel_type": row.get("kernel_type"),
                    "files": files,
                },
            )
            artifact, source = enrich_kernel_artifact(
                artifact,
                competition=competition,
                source_text=source_text,
                llm_client=self.llm_client,
            )
            if source == "llm":
                result.llm_enriched += 1
            else:
                result.rule_engine_enriched += 1
            store.upsert_artifact(artifact)
            return artifact
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _fetch_discussions(
        self,
        client: KaggleClient,
        store: KnowledgeStore,
        raw: RawStore,
        competition: str,
        *,
        sort_by: str,
        limit: int,
        refresh: bool,
        max_pages: int,
        result: FetchResult,
    ) -> None:
        written = 0
        try:
            for page in range(1, max_pages + 1):
                result.pages_scanned += 1
                try:
                    rows = client.list_competition_topics(
                        competition, sort_by=sort_by, page=page
                    )
                except Exception as exc:
                    result.notes.append(f"[discussions] list unavailable: {exc}")
                    return
                if not rows:
                    break
                for row in rows:
                    result.fetched += 1
                    topic_id = int(row.get("id") or 0)
                    if not topic_id:
                        continue
                    artifact_id = f"kaggle-discussion:{competition}:{topic_id}"
                    existing = store.get_artifact(artifact_id)
                    if existing is not None and not refresh:
                        result.skipped_existing += 1
                        continue
                    try:
                        artifact = self._ingest_discussion(
                            client,
                            store,
                            raw,
                            competition,
                            row=row,
                            sort_by=sort_by,
                            refresh=refresh,
                            result=result,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Discussion ingest failed for %s: %s", topic_id, exc
                        )
                        result.notes.append(
                            f"[discussions] skip topic {topic_id}: {exc}"
                        )
                        continue
                    result.written += 1
                    result.artifact_ids.append(artifact.id)
                    written += 1
                    if written >= limit:
                        result.notes.append(
                            f"[discussions] reached unique limit={limit} "
                            f"(sort={sort_by})."
                        )
                        return
            result.notes.append(
                f"[discussions] exhausted pages (written={written}, limit={limit})."
            )
        except Exception as exc:
            result.notes.append(f"[discussions] failed: {exc}")

    def _ingest_discussion(
        self,
        client: KaggleClient,
        store: KnowledgeStore,
        raw: RawStore,
        competition: str,
        *,
        row: dict[str, Any],
        sort_by: str,
        refresh: bool,
        result: FetchResult,
    ) -> ResearchArtifact:
        topic_id = int(row["id"])
        artifact_id = f"kaggle-discussion:{competition}:{topic_id}"
        messages = client.fetch_topic_messages(competition, topic_id, page_size=-1)
        title = str(row.get("title") or f"topic-{topic_id}")
        text = thread_text_from_messages(title, messages)
        payload = {"catalog": row, "messages": messages, "sort": sort_by}
        raw.write(
            "discussions",
            f"topic_{topic_id}",
            json.dumps(payload, indent=2, default=str),
            refresh=refresh,
            ext="json",
        )
        artifact = ResearchArtifact(
            id=artifact_id,
            type=ResearchArtifactType.DISCUSSION,
            source="kaggle",
            title=title,
            summary=title,
            competition_slug=competition,
            confidence=0.5,
            metadata={
                "topic_id": topic_id,
                "votes": row.get("votes"),
                "url": row.get("topic_url"),
                "author": row.get("author_name"),
                "created_at": row.get("post_date"),
                "comment_count": row.get("comment_count"),
                "sort": sort_by,
            },
        )
        artifact, source = enrich_discussion_artifact(
            artifact,
            competition=competition,
            thread_text=text,
            llm_client=self.llm_client,
        )
        if source == "llm":
            result.llm_enriched += 1
        else:
            result.rule_engine_enriched += 1
        store.upsert_artifact(artifact)
        return artifact


def _safe(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"
