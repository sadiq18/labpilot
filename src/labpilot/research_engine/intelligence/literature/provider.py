"""LiteratureProvider facade + chained S2 → OpenAlex → arXiv → HF pipeline."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from labpilot.research_engine.intelligence.literature.cache import PaperCatalogStore
from labpilot.research_engine.intelligence.literature.clients import (
    ArxivClient,
    HuggingFacePapersClient,
    OpenAlexClient,
    SemanticScholarClient,
)
from labpilot.research_engine.intelligence.literature.models import Paper
from labpilot.research_engine.intelligence.models import AnalyzeContext

logger = logging.getLogger("labpilot.research_engine.intelligence.literature.provider")

DEFAULT_SEARCH_LIMIT = 40


class LiteratureProvider(ABC):
    """Facade over the literature chain. Extensible (ACL, CVF, PubMed, …)."""

    @abstractmethod
    def search(
        self,
        query: str | list[str],
        *,
        context: AnalyzeContext,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[Paper]:
        ...


class ChainedLiteratureProvider(LiteratureProvider):
    """S2 + arXiv (search, merge/dedupe) → OpenAlex (enrich) → arXiv PDF → HF.

    Discovery peers: Semantic Scholar and arXiv Search run together; results are
    merged and deduped by DOI / arXiv id / stable id. OpenAlex search remains a
    soft fallback when both discovery peers return nothing. Soft-fail per
    client; notes collected on ``self.notes``.
    """

    def __init__(
        self,
        *,
        semantic_scholar: SemanticScholarClient | None = None,
        openalex: OpenAlexClient | None = None,
        arxiv: ArxivClient | None = None,
        huggingface: HuggingFacePapersClient | None = None,
        catalog: PaperCatalogStore | None = None,
        download_pdfs: bool = True,
        search_fn: Callable[..., list[Paper]] | None = None,
    ) -> None:
        self.semantic_scholar = semantic_scholar or SemanticScholarClient()
        self.openalex = openalex or OpenAlexClient()
        self.arxiv = arxiv or ArxivClient()
        self.huggingface = huggingface or HuggingFacePapersClient()
        self.catalog = catalog
        self.download_pdfs = download_pdfs
        self.search_fn = search_fn
        self.notes: list[str] = []

    def search(
        self,
        query: str | list[str],
        *,
        context: AnalyzeContext,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[Paper]:
        self.notes = []
        queries = [query] if isinstance(query, str) else list(query)
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            self.notes.append("literature search: empty query.")
            return []

        catalog = self.catalog or PaperCatalogStore(
            context.knowledge_dir, context.competition
        )
        refresh = context.refresh

        candidates = self._discover(queries, limit=limit)
        if not candidates:
            # Still return cached catalog when search fails soft.
            cached = catalog.list_papers()
            if cached:
                self.notes.append(
                    "literature search: live search empty/failed — using cached catalog."
                )
                return _dedupe_rank(cached)[:limit]
            return []

        merged = catalog.merge_or_fetch(candidates, refresh=refresh)
        # Enrich only the ranked head so raising limit later fetches only missing ids.
        ranked = _dedupe_rank(merged)[:limit]
        enriched: list[Paper] = []
        skip_arxiv_hf = 0
        openalex_hits = 0
        for paper in ranked:
            cached = catalog.load_paper(paper.id)
            # Re-run enrich when missing or when a prior soft-fail left no OpenAlex payload.
            need_network = (
                refresh
                or cached is None
                or not (cached.payload or {}).get("openalex")
            )
            current = paper
            if need_network:
                before_arxiv = current.arxiv_id
                current = self._enrich_chain(current, catalog=catalog, refresh=refresh)
                if (current.payload or {}).get("openalex"):
                    openalex_hits += 1
                if not current.arxiv_id and not before_arxiv:
                    skip_arxiv_hf += 1
                catalog.save_paper(current, refresh=refresh)
            else:
                current = cached if cached is not None else paper
                # Prefer fresher search relevance; keep enriched fields from cache.
                if paper.relevance > current.relevance:
                    current = current.model_copy(update={"relevance": paper.relevance})
                # Ensure pdf_path is wired from cache when present.
                pdf = catalog.load_pdf_path(current.id)
                if pdf is not None and not current.pdf_path:
                    current = current.model_copy(update={"pdf_path": str(pdf)})
                if (current.payload or {}).get("openalex"):
                    openalex_hits += 1
                if not current.arxiv_id:
                    skip_arxiv_hf += 1
            enriched.append(current)

        if openalex_hits:
            self.notes.append(
                f"openalex enrich: present on {openalex_hits}/{len(enriched)} paper(s)."
            )
        if skip_arxiv_hf:
            self.notes.append(
                f"arxiv/huggingface attach: skipped for {skip_arxiv_hf} paper(s) "
                "without arxiv_id (OpenAlex enrich still ran when possible; "
                "HF Papers API is arXiv-keyed)."
            )

        return _dedupe_rank(enriched)[:limit]

    def _discover(self, queries: list[str], *, limit: int) -> list[Paper]:
        if self.search_fn is not None:
            try:
                return self.search_fn(queries, limit=limit)
            except Exception as exc:
                self.notes.append(f"literature search_fn: unavailable — {_short_err(exc)}")
                return []

        per_query = max(5, limit // max(len(queries), 1))
        s2_hits: list[Paper] = []
        arxiv_hits: list[Paper] = []
        s2_errors = 0
        arxiv_errors = 0

        for q in queries:
            try:
                s2_hits.extend(self.semantic_scholar.search(q, limit=per_query))
            except Exception as exc:
                s2_errors += 1
                self.notes.append(
                    f"semantic scholar: soft-fail for query {q!r} — {_short_err(exc)}"
                )

        for q in queries:
            try:
                arxiv_hits.extend(self.arxiv.search(q, limit=per_query))
            except Exception as exc:
                arxiv_errors += 1
                self.notes.append(
                    f"arxiv search: soft-fail for query {q!r} — {_short_err(exc)}"
                )

        found = _merge_discovery(s2_hits, arxiv_hits)
        if s2_hits or arxiv_hits:
            self.notes.append(
                f"discovery merge: semantic_scholar={len(s2_hits)}, "
                f"arxiv={len(arxiv_hits)}, unique={len(found)}"
                + (
                    f" (s2_errors={s2_errors})" if s2_errors else ""
                )
                + (
                    f" (arxiv_errors={arxiv_errors})" if arxiv_errors else ""
                )
                + "."
            )
            return found

        # Soft discovery fallback — OpenAlex search when both peers are empty.
        self.notes.append(
            "semantic scholar + arxiv search: no usable results — "
            "trying OpenAlex search fallback."
        )
        oa_hits: list[Paper] = []
        for q in queries:
            try:
                oa_hits.extend(self.openalex.search(q, limit=per_query))
            except Exception as exc:
                self.notes.append(f"openalex search: soft-fail — {_short_err(exc)}")
        if not oa_hits:
            self.notes.append(
                "literature discovery: no results from S2, arXiv, or OpenAlex."
            )
        return _dedupe_rank(oa_hits)

    def _enrich_chain(
        self,
        paper: Paper,
        *,
        catalog: PaperCatalogStore,
        refresh: bool,
    ) -> Paper:
        current = paper
        try:
            current = self.openalex.enrich(current)
        except Exception as exc:
            self.notes.append(f"openalex: soft-fail for {paper.id} — {_short_err(exc)}")

        try:
            current = self.arxiv.attach_pdf_meta(current)
            if self.download_pdfs and (refresh or not catalog.has_pdf(current.id)):
                if current.arxiv_id:
                    try:
                        blob = self.arxiv.download_pdf(current.arxiv_id)
                        path = catalog.save_pdf(current.id, blob, refresh=refresh)
                        current = current.model_copy(update={"pdf_path": str(path)})
                    except Exception as exc:
                        self.notes.append(
                            f"arxiv pdf: soft-fail for {paper.id} — {_short_err(exc)}"
                        )
                elif current.pdf_url:
                    # OpenAlex OA PDF when no arXiv id (still cache for future extract).
                    try:
                        from labpilot.research_engine.intelligence.literature.clients import (
                            _get_bytes,
                        )

                        blob = _get_bytes(current.pdf_url, timeout=90.0)
                        path = catalog.save_pdf(current.id, blob, refresh=refresh)
                        current = current.model_copy(update={"pdf_path": str(path)})
                    except Exception as exc:
                        self.notes.append(
                            f"oa pdf: soft-fail for {paper.id} — {_short_err(exc)}"
                        )
            elif catalog.has_pdf(current.id):
                path = catalog.load_pdf_path(current.id)
                if path is not None:
                    current = current.model_copy(update={"pdf_path": str(path)})
        except Exception as exc:
            self.notes.append(f"arxiv: soft-fail for {paper.id} — {_short_err(exc)}")

        try:
            if current.arxiv_id:
                current = self.huggingface.attach(current)
        except Exception as exc:
            self.notes.append(
                f"huggingface: soft-fail for {paper.id} — {_short_err(exc)}"
            )

        return current


def _short_err(exc: Exception) -> str:
    from labpilot.research_engine.intelligence.literature.clients import _short_http_error

    return _short_http_error(exc)


def _merge_discovery(*groups: list[Paper]) -> list[Paper]:
    """Merge peer search results and collapse duplicates across id schemes."""
    merged: list[Paper] = []
    for group in groups:
        merged.extend(group)
    return _dedupe_rank(merged)


def _paper_keys(paper: Paper) -> set[str]:
    """Identity keys used to collapse S2 / arXiv / OpenAlex duplicates."""
    keys = {paper.id}
    if paper.doi:
        doi = paper.doi.lower().removeprefix("https://doi.org/").removeprefix(
            "http://doi.org/"
        )
        keys.add(f"doi:{doi}")
    if paper.arxiv_id:
        from labpilot.research_engine.intelligence.literature.clients import (
            normalize_arxiv_id,
        )

        aid = normalize_arxiv_id(paper.arxiv_id) or paper.arxiv_id
        keys.add(f"arxiv:{aid}")
    return {k for k in keys if k}


def _prefer_paper(a: Paper, b: Paper) -> Paper:
    """Keep the stronger-ranked paper and fill missing fields from the other."""
    winner, donor = (a, b) if a.rank_score() >= b.rank_score() else (b, a)
    updated = winner.model_copy(deep=True)
    if not updated.abstract and donor.abstract:
        updated.abstract = donor.abstract
    if not updated.authors and donor.authors:
        updated.authors = list(donor.authors)
    if updated.year is None and donor.year is not None:
        updated.year = donor.year
    if not updated.venue and donor.venue:
        updated.venue = donor.venue
    if updated.citations is None and donor.citations is not None:
        updated.citations = donor.citations
    elif (
        updated.citations is not None
        and donor.citations is not None
        and donor.citations > updated.citations
    ):
        updated.citations = donor.citations
    if not updated.pdf_url and donor.pdf_url:
        updated.pdf_url = donor.pdf_url
    if not updated.arxiv_id and donor.arxiv_id:
        updated.arxiv_id = donor.arxiv_id
    if not updated.doi and donor.doi:
        updated.doi = donor.doi
    updated.relevance = max(updated.relevance, donor.relevance)
    for url_key, url_val in donor.urls.items():
        updated.urls.setdefault(url_key, url_val)
    for gh in donor.github_urls:
        if gh not in updated.github_urls:
            updated.github_urls.append(gh)
    for key, val in donor.payload.items():
        updated.payload.setdefault(key, val)
    # Prefer DOI-stable id when either side has a DOI.
    if updated.doi:
        from labpilot.research_engine.intelligence.literature.clients import (
            stable_paper_id,
        )

        updated.id = stable_paper_id(
            doi=updated.doi, arxiv_id=updated.arxiv_id, s2_id=None
        )
    elif updated.arxiv_id and not updated.id.startswith("doi:"):
        from labpilot.research_engine.intelligence.literature.clients import (
            stable_paper_id,
        )

        updated.id = stable_paper_id(
            doi=None, arxiv_id=updated.arxiv_id, s2_id=None
        )
    return updated


def _dedupe_rank(papers: list[Paper]) -> list[Paper]:
    """Dedupe by id / DOI / arXiv id, then sort by rank descending."""
    clusters: list[tuple[set[str], Paper]] = []
    for paper in papers:
        keys = _paper_keys(paper)
        matched_idx: int | None = None
        for idx, (existing_keys, existing) in enumerate(clusters):
            if keys & existing_keys:
                matched_idx = idx
                merged = _prefer_paper(existing, paper)
                clusters[idx] = (_paper_keys(merged) | existing_keys | keys, merged)
                break
        if matched_idx is None:
            clusters.append((keys, paper))
    return sorted(
        (paper for _, paper in clusters),
        key=lambda p: p.rank_score(),
        reverse=True,
    )


def literature_from_settings(
    *,
    knowledge_dir: Path | None = None,
    competition: str | None = None,
    download_pdfs: bool = True,
) -> ChainedLiteratureProvider:
    """Build a provider using optional Settings env keys."""
    from labpilot.config import Settings

    settings = Settings()
    catalog = None
    if knowledge_dir is not None and competition:
        catalog = PaperCatalogStore(knowledge_dir, competition)
    return ChainedLiteratureProvider(
        semantic_scholar=SemanticScholarClient(
            api_key=getattr(settings, "semantic_scholar_api_key", "") or ""
        ),
        openalex=OpenAlexClient(
            mailto=getattr(settings, "openalex_mailto", "") or "",
            api_key=getattr(settings, "openalex_api_key", "") or "",
        ),
        arxiv=ArxivClient(),
        huggingface=HuggingFacePapersClient(
            token=getattr(settings, "hf_token", "") or ""
        ),
        catalog=catalog,
        download_pdfs=download_pdfs,
    )
