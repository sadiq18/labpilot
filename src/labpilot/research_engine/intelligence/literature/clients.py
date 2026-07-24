"""HTTP clients for the literature provider chain (Plan 6)."""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import quote

import httpx

from labpilot.research_engine.intelligence.literature.models import Paper

logger = logging.getLogger("labpilot.research_engine.intelligence.literature.clients")

_DEFAULT_TIMEOUT = 60.0
# Semantic Scholar (and shared HTTP) — wait at least 6s between calls; retry with
# exponential backoff floored at the same minimum.
_HTTP_MIN_WAIT_S = 6.0
_S2_MIN_INTERVAL_S = _HTTP_MIN_WAIT_S
_HTTP_MAX_ATTEMPTS = 7
_HTTP_BASE_BACKOFF_S = 2
_HTTP_BACKOFF_CAP_S = 60.0


def _retry_after_seconds(response: httpx.Response, *, attempt: int) -> float:
    """Prefer Retry-After header; else exponential backoff. Always ≥ min wait."""
    header = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if header:
        try:
            return max(_HTTP_MIN_WAIT_S, float(header))
        except ValueError:
            pass
    raw = _HTTP_BASE_BACKOFF_S * (2 ** (attempt - 1))
    return max(_HTTP_MIN_WAIT_S, min(_HTTP_BACKOFF_CAP_S, raw))


def _backoff_seconds(*, attempt: int) -> float:
    raw = _HTTP_BASE_BACKOFF_S * (2 ** (attempt - 1))
    return max(_HTTP_MIN_WAIT_S, min(_HTTP_BACKOFF_CAP_S, raw))


def _get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    max_attempts: int = _HTTP_MAX_ATTEMPTS,
) -> Any:
    """GET JSON with retries on 429 / 503 (honors Retry-After when present)."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.get(
                url,
                headers=headers or {},
                params=params,
                timeout=timeout,
                follow_redirects=True,
            )
            if response.status_code in {429, 503} and attempt < max_attempts:
                delay = _retry_after_seconds(response, attempt=attempt)
                logger.info(
                    "HTTP %s from %s (attempt %d/%d); sleeping %.1fs.",
                    response.status_code,
                    url.split("?", 1)[0],
                    attempt,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status in {429, 503} and attempt < max_attempts:
                delay = _retry_after_seconds(exc.response, attempt=attempt)
                time.sleep(delay)
                continue
            raise
        except httpx.TransportError as exc:
            last_error = exc
            if attempt < max_attempts:
                delay = _backoff_seconds(attempt=attempt)
                logger.info(
                    "Transport error for %s (attempt %d/%d); sleeping %.1fs.",
                    url.split("?", 1)[0],
                    attempt,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
    assert last_error is not None
    raise last_error


def _get_bytes(url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> bytes:
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.content


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text, flags=re.I)
    text = text.removesuffix(".pdf")
    text = text.split("v")[0] if re.search(r"v\d+$", text) else text
    # Classic ids like hep-th/9901001 — keep slash form; new ids like 2106.09685
    text = text.strip()
    if not text:
        return None
    return text


def stable_paper_id(*, doi: str | None, arxiv_id: str | None, s2_id: str | None) -> str:
    if doi:
        return f"doi:{doi.lower().removeprefix('https://doi.org/').removeprefix('http://doi.org/')}"
    if arxiv_id:
        return f"arxiv:{normalize_arxiv_id(arxiv_id) or arxiv_id}"
    if s2_id:
        return f"s2:{s2_id}"
    return "paper:unknown"


def _arxiv_id_from_url(url: str | None) -> str | None:
    if not url or "arxiv.org" not in url.lower():
        return None
    return normalize_arxiv_id(url)


def _arxiv_id_from_locations(work: dict[str, Any]) -> str | None:
    """Pull arXiv id from OpenAlex locations / OA URLs when ids.arxiv is absent."""
    candidates: list[str] = []
    for key in ("primary_location", "best_oa_location"):
        loc = work.get(key)
        if isinstance(loc, dict):
            candidates.extend(
                [
                    str(loc.get("landing_page_url") or ""),
                    str(loc.get("pdf_url") or ""),
                ]
            )
    for loc in work.get("locations") or []:
        if isinstance(loc, dict):
            candidates.extend(
                [
                    str(loc.get("landing_page_url") or ""),
                    str(loc.get("pdf_url") or ""),
                ]
            )
    oa = work.get("open_access")
    if isinstance(oa, dict) and oa.get("oa_url"):
        candidates.append(str(oa["oa_url"]))
    for candidate in candidates:
        aid = _arxiv_id_from_url(candidate)
        if aid:
            return aid
    return None


def _short_http_error(exc: Exception) -> str:
    """Compact note text — avoid dumping full query URLs into analyze notes."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        code = exc.response.status_code
        if code == 429:
            return "HTTP 429 (rate limited)"
        return f"HTTP {code}"
    text = str(exc).strip()
    if "429" in text:
        return "HTTP 429 (rate limited)"
    return text[:120] or type(exc).__name__


class SemanticScholarClient:
    """Primary search — Semantic Scholar Graph API."""

    BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        api_key: str = "",
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        min_interval_s: float = _S2_MIN_INTERVAL_S,
    ) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.min_interval_s = max(0.0, min_interval_s)
        self._last_request_at = 0.0

    def _pace(self) -> None:
        if self.min_interval_s <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at > 0 and elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def search(self, query: str, *, limit: int = 40) -> list[Paper]:
        if not query.strip():
            return []
        headers: dict[str, str] = {"User-Agent": "LabPilot/0.1 (research)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        fields = (
            "paperId,title,abstract,year,venue,citationCount,authors,"
            "externalIds,url,openAccessPdf"
        )
        self._pace()
        try:
            data = _get_json(
                f"{self.BASE}/paper/search",
                headers=headers,
                params={
                    "query": query,
                    "limit": max(1, min(limit, 100)),
                    "fields": fields,
                },
                timeout=self.timeout,
            )
        finally:
            self._last_request_at = time.monotonic()
        papers: list[Paper] = []
        total = max(int(data.get("total") or 1), 1)
        for idx, item in enumerate(data.get("data") or []):
            paper = self._normalize(item, rank_hint=1.0 - (idx / max(total, limit)))
            if paper is not None:
                papers.append(paper)
        return papers

    def _normalize(self, item: dict[str, Any], *, rank_hint: float) -> Paper | None:
        title = (item.get("title") or "").strip()
        if not title:
            return None
        external = item.get("externalIds") or {}
        arxiv_id = normalize_arxiv_id(external.get("ArXiv") or external.get("arXiv"))
        doi = external.get("DOI")
        s2_id = item.get("paperId")
        authors = [
            a.get("name")
            for a in (item.get("authors") or [])
            if isinstance(a, dict) and a.get("name")
        ]
        oa = item.get("openAccessPdf") or {}
        pdf_url = oa.get("url") if isinstance(oa, dict) else None
        relevance = max(0.05, min(1.0, float(rank_hint)))
        paper_id = stable_paper_id(doi=doi, arxiv_id=arxiv_id, s2_id=s2_id)
        urls: dict[str, str] = {}
        if item.get("url"):
            urls["semantic_scholar"] = str(item["url"])
        if s2_id:
            urls["s2"] = f"https://www.semanticscholar.org/paper/{s2_id}"
        return Paper(
            id=paper_id,
            title=title,
            abstract=(item.get("abstract") or "") or "",
            authors=[str(a) for a in authors],
            year=item.get("year"),
            venue=item.get("venue"),
            citations=item.get("citationCount"),
            pdf_url=pdf_url,
            arxiv_id=arxiv_id,
            doi=doi,
            relevance=relevance,
            urls=urls,
            payload={"semantic_scholar": {"paperId": s2_id}},
        )


class OpenAlexClient:
    """Enrichment — citations, concepts, OA links. Soft search fallback if S2 fails."""

    BASE = "https://api.openalex.org"

    def __init__(
        self,
        *,
        mailto: str = "",
        api_key: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.mailto = mailto.strip()
        self.api_key = api_key.strip()
        self.timeout = timeout

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = dict(extra or {})
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def search(self, query: str, *, limit: int = 40) -> list[Paper]:
        """Soft discovery path when Semantic Scholar is unavailable."""
        if not query.strip():
            return []
        data = _get_json(
            f"{self.BASE}/works",
            params=self._params(
                {
                    "search": query,
                    "per_page": max(1, min(limit, 50)),
                    "sort": "relevance_score:desc",
                }
            ),
            timeout=self.timeout,
        )
        papers: list[Paper] = []
        results = data.get("results") or []
        for idx, work in enumerate(results):
            if not isinstance(work, dict):
                continue
            paper = self._work_to_paper(
                work, rank_hint=1.0 - (idx / max(len(results), limit))
            )
            if paper is not None:
                papers.append(paper)
        return papers

    def _work_to_paper(self, work: dict[str, Any], *, rank_hint: float) -> Paper | None:
        title = (work.get("display_name") or work.get("title") or "").strip()
        if not title:
            return None
        ids = work.get("ids") or {}
        doi = None
        arxiv_id = None
        if isinstance(ids, dict):
            if ids.get("doi"):
                doi = str(ids["doi"]).removeprefix("https://doi.org/")
            # OpenAlex sometimes exposes arxiv under ids
            for key, val in ids.items():
                if "arxiv" in str(key).lower() and val:
                    arxiv_id = normalize_arxiv_id(str(val).split("/")[-1])
                    break
        # Also check open access / locations for arxiv pdf
        if not arxiv_id:
            arxiv_id = _arxiv_id_from_locations(work)
        authors = []
        for authorship in work.get("authorships") or []:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author") or {}
            name = author.get("display_name") if isinstance(author, dict) else None
            if name:
                authors.append(str(name))
        abstract = ""
        inv = work.get("abstract_inverted_index")
        if isinstance(inv, dict) and inv:
            # Reconstruct rough abstract from inverted index
            positions: list[tuple[int, str]] = []
            for word, idxs in inv.items():
                if isinstance(idxs, list):
                    for i in idxs:
                        if isinstance(i, int):
                            positions.append((i, str(word)))
            positions.sort()
            abstract = " ".join(w for _, w in positions)
        oa = work.get("open_access") if isinstance(work.get("open_access"), dict) else {}
        pdf_url = oa.get("oa_url") if isinstance(oa, dict) else None
        concepts = []
        for c in work.get("concepts") or work.get("topics") or []:
            if isinstance(c, dict):
                name = c.get("display_name") or c.get("name")
                if name:
                    concepts.append(str(name))
        paper_id = stable_paper_id(doi=doi, arxiv_id=arxiv_id, s2_id=None)
        if paper_id == "paper:unknown" and work.get("id"):
            paper_id = f"openalex:{str(work['id']).rstrip('/').split('/')[-1]}"
        urls: dict[str, str] = {}
        if isinstance(ids, dict) and ids.get("openalex"):
            urls["openalex"] = str(ids["openalex"])
        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=work.get("publication_year"),
            venue=None,
            citations=work.get("cited_by_count"),
            concepts=concepts[:20],
            pdf_url=str(pdf_url) if pdf_url else None,
            arxiv_id=arxiv_id,
            doi=doi,
            relevance=max(0.05, min(1.0, float(rank_hint))),
            urls=urls,
            payload={"openalex": {"id": work.get("id")}},
        )

    def enrich(self, paper: Paper) -> Paper:
        work = self._lookup(paper)
        if work is None:
            return paper
        return self._merge(paper, work)

    def _lookup(self, paper: Paper) -> dict[str, Any] | None:
        try:
            if paper.doi:
                doi = paper.doi.lower().removeprefix("https://doi.org/")
                return _get_json(
                    f"{self.BASE}/works/https://doi.org/{doi}",
                    params=self._params(),
                    timeout=self.timeout,
                )
            if paper.arxiv_id:
                aid = normalize_arxiv_id(paper.arxiv_id) or paper.arxiv_id
                data = _get_json(
                    f"{self.BASE}/works",
                    params=self._params(
                        {
                            "filter": f"ids.arxiv:{aid}",
                            "per_page": 1,
                        }
                    ),
                    timeout=self.timeout,
                )
                results = data.get("results") or []
                return results[0] if results else None
            if paper.title:
                data = _get_json(
                    f"{self.BASE}/works",
                    params=self._params(
                        {
                            "search": paper.title,
                            "per_page": 1,
                            "sort": "relevance_score:desc",
                        }
                    ),
                    timeout=self.timeout,
                )
                results = data.get("results") or []
                return results[0] if results else None
        except Exception:
            logger.info("OpenAlex lookup failed for %s", paper.id, exc_info=False)
            return None
        return None

    def _merge(self, paper: Paper, work: dict[str, Any]) -> Paper:
        updated = paper.model_copy(deep=True)
        cites = work.get("cited_by_count")
        if isinstance(cites, int):
            updated.citations = cites
        concepts = []
        for c in work.get("concepts") or work.get("topics") or []:
            if isinstance(c, dict):
                name = c.get("display_name") or c.get("name")
                if name:
                    concepts.append(str(name))
        if concepts:
            updated.concepts = list(dict.fromkeys([*updated.concepts, *concepts]))[:20]
        if not updated.year and work.get("publication_year"):
            updated.year = work.get("publication_year")
        oa = (work.get("open_access") or {}) if isinstance(work.get("open_access"), dict) else {}
        oa_url = oa.get("oa_url")
        if oa_url and not updated.pdf_url:
            updated.pdf_url = str(oa_url)
        ids = work.get("ids") or {}
        if isinstance(ids, dict):
            if ids.get("doi") and not updated.doi:
                updated.doi = str(ids["doi"]).removeprefix("https://doi.org/")
            if ids.get("openalex"):
                updated.urls["openalex"] = str(ids["openalex"])
            if not updated.arxiv_id:
                for key, val in ids.items():
                    if "arxiv" in str(key).lower() and val:
                        updated.arxiv_id = normalize_arxiv_id(str(val).split("/")[-1])
                        if updated.arxiv_id:
                            break
        if not updated.arxiv_id:
            updated.arxiv_id = _arxiv_id_from_locations(work)
        if updated.arxiv_id:
            updated.urls.setdefault(
                "arxiv", f"https://arxiv.org/abs/{normalize_arxiv_id(updated.arxiv_id)}"
            )
        updated.payload["openalex"] = {"id": work.get("id"), "cited_by_count": cites}
        return updated


class ArxivClient:
    """Search + PDF attach via the official ``arxiv`` package (export.arxiv.org).

    Rate limit: at most one request every three seconds per client
    (``delay_seconds=3``). Search is a discovery peer of Semantic Scholar;
    attach/download remain the enrich hop for papers that already have an id.
    """

    API = "https://export.arxiv.org/api/query"
    # Official API: ≤1 request / 3s. Keep page_size large so one query = one GET
    # when limit ≤ page_size.
    DEFAULT_PAGE_SIZE = 1000
    DEFAULT_DELAY_SECONDS = 3.0
    DEFAULT_NUM_RETRIES = 3

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        page_size: int = DEFAULT_PAGE_SIZE,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        num_retries: int = DEFAULT_NUM_RETRIES,
        client: Any | None = None,
    ) -> None:
        self.timeout = timeout
        self.page_size = max(1, page_size)
        self.delay_seconds = max(3.0, float(delay_seconds))
        self.num_retries = max(0, num_retries)
        self._client = client

    def _arxiv_client(self) -> Any:
        if self._client is not None:
            return self._client
        import arxiv

        self._client = arxiv.Client(
            page_size=self.page_size,
            delay_seconds=self.delay_seconds,
            num_retries=self.num_retries,
        )
        return self._client

    def search(self, query: str, *, limit: int = 40) -> list[Paper]:
        """Relevance-ranked arXiv search (soft-fail at the provider layer)."""
        if not query.strip():
            return []
        import arxiv

        max_results = max(1, min(int(limit), self.page_size))
        search = arxiv.Search(
            query=query.strip(),
            id_list=[],
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        papers: list[Paper] = []
        for idx, result in enumerate(self._arxiv_client().results(search)):
            if idx >= max_results:
                break
            paper = self._result_to_paper(
                result, rank_hint=1.0 - (idx / max(max_results, 1))
            )
            if paper is not None:
                papers.append(paper)
        return papers

    def _result_to_paper(self, result: Any, *, rank_hint: float) -> Paper | None:
        title = (getattr(result, "title", None) or "").strip()
        if not title:
            return None
        short_id = ""
        if hasattr(result, "get_short_id"):
            short_id = str(result.get_short_id() or "")
        arxiv_id = normalize_arxiv_id(short_id) or normalize_arxiv_id(
            str(getattr(result, "entry_id", "") or "")
        )
        if not arxiv_id:
            return None
        doi_raw = getattr(result, "doi", None) or None
        doi = str(doi_raw).strip() if doi_raw else None
        authors = []
        for author in getattr(result, "authors", None) or []:
            name = getattr(author, "name", None) or str(author)
            if name:
                authors.append(str(name))
        year = None
        published = getattr(result, "published", None)
        if published is not None and hasattr(published, "year"):
            year = int(published.year)
        pdf_url = getattr(result, "pdf_url", None) or self.pdf_url_for(arxiv_id)
        relevance = max(0.05, min(1.0, float(rank_hint)))
        paper_id = stable_paper_id(doi=doi, arxiv_id=arxiv_id, s2_id=None)
        return Paper(
            id=paper_id,
            title=title,
            abstract=(getattr(result, "summary", None) or "") or "",
            authors=authors,
            year=year,
            venue="arXiv",
            citations=None,
            pdf_url=str(pdf_url) if pdf_url else None,
            arxiv_id=arxiv_id,
            doi=doi,
            relevance=relevance,
            urls={
                "arxiv": f"https://arxiv.org/abs/{arxiv_id}",
            },
            payload={"arxiv": {"entry_id": str(getattr(result, "entry_id", "") or "")}},
        )

    def pdf_url_for(self, arxiv_id: str) -> str:
        aid = normalize_arxiv_id(arxiv_id) or arxiv_id
        return f"https://export.arxiv.org/pdf/{aid}.pdf"

    def attach_pdf_meta(self, paper: Paper) -> Paper:
        if not paper.arxiv_id:
            return paper
        updated = paper.model_copy(deep=True)
        updated.pdf_url = updated.pdf_url or self.pdf_url_for(paper.arxiv_id)
        updated.urls.setdefault(
            "arxiv", f"https://arxiv.org/abs/{normalize_arxiv_id(paper.arxiv_id)}"
        )
        return updated

    def download_pdf(self, arxiv_id: str) -> bytes:
        return _get_bytes(self.pdf_url_for(arxiv_id), timeout=90.0)


class HuggingFacePapersClient:
    """Code / Hub attach — replaces dead Papers with Code (Plan 6).

    Requires an arXiv id today. Non-arXiv attach is backlog.
    """

    BASE = "https://huggingface.co"

    def __init__(self, token: str = "", *, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.token = token.strip()
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "LabPilot/0.1 (research)"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def attach(self, paper: Paper) -> Paper:
        aid = normalize_arxiv_id(paper.arxiv_id)
        if not aid:
            # Backlog: DOI / title match without arXiv id.
            return paper
        try:
            meta = _get_json(
                f"{self.BASE}/api/papers/{quote(aid)}",
                headers=self._headers(),
                timeout=self.timeout,
            )
        except Exception:
            logger.info("HF paper page miss for arxiv:%s", aid, exc_info=False)
            return paper

        updated = paper.model_copy(deep=True)
        github = meta.get("githubRepo")
        if github:
            url = str(github).strip()
            if url and url not in updated.github_urls:
                updated.github_urls.append(url)
        project = meta.get("projectPage")
        if project:
            updated.urls["project"] = str(project)

        models = [
            m.get("id")
            for m in (meta.get("linkedModels") or [])
            if isinstance(m, dict) and m.get("id")
        ]
        datasets = [
            d.get("id")
            for d in (meta.get("linkedDatasets") or [])
            if isinstance(d, dict) and d.get("id")
        ]
        spaces = [
            s.get("id")
            for s in (meta.get("linkedSpaces") or [])
            if isinstance(s, dict) and s.get("id")
        ]
        for ds in datasets[:15]:
            if ds not in updated.datasets:
                updated.datasets.append(str(ds))
        updated.payload["hf"] = {
            "arxiv_id": aid,
            "upvotes": meta.get("upvotes"),
            "models": [str(m) for m in models[:20]],
            "datasets": [str(d) for d in datasets[:20]],
            "spaces": [str(s) for s in spaces[:10]],
            "githubRepo": github,
        }
        updated.urls.setdefault("huggingface", f"{self.BASE}/papers/{aid}")
        return updated
