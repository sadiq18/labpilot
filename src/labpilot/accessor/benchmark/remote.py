"""A competition's file list, without its bytes.

M24. `capture` walks a local directory, which means a media competition costs
its full download before it can be a fixture: `biohub-cell-tracking` is 4.5 MB
zarr chunks, ≥0.99 GB in the first two hundred files and still paging when the
API rate-limited an attempt to count them. That is why the corpus is five
tabular fixtures and none of the text, image or audio ones the plan names.

The listing is all modality detection needs. `_detect_image` counts by
extension and never opens a file; `_role_of` reads directory parts. Names and
sizes are exactly what the Kaggle API returns, so a media fixture costs
kilobytes and one request page per two hundred files.

**What this cannot carry, and says so.** The API returns no checksum, so a
remotely-listed file has a name and a size and nothing that proves the bytes.
A locally-walked listing records a sha256 of what it saw. The two are not the
same evidence and `RemoteListing.verifiable` is how a fixture stops itself
claiming otherwise — tier 3 re-probing the real dataset is what closes the gap,
and until that job exists a remote listing is a paraphrase with provenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import sleep
from typing import Any, Protocol

logger = logging.getLogger(__name__)

__all__ = ["ListingUnavailable", "RemoteFile", "RemoteListing", "fetch_listing"]

#: Kaggle's own maximum. Larger values are silently clamped by the API, which
#: would make a page look complete when it was not.
_PAGE_SIZE = 200

#: A backstop, not a policy. `birdclef` has tens of thousands of files and the
#: listing is the cheap part; this exists so a runaway pagination bug costs a
#: bounded number of requests rather than an afternoon of them.
_MAX_PAGES = 400

#: Kaggle rate-limits a fast page walk. `biohub-cell-tracking` needs ~120 pages
#: and was cut off at 23,800 files twice before this existed. A fifth of a
#: second between pages costs ~24 seconds on that competition, which is nothing
#: against the download it replaces.
_PAGE_PAUSE_SECONDS = 0.2

#: Retries on a 429, with the pause doubling each time. Bounded because a limit
#: that has not lifted in three escalating waits is a limit, not a blip, and
#: sitting in a loop is worse than saying so.
_RATE_LIMIT_RETRIES = 3


class ListingUnavailable(RuntimeError):
    """The listing could not be fetched, with a reason an operator can act on."""


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int


@dataclass(frozen=True)
class RemoteListing:
    """Every file a competition holds, as the API reports it."""

    slug: str
    files: tuple[RemoteFile, ...]
    #: Whether the page walk finished. A truncated listing has the wrong ratios,
    #: and ratios are the entire reason to record one — so a fixture built from
    #: a partial listing would validate the wrong answer rather than fail.
    complete: bool
    #: False, always, for a remote listing: the API returns no checksum. Carried
    #: rather than assumed so `capture` can record which kind of evidence a
    #: fixture rests on.
    verifiable: bool = False

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


class _ListsFiles(Protocol):
    def competition_list_files(
        self, competition: str, page_token: str | None = ..., page_size: int = ...
    ) -> Any: ...


def _page(api: _ListsFiles, slug: str, token: str | None, so_far: int) -> Any:
    """One page, retrying a rate limit with a widening pause.

    Backoff belongs here rather than in advice to the operator: the listing is
    a hundred-odd requests for a large competition, being told to "wait and
    re-run" restarts all of them, and a re-run hits the same limit at the same
    place.
    """
    pause = _PAGE_PAUSE_SECONDS
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            return api.competition_list_files(slug, page_token=token, page_size=_PAGE_SIZE)
        except Exception as exc:  # noqa: BLE001 — the reason is the useful part
            detail = str(exc)
            limited = "429" in detail or "Too Many Requests" in detail
            if not limited:
                raise ListingUnavailable(f"could not list {slug!r}: {detail}") from exc
            if attempt == _RATE_LIMIT_RETRIES:
                raise ListingUnavailable(
                    f"Kaggle rate-limited the listing for {slug!r} after {so_far} file(s), "
                    f"through {_RATE_LIMIT_RETRIES} backoffs. The limit is not lifting; "
                    "try again later."
                ) from exc
            pause *= 4
            logger.info("rate-limited listing %s at %d file(s); waiting %.1fs", slug, so_far, pause)
            sleep(pause)
    raise AssertionError("unreachable")  # pragma: no cover


def fetch_listing(slug: str, api: _ListsFiles) -> RemoteListing:
    """Every file in `slug`, paged to the end.

    Raises `ListingUnavailable` rather than returning a short list. A partial
    listing is worse than none: the counts and ratios are the fixture's whole
    content, and half of them describe a dataset that does not exist.
    """
    files: list[RemoteFile] = []
    token: str | None = None
    for page in range(_MAX_PAGES):
        if page:
            sleep(_PAGE_PAUSE_SECONDS)
        response = _page(api, slug, token, len(files))
        batch = list(getattr(response, "files", None) or [])
        for entry in batch:
            name = str(getattr(entry, "name", "") or "")
            if not name:
                continue
            files.append(RemoteFile(name=name, size=int(getattr(entry, "total_bytes", 0) or 0)))

        token = getattr(response, "next_page_token", None) or None
        if not token or not batch:
            return RemoteListing(slug=slug, files=tuple(files), complete=True)
        logger.debug("listing %s: %d file(s) after page %d", slug, len(files), page + 1)

    raise ListingUnavailable(
        f"{slug!r} did not finish listing within {_MAX_PAGES} pages "
        f"({len(files)} file(s) so far); refusing to build a fixture from a partial listing"
    )
