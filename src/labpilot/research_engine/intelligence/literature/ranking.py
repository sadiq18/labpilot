"""Paper ranking: age decay, citation velocity, foundational/recent mix."""

from __future__ import annotations

from datetime import UTC, datetime

from labpilot.research_engine.intelligence.literature.models import Paper

# Mild age penalty — classics still score if highly relevant + cited.
DEFAULT_AGE_ALPHA = 0.55
# Papers published within this many years are "recent".
DEFAULT_RECENT_YEARS = 3
# Share of extract slots reserved for recent (rest → foundational).
DEFAULT_RECENT_SHARE = 0.7


def current_year() -> int:
    return datetime.now(UTC).year


def years_old(paper: Paper, *, as_of_year: int | None = None) -> float:
    """Whole years since publication. Missing year → 0 (no age penalty)."""
    if paper.year is None:
        return 0.0
    ref = as_of_year if as_of_year is not None else current_year()
    return float(max(0, ref - int(paper.year)))


def citation_velocity(paper: Paper, *, as_of_year: int | None = None) -> float:
    """Citations per year of age (floor age at 1 so brand-new papers work)."""
    cites = float(paper.citations or 0)
    age = max(years_old(paper, as_of_year=as_of_year), 1.0)
    return cites / age


def age_decay(
    paper: Paper,
    *,
    alpha: float = DEFAULT_AGE_ALPHA,
    as_of_year: int | None = None,
) -> float:
    """``1 / (1 + years_old)^α``."""
    age = years_old(paper, as_of_year=as_of_year)
    return 1.0 / ((1.0 + age) ** alpha)


def is_recent(
    paper: Paper,
    *,
    recent_years: int = DEFAULT_RECENT_YEARS,
    as_of_year: int | None = None,
) -> bool:
    """Unknown year counts as recent so we do not bury incomplete metadata."""
    if paper.year is None:
        return True
    return years_old(paper, as_of_year=as_of_year) <= float(recent_years)


def rank_score(
    paper: Paper,
    *,
    alpha: float = DEFAULT_AGE_ALPHA,
    as_of_year: int | None = None,
) -> float:
    """relevance × velocity factor × age decay.

    Uses citation **velocity** (cites/year) instead of raw citation count, then
    multiplies by ``1/(1+age)^α`` so old high-cite papers do not dominate solely
    on absolute citations. Foundational classics still get extract slots via
    :func:`select_for_extract` bucketing.
    """
    velocity = citation_velocity(paper, as_of_year=as_of_year)
    # Soft scale — zero-velocity papers still compete on relevance × decay.
    velocity_factor = 1.0 + (velocity**0.5) / 10.0
    decay = age_decay(paper, alpha=alpha, as_of_year=as_of_year)
    return paper.relevance * velocity_factor * decay


def select_for_extract(
    papers: list[Paper],
    *,
    limit: int,
    recent_share: float = DEFAULT_RECENT_SHARE,
    recent_years: int = DEFAULT_RECENT_YEARS,
    alpha: float = DEFAULT_AGE_ALPHA,
    as_of_year: int | None = None,
) -> list[Paper]:
    """Mix recent + foundational buckets for extract top-N.

    1. Split by :func:`is_recent`.
    2. Rank each bucket by :func:`rank_score`.
    3. Fill ``ceil(limit * recent_share)`` from recent, remainder from
       foundational; if a bucket is short, backfill from the other.
    4. Return interleaved list (recent picks first, then foundational) so
       artifact order surfaces newer work without dropping classics.
    """
    if limit <= 0 or not papers:
        return []

    recent_n = max(1, round(limit * recent_share)) if limit > 1 else 1
    # For limit=1 prefer recent.
    foundational_n = max(0, limit - recent_n)

    recent = sorted(
        (p for p in papers if is_recent(p, recent_years=recent_years, as_of_year=as_of_year)),
        key=lambda p: rank_score(p, alpha=alpha, as_of_year=as_of_year),
        reverse=True,
    )
    foundational = sorted(
        (
            p
            for p in papers
            if not is_recent(p, recent_years=recent_years, as_of_year=as_of_year)
        ),
        key=lambda p: rank_score(p, alpha=alpha, as_of_year=as_of_year),
        reverse=True,
    )

    picked: list[Paper] = []
    seen: set[str] = set()

    def _take(pool: list[Paper], n: int) -> list[Paper]:
        out: list[Paper] = []
        for paper in pool:
            if len(out) >= n:
                break
            if paper.id in seen:
                continue
            seen.add(paper.id)
            out.append(paper)
        return out

    picked.extend(_take(recent, recent_n))
    picked.extend(_take(foundational, foundational_n))

    # Backfill whichever bucket ran short.
    if len(picked) < limit:
        leftover = sorted(
            (p for p in papers if p.id not in seen),
            key=lambda p: rank_score(p, alpha=alpha, as_of_year=as_of_year),
            reverse=True,
        )
        picked.extend(_take(leftover, limit - len(picked)))

    return picked[:limit]
