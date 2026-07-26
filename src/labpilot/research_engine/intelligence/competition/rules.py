"""Best-effort fetch of competition rules page text for brief/reflection context."""

import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def fetch_rules_excerpt(rules_url: str, timeout: float = 30.0) -> str:
    """Fetch and extract plain text from a competition rules page."""
    if not rules_url.strip():
        return ""

    try:
        response = httpx.get(rules_url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except Exception:
        logger.warning("Could not fetch rules page at %s.", rules_url, exc_info=True)
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text
