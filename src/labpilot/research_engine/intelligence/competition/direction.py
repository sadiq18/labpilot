"""Which way is better — resolve a competition's metric direction.

Every conclusion the research engine draws is signed. "Did this technique help?"
is `treatment - parent`, and whether that number is good news depends entirely on
whether the metric is maximised or minimised. Getting it wrong does not degrade
the answer, it *reverses* it.

That is not hypothetical. Measured on rogii 2026-08-07: `build_evidence_card`
took ``maximize: bool = True`` as a default and **no call site passed it**, so all
fifteen evidence cards were built as if MSE were a score to maximise. The single
genuine improvement the system ever produced — SWA cutting MSE 194.80 to 190.97 —
is recorded ``rejected``, while a run that made the metric worse is ``accepted``.
The competition profile said ``direction: minimize`` the whole time; nothing read
it.

So this module exists to give that question exactly one answer, resolved from the
artifacts that already record it, and to make "I don't know" an outcome the caller
must handle rather than a silent `True`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MINIMIZE = "minimize"
_MAXIMIZE = "maximize"


def _direction_to_maximize(raw: Any) -> bool | None:
    text = str(raw or "").strip().lower()
    if text.startswith("min"):
        return False
    if text.startswith("max"):
        return True
    return None


def _from_competition_json(path: Path) -> bool | None:
    """Read the metric direction from a ``competition.json``.

    Two shapes share the filename. The hand-written workspace config nests the
    metric under ``metric``; `CompetitionParser.save` writes a
    ``CompetitionSpec``, whose metric is ``evaluation_metric``. Reading only
    the first meant every parser-written spec — the machine-generated ones,
    which is most of them — answered "unknown" here and fell through to the
    profile artifact, so a competition with a perfectly explicit direction on
    disk could still be unresolvable.

    Read as a dict rather than through `CompetitionSpec`: the model defaults
    `direction` to ``"maximize"``, which would turn an absent field into a
    confident wrong answer instead of the ``None`` that lets the caller keep
    looking.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — an unreadable file is "unknown"
        logger.debug("competition.json unreadable at %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    for key in ("metric", "evaluation_metric"):
        metric = data.get(key)
        if isinstance(metric, dict):
            # The first block present decides, including when its answer is
            # "unknown". Falling through on an unparseable direction would let
            # a machine-written `evaluation_metric` override a hand-written
            # `metric` whose direction is merely misspelled — the deliberate
            # source losing to the generated one.
            return _direction_to_maximize(metric.get("direction"))
    return None


def _from_profile_artifact(extracted_dir: Path, competition: str) -> bool | None:
    """Read ``metadata.profile.metric.direction`` from the Analyze profile artifact.

    This is where rogii's ``minimize`` actually lived, so the fallback is not
    speculative — it is the source that was present and unread.
    """
    candidate = extracted_dir / "misc" / f"competition_{competition}.json"
    if not candidate.is_file():
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("profile artifact unreadable at %s: %s", candidate, exc)
        return None
    profile = (data or {}).get("metadata", {}).get("profile", {})
    metric = profile.get("metric") if isinstance(profile, dict) else None
    if isinstance(metric, dict):
        return _direction_to_maximize(metric.get("direction"))
    return None


def resolve_maximize(
    *,
    competition: str,
    workspace_root: Path | None = None,
    knowledge_root: Path | None = None,
    extracted_dir: Path | None = None,
) -> bool | None:
    """``True`` to maximise, ``False`` to minimise, ``None`` if unknowable.

    Sources are tried nearest-first: the ``competition.json`` the run itself was
    given, then the knowledge copy, then the Analyze profile artifact. ``None``
    is a real answer and callers must not paper over it — see
    `evidence/builder.py::build_evidence_card`, which refuses to write a card
    rather than record a signed conclusion it cannot orient.
    """
    for path in (workspace_root, knowledge_root):
        if path is not None:
            found = _from_competition_json(Path(path) / "competition.json")
            if found is not None:
                return found
    if extracted_dir is not None:
        return _from_profile_artifact(Path(extracted_dir), competition)
    return None


def direction_label(maximize: bool) -> str:
    return _MAXIMIZE if maximize else _MINIMIZE
