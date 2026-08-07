"""Derive technique vocabulary status from evidence cards (M-25).

Status is recomputed from the current card set — never stepped — so repairing a
card afterwards changes the vocabulary the same way ``belief_repair`` does for
beliefs.

Step 2 adds campaign-aged ``dormant`` and consumer filters elsewhere. Fresh
unmeasured techniques stay ``candidate`` until ``DORMANT_AFTER_CAMPAIGNS``
later sessions pass without selection — aging is what keeps the vocabulary open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labpilot.research_engine.execution.technique.status_constants import (
    DORMANT_AFTER_CAMPAIGNS,
)
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter

logger = logging.getLogger(__name__)

#: Same bar as ``ClaimPromoter`` — a second epsilon would let the two drift.
_NO_EFFECT_EPSILON = 1e-9


@dataclass(frozen=True)
class TechniqueStatusDerivation:
    """One technique's derived status and the measurements behind it."""

    technique_id: str
    name: str
    status: str
    reason: str
    observations: int = 0
    net_effect: float = 0.0
    signed_net: float = 0.0
    evidence_card_id: str | None = None


def _list_cards(promoter: ClaimPromoter) -> list[Any]:
    try:
        return list(promoter._evidence.list())  # noqa: SLF001 — shared card walk
    except Exception:  # noqa: BLE001
        return []


def _signed_measured_effect(
    promoter: ClaimPromoter,
    technique: str,
    cards: list[Any],
) -> tuple[int, float, float, str | None]:
    """``(observations, raw_net, signed_net, last_card_id)`` for ``technique``.

    Raw net matches ``ClaimPromoter.measured_effect``. Signed net orients credit
    with each card's ``maximize`` flag — the same flip ``_claim_updates_from_attribution``
    uses — so MSE improvements read as positive without a second definition.
    """
    label = technique.strip().lower()
    if not label:
        return 0, 0.0, 0.0, None

    observations = 0
    raw_net = 0.0
    signed_net = 0.0
    last_card: str | None = None
    for card in cards:
        if not promoter._card_compared_something_real(card):  # noqa: SLF001
            continue
        matched = False
        for name, credit in (card.technique_attribution or {}).items():
            if str(name).strip().lower() != label:
                continue
            matched = True
            observations += 1
            try:
                value = float(credit)
            except (TypeError, ValueError):
                continue
            raw_net += value
            maximize = bool(getattr(card, "maximize", True))
            signed_net += value if maximize else -value
        if matched:
            last_card = str(getattr(card, "id", "") or "") or last_card
    return observations, raw_net, signed_net, last_card


def selected_technique_names(knowledge_dir: Path, competition: str) -> set[str]:
    """Technique labels that appear on any hypothesis for this competition."""
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    names: set[str] = set()
    for hyp in HypothesisStore(Path(knowledge_dir), competition).list():
        if hyp.technique:
            names.add(str(hyp.technique).strip().lower())
        for item in hyp.technique_stack or []:
            if str(item).strip():
                names.add(str(item).strip().lower())
        for item in hyp.combo_techniques or []:
            if str(item).strip():
                names.add(str(item).strip().lower())
    return names


def campaign_created_ats(knowledge_dir: Path, competition: str) -> list[str]:
    """Session ``created_at`` stamps for ``competition``, oldest first.

    Reads ``os_sessions`` through SqliteClient — not ConductorStore — so
    ``execution/`` stays free of conductor imports (architecture boundary).
    """
    from labpilot.accessor.sqlite import SqliteClient
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    client = SqliteClient(ResearchPaths(Path(knowledge_dir), competition).db_path)
    try:
        rows = client.conn.execute(
            "SELECT created_at FROM os_sessions WHERE competition = ? "
            "ORDER BY created_at",
            (competition,),
        ).fetchall()
        return [str(r["created_at"]) for r in rows if r["created_at"]]
    finally:
        client.close()


def campaigns_since(created_at: str | None, session_times: list[str]) -> int:
    """How many campaigns started after ``created_at`` (string-ordered ISO times)."""
    if not created_at:
        return 0
    stamp = str(created_at)
    return sum(1 for t in session_times if t > stamp)


def derive_technique_status(
    name: str,
    promoter: ClaimPromoter,
    *,
    cards: list[Any] | None = None,
    selected: set[str] | None = None,
    created_at: str | None = None,
    session_times: list[str] | None = None,
    dormant_after: int = DORMANT_AFTER_CAMPAIGNS,
) -> tuple[str, str, int, float, float, str | None]:
    """Return ``(status, reason, observations, raw_net, signed_net, evidence_card_id)``.

    ``dormant`` requires all three: never measured, never selected, and at least
    ``dormant_after`` campaigns started after the technique was proposed. Without
    the aging clause, first recompute would permanently exclude every new row.
    """
    card_list = cards if cards is not None else _list_cards(promoter)
    observations, raw_net, signed_net, last_card = _signed_measured_effect(
        promoter, name, card_list
    )
    if observations == 0:
        label = name.strip().lower()
        age = campaigns_since(created_at, session_times or [])
        if (
            selected is not None
            and label
            and label not in selected
            and age >= dormant_after
        ):
            return (
                "dormant",
                (
                    f"proposed {age} campaign(s) ago, never selected for a "
                    "hypothesis, never measured"
                ),
                0,
                0.0,
                0.0,
                None,
            )
        return (
            "candidate",
            "proposed; no conclusive evidence card attributes a result yet",
            0,
            0.0,
            0.0,
            None,
        )
    if abs(signed_net) < _NO_EFFECT_EPSILON:
        return (
            "candidate",
            f"{observations} observation(s) but net signed effect is ~0",
            observations,
            raw_net,
            signed_net,
            last_card,
        )
    if signed_net <= -_NO_EFFECT_EPSILON:
        return (
            "rejected",
            f"measured adverse effect (signed net={signed_net:+.6g} over {observations} run(s))",
            observations,
            raw_net,
            signed_net,
            last_card,
        )
    return (
        "confirmed",
        f"measured non-zero effect (signed net={signed_net:+.6g} over {observations} run(s))",
        observations,
        raw_net,
        signed_net,
        last_card,
    )


def derive_all_technique_statuses(
    store: KnowledgeStore,
    promoter: ClaimPromoter,
    *,
    cards: list[Any] | None = None,
    selected: set[str] | None = None,
    session_times: list[str] | None = None,
    dormant_after: int = DORMANT_AFTER_CAMPAIGNS,
) -> list[TechniqueStatusDerivation]:
    """Derive status for every row in ``techniques``."""
    card_list = cards if cards is not None else _list_cards(promoter)
    out: list[TechniqueStatusDerivation] = []
    for row in store.list_techniques():
        name = str(row.get("name") or "")
        tid = str(row.get("id") or "")
        status, reason, obs, raw_net, signed_net, card_id = derive_technique_status(
            name,
            promoter,
            cards=card_list,
            selected=selected,
            created_at=str(row.get("created_at") or "") or None,
            session_times=session_times,
            dormant_after=dormant_after,
        )
        out.append(
            TechniqueStatusDerivation(
                technique_id=tid,
                name=name,
                status=status,
                reason=reason,
                observations=obs,
                net_effect=raw_net,
                signed_net=signed_net,
                evidence_card_id=card_id,
            )
        )
    return out


def recompute_technique_status(knowledge_dir: Path, competition: str) -> list[str]:
    """Recompute every technique status from current cards. Returns changed ids."""
    changed: list[str] = []
    promoter = ClaimPromoter(Path(knowledge_dir), competition)
    try:
        cards = _list_cards(promoter)
        selected = selected_technique_names(Path(knowledge_dir), competition)
        sessions = campaign_created_ats(Path(knowledge_dir), competition)
        with KnowledgeStore(Path(knowledge_dir), competition) as store:
            for row in store.list_techniques():
                tid = str(row.get("id") or "")
                name = str(row.get("name") or "")
                if not tid or not name:
                    continue
                try:
                    status, reason, obs, _raw, signed_net, card_id = derive_technique_status(
                        name,
                        promoter,
                        cards=cards,
                        selected=selected,
                        created_at=str(row.get("created_at") or "") or None,
                        session_times=sessions,
                    )
                    prior = str(row.get("status") or "candidate")
                    if prior == status:
                        continue
                    store.set_technique_status(
                        tid,
                        status,
                        competition=competition,
                        from_status=prior,
                        reason=reason,
                        evidence_card_id=card_id,
                        observations=obs,
                        signed_net=signed_net,
                    )
                    changed.append(tid)
                    logger.info(
                        "Technique %s (%s): %s → %s (%s)",
                        name,
                        tid,
                        prior,
                        status,
                        reason,
                    )
                except Exception as exc:  # noqa: BLE001 — one bad row must not abort
                    logger.warning(
                        "technique status recompute failed for %s (%s): %s",
                        name,
                        tid,
                        exc,
                    )
    finally:
        promoter.close()
    return changed


def technique_status_report(
    knowledge_dir: Path,
    competition: str,
) -> dict[str, Any]:
    """Summarise derived statuses without writing — for review before filtering."""
    promoter = ClaimPromoter(Path(knowledge_dir), competition)
    try:
        cards = _list_cards(promoter)
        selected = selected_technique_names(Path(knowledge_dir), competition)
        sessions = campaign_created_ats(Path(knowledge_dir), competition)
        with KnowledgeStore(Path(knowledge_dir), competition) as store:
            derived = derive_all_technique_statuses(
                store,
                promoter,
                cards=cards,
                selected=selected,
                session_times=sessions,
            )
            stored = {
                str(r["id"]): str(r.get("status") or "candidate")
                for r in store.list_techniques()
            }
    finally:
        promoter.close()

    by_status: dict[str, list[dict[str, Any]]] = {
        s: [] for s in ("candidate", "confirmed", "rejected", "dormant")
    }
    would_change: list[dict[str, Any]] = []
    for item in derived:
        row = {
            "id": item.technique_id,
            "name": item.name,
            "status": item.status,
            "stored_status": stored.get(item.technique_id, "candidate"),
            "reason": item.reason,
            "observations": item.observations,
            "net_effect": item.net_effect,
            "signed_net": item.signed_net,
            "evidence_card_id": item.evidence_card_id,
        }
        by_status.setdefault(item.status, []).append(row)
        if row["stored_status"] != item.status:
            would_change.append(row)

    return {
        "competition": competition,
        "total": len(derived),
        "counts": {status: len(rows) for status, rows in by_status.items()},
        "by_status": by_status,
        "would_change": would_change,
    }


def format_technique_status_report(report: dict[str, Any]) -> str:
    """Plain-text report for CLI output."""
    lines = [
        f"Technique vocabulary — {report.get('competition', '')}",
        f"Total: {report.get('total', 0)}",
    ]
    counts = report.get("counts") or {}
    lines.append(
        "Derived counts: "
        + ", ".join(f"{status}={counts.get(status, 0)}" for status in sorted(counts))
    )
    would = report.get("would_change") or []
    lines.append(f"Would change stored status: {len(would)}")
    for row in sorted(would, key=lambda r: str(r.get("name") or ""))[:40]:
        lines.append(
            f"  {row.get('name')}: {row.get('stored_status')} → {row.get('status')} "
            f"(obs={row.get('observations')}, signed={row.get('signed_net'):+.4g}) "
            f"— {row.get('reason')}"
        )
    if len(would) > 40:
        lines.append(f"  … +{len(would) - 40} more")
    for status in ("confirmed", "rejected", "dormant"):
        rows = (report.get("by_status") or {}).get(status) or []
        if not rows:
            continue
        lines.append(f"\n{status.upper()} ({len(rows)}):")
        for row in sorted(rows, key=lambda r: str(r.get("name") or ""))[:15]:
            lines.append(
                f"  {row.get('name')} (obs={row.get('observations')}, "
                f"signed={row.get('signed_net'):+.4g})"
            )
        if len(rows) > 15:
            lines.append(f"  … +{len(rows) - 15} more")
    return "\n".join(lines)
