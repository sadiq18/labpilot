"""Re-derive competition skill overlays from the evidence cards that exist now.

`upsert_skill_overlay` appends one block per execution, keyed by `lesson_id`, and
returns early when that key is already present. That is correct while the verdict
behind it is correct, and permanent when it is not: repairing a card afterwards
does not rewrite the lesson it already produced, and the key guarantees nothing
ever will.

Measured on rogii 2026-08-08 — every overlay under `.labpilot/skills/` carried

    - Avoid: SWA
    - Avoid: regression on E-026

for E-026, the **only** execution that ever improved the metric (MSE
194.80 → 190.97). `repair_card_directions` had already re-oriented EV-012 to
`accepted`; the overlay kept telling six agents to avoid it, on every run, in
their system prompt. The same file said `Keep: hyp:H-010` — a hypothesis id — and
`Keep: dataset, rolling_features` from E-030, a genuine regression.

This is the third artifact of one shape: derived, written once, and with no path
back to its source. Evidence cards got `repair_card_directions`, beliefs got
`rederive_beliefs_from_cards`, plan projections got a staleness stamp. Overlays
are the one that reaches the model, so it is the one whose staleness costs the
most.

So this recomputes rather than un-does. Polarity comes from the card's *current*
decision, which makes the result a function of the cards alone: idempotent, and
correct after any card repair without needing to know what the cards used to say.

What it deliberately does not do:

* **Invent lessons.** A block whose execution has no card, or whose card is
  `inconclusive`, is dropped rather than rewritten. There is nothing to restate
  it from, and a confident lesson with no evidence behind it is the failure this
  module exists to remove.
* **Touch `Try:` or `Note:` lines.** Those are prose from the reviewer, not a
  verdict this can re-derive. They travel with their block and are dropped with
  it.
* **Re-derive prose.** `Try:` and `Note:` are the reviewer's words. A note may
  therefore still read oddly beside a flipped verdict — `evidence
  strength=strong` on a block now marked `Avoid`. Left alone deliberately: the
  measurement it reports did not change, and rewriting someone's observation to
  match a verdict is how a record stops being a record.

Record references *are* dropped here, using the shared rule from
`shared/labels.py`. An earlier version deferred that to `outcome.py`'s write
guard on the reasoning that one owner is better than two — which read well and
was wrong: this function rewrites the file, so declining to filter meant
knowingly writing `Keep: hyp:H-005` back out. The single owner is the *rule*,
which `labels.py` says explicitly is to be called wherever tags are read as
techniques; this is one of those places.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from labpilot.accessor.common.derived import strip_derived_note
from labpilot.research_engine.evidence.models import EvidenceDecision
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.shared.labels import is_record_reference
from labpilot.research_engine.shared.skills import overlay_dir, stamped_overlay

logger = logging.getLogger(__name__)

_LESSON_MARKER = re.compile(r"<!--\s*lesson:(?P<id>[^\s>]+)\s*-->")
_BULLET = re.compile(r"^-\s*(?P<label>Keep|Avoid|Try|Note):\s*(?P<value>.+)$", re.I)

#: Bullet labels this module owns. `Try`/`Note` are prose and travel with the
#: block rather than being re-derived.
_POLARITY_LABELS = ("keep", "avoid")


def _blocks(text: str) -> list[tuple[str, str]]:
    """Split an overlay into ``(lesson_id, block_text)`` in file order."""
    matches = list(_LESSON_MARKER.finditer(text))
    out: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((match.group("id"), text[match.start() : end].rstrip()))
    return out


def _polarity_for(decision: EvidenceDecision) -> str | None:
    """`Keep` for an accepted card, `Avoid` for a rejected one, else nothing."""
    if decision == EvidenceDecision.ACCEPTED:
        return "Keep"
    if decision == EvidenceDecision.REJECTED:
        return "Avoid"
    return None


def _rewrite_block(block: str, polarity: str) -> str:
    """Flip Keep/Avoid bullets to `polarity`, preserving order and prose."""
    lines: list[str] = []
    for line in block.splitlines():
        match = _BULLET.match(line.strip())
        if match and match.group("label").lower() in _POLARITY_LABELS:
            value = match.group("value").strip()
            # "regression on E-026" is a claim about the verdict, so it is only
            # true on the Avoid side. Left as-is it would read as a reason to
            # keep the thing it says regressed.
            if value.lower().startswith("regression on ") and polarity == "Keep":
                continue
            if is_record_reference(value):
                # A hypothesis id is not a method name. Rewriting the file and
                # putting it back would be this module preserving exactly what
                # `labels.py` exists to remove.
                continue
            lines.append(f"- {polarity}: {value}")
        else:
            lines.append(line)
    return "\n".join(lines)


def repair_skill_overlays(
    workspace_root: Path | str,
    knowledge_dir: Path | str,
    competition: str,
) -> list[str]:
    """Rebuild overlay lessons from current cards. Returns changed file names.

    A no-op when there are no overlays or no cards — there is then nothing to
    correct *towards*, and rewriting on a guess would be the original defect
    wearing a different hat.
    """
    root = overlay_dir(workspace_root)
    if root is None or not root.is_dir():
        return []

    try:
        cards = EvidenceCardStore(Path(knowledge_dir), competition).list()
    except Exception as exc:  # noqa: BLE001 — repair must never break a run
        logger.warning("could not read evidence cards for overlay repair: %s", exc)
        return []
    if not cards:
        return []

    # Keyed by the execution the lesson is about; `lesson_id` is the execution
    # id (`outcome.py` sets `lesson_id = summary.execution_id`).
    by_execution = {
        str(card.treatment_experiment or "").strip(): card
        for card in cards
        if str(card.treatment_experiment or "").strip()
    }

    changed: list[str] = []
    for path in sorted(root.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("could not read overlay %s: %s", path.name, exc)
            continue
        original = strip_derived_note(raw)
        # `strip_derived_note` returns its argument unchanged when there is none.
        unstamped = raw == original
        if not original.strip():
            if unstamped:
                try:
                    path.write_text(stamped_overlay(original), encoding="utf-8")
                except OSError as exc:
                    logger.warning("could not stamp overlay %s: %s", path.name, exc)
                else:
                    changed.append(path.name)
            continue

        kept: list[str] = []
        for lesson_id, block in _blocks(original):
            card = by_execution.get(lesson_id)
            if card is None:
                # No card to justify it. Includes lessons from executions that
                # never trained a model, which is how `Keep: vit` survived.
                continue
            polarity = _polarity_for(card.decision)
            if polarity is None:
                continue
            rewritten = _rewrite_block(block, polarity)
            if _has_content(rewritten):
                kept.append(rewritten)

        updated = ("\n\n".join(kept).rstrip() + "\n") if kept else ""
        # On content: stripping the note does not restore the trailing newline.
        if updated.strip() != original.strip() or unstamped:
            try:
                path.write_text(stamped_overlay(updated), encoding="utf-8")
            except OSError as exc:
                logger.warning("could not rewrite overlay %s: %s", path.name, exc)
                continue
            changed.append(path.name)
            logger.info(
                "Rebuilt %s from %d card(s): %d lesson(s) kept",
                path.name,
                len(cards),
                len(kept),
            )
    return changed


def _has_content(block: str) -> bool:
    """True when a block still carries a bullet, not just its heading."""
    return any(_BULLET.match(line.strip()) for line in block.splitlines())


def record_references_in_overlays(workspace_root: Path | str) -> list[str]:
    """Overlay files still naming a record reference. Diagnostic, not a repair.

    Separate from the rebuild because it answers a different question: the
    rebuild fixes *polarity*, and this reports whether the write-path guard in
    `outcome.py` is holding. A non-empty result on a workspace that has run
    since that guard landed means a sixth write site exists.
    """
    root = overlay_dir(workspace_root)
    if root is None or not root.is_dir():
        return []
    offenders: list[str] = []
    for path in sorted(root.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            match = _BULLET.match(line.strip())
            if match and is_record_reference(match.group("value")):
                offenders.append(path.name)
                break
    return offenders
