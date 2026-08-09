"""Is this hypothesis already implemented? — asked before the experiment runs.

Measured on rogii 2026-08-09: four campaigns with `codegen.strategy: delta`
produced **no delta experiment**, and the editor was right to refuse every time:

    The code currently: 1 Trains a LightGBM model … 3 Averages predictions from
    both models … Since no modifications are required, there are no
    SEARCH/REPLACE blocks to output.

The hypothesis asked for an ensemble `train.py` already had. Nothing marked it
as implemented, so it stayed `proposed` and was selected again on the next step,
and the next.

**This is the same question `check_addition` already answers, asked earlier.**
M19 1c produced both halves: `DeltaBriefAgent` emits `added` as *code
identifiers*, and `consistency.py` parses the parent's AST. If every symbol the
hypothesis promises to introduce is already called or imported, there is nothing
to implement — deterministically, for free, and with no judgement to be wrong
about.

Doing it with a model instead would be slower, cost a call, and put a plausible
answer where a certain one is available. The LLM critic belongs on the question
mechanism *cannot* answer — is this promising? — not this one.

### Why `added` and not `kept` or `combined`

`added` is the only list that means "this must not be there yet". `kept` names
what must survive, so its symbols are *expected* in the parent; treating their
presence as redundancy would call every well-formed hypothesis redundant.
`combined` is about how outputs are blended, which the parent may do already
with different components.

So a hypothesis that claims nothing new is not judged redundant here — it is
judged unverifiable, which `delta_unchecked` already records.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from labpilot.research_engine.execution.delta.consistency import (
    called_names,
    imported_modules,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedundancyVerdict:
    """Whether the parent already does what the hypothesis proposes."""

    redundant: bool = False
    #: Symbols the hypothesis promised to add that the parent already uses.
    already_present: list[str] = field(default_factory=list)
    #: Human-readable, and required: a verdict that retires a hypothesis has to
    #: say which symbol proved it, or the next reader cannot check the claim.
    reason: str = ""


def check_redundancy(parent_source: str, added: list[str]) -> RedundancyVerdict:
    """Redundant when *every* promised symbol is already called or imported.

    All, not any, and the distinction is the whole design. "Ensemble LightGBM
    with CatBoost" on a file that already imports `lgb` claims `added=['cb']`
    plus `kept=['lgb']`; judging on *any* present symbol would retire it because
    the parent has LightGBM — killing the experiment the hypothesis exists to
    run. Only when nothing new remains is there nothing to do.

    An empty claim is never redundant. `DeltaBriefAgent` soft-fails to an empty
    brief, and a failed metadata call must not be read as "already done" — that
    would retire good hypotheses whenever the brief model was unavailable.
    """
    names = [str(name).strip() for name in (added or []) if str(name).strip()]
    if not names:
        return RedundancyVerdict()
    if not parent_source.strip():
        return RedundancyVerdict()

    try:
        tree = ast.parse(parent_source)
    except SyntaxError as exc:
        # A parent that does not parse cannot be shown to contain anything, and
        # guessing "redundant" here would retire a hypothesis on the strength of
        # a broken file. The consistency checks own that failure.
        logger.debug("parent does not parse; not judging redundancy: %s", exc)
        return RedundancyVerdict()

    present = called_names(tree) | imported_modules(tree)
    found = [name for name in names if name in present]
    if len(found) != len(names):
        return RedundancyVerdict(already_present=found)

    listed = ", ".join(repr(name) for name in found)
    return RedundancyVerdict(
        redundant=True,
        already_present=found,
        reason=(
            f"already implemented: the parent already calls or imports {listed}, "
            "so this change has nothing left to introduce"
        ),
    )
