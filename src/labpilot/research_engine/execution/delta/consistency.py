"""Did the delta do what the hypothesis claimed?

A delta that applies cleanly is not a delta that tested the hypothesis. Take
*"ensemble LightGBM with CatBoost"* — three plausible outcomes, and all three
run, produce a score, and write an evidence card:

===========================================  ==========================================
what the delta actually did                  what the card claims
===========================================  ==========================================
*replaced* LightGBM with CatBoost            "ensembling improved MSE" — it measured
                                             **substitution**
added CatBoost, never averaged               "ensembling" — nothing was ensembled
added CatBoost **and** retuned LightGBM      the whole ``cv_gain`` credited to
                                             "ensemble" — **two changes, one
                                             attribution**
===========================================  ==========================================

The third is the dangerous one. ``technique_attribution`` assigns the full gain
to the named technique, so a delta that did more than it claimed makes that
credit **false** — and invisibly, because the number itself is real. Same class
as the inverted metric direction and the placeholder cards: a plausible
measurement of the wrong thing.

Whole-file regeneration had this problem too and hid it better. Expressing an
experiment as a delta makes it checkable for the first time, because the change
is a first-class object.

**Only labpilot can do this.** A coding agent knows whether an edit *applied*;
only the system holding the hypothesis knows what the experiment *claimed*.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: A delta touching more than this many functions is *flagged*, not refused.
#: Chosen to catch "a second change rode along" without rejecting a real
#: refactor — which is why the outcome is a note on the card, not a rejection.
_WIDE_DELTA_FUNCTIONS = 5


@dataclass
class ConsistencyReport:
    """What the delta did, measured against what the hypothesis claimed."""

    ok: bool = True
    #: Failures that should be re-asked, with the reason named — the mechanism
    #: that took prose-reply failures from three-in-eight-steps to 30 of 30.
    violations: list[str] = field(default_factory=list)
    #: Recorded on the evidence card, never a reason to refuse.
    flags: list[str] = field(default_factory=list)
    touched_functions: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict[str, object]:
        return {
            "consistent": self.ok,
            "violations": list(self.violations),
            "flags": list(self.flags),
            "touched_functions": list(self.touched_functions),
        }


def _parse(source: str, label: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        logger.warning("cannot analyse %s: %s", label, exc)
        return None


def called_names(tree: ast.Module) -> set[str]:
    """Every name that appears as a call, including attribute calls.

    ``lgb.LGBMRegressor(...)`` yields both ``lgb.LGBMRegressor`` and
    ``LGBMRegressor``, so a check can ask about a library or a specific
    constructor without knowing how the file happens to import things.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            found.add(target.id)
        elif isinstance(target, ast.Attribute):
            found.add(target.attr)
            value = target.value
            if isinstance(value, ast.Name):
                found.add(f"{value.id}.{target.attr}")
    return found


def imported_modules(tree: ast.Module) -> set[str]:
    """Module names *and their aliases*.

    Both, because a hypothesis names a library the way a person would — "keep
    LightGBM" — while the code says ``import lightgbm as lgb`` and then only
    ever writes ``lgb``. Collecting the module name alone made every check fail
    on correct code; collecting the alias alone would miss ``import catboost``
    with no ``as``. The imported symbols of a ``from`` import are included for
    the same reason: ``from lightgbm import LGBMRegressor`` should satisfy a
    claim about either name.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
                found.add(alias.name)
                if alias.asname:
                    found.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[0])
                found.add(node.module)
            for alias in node.names:
                found.add(alias.name)
                if alias.asname:
                    found.add(alias.asname)
    return found


def _function_bodies(tree: ast.Module) -> dict[str, str]:
    return {
        node.name: ast.dump(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def touched_functions(parent: ast.Module, child: ast.Module) -> list[str]:
    """Functions whose body differs, plus those added or removed.

    Compared on the dumped AST rather than the text, so reformatting and comment
    changes do not register as a change to behaviour.
    """
    before, after = _function_bodies(parent), _function_bodies(child)
    changed = {name for name in before.keys() & after.keys() if before[name] != after[name]}
    return sorted(changed | (before.keys() ^ after.keys()))


def check_preservation(child: ast.Module, keep: list[str]) -> list[str]:
    """Things the hypothesis says to *keep* must still be called.

    Catches substitution disguised as addition: "ensemble LightGBM with
    CatBoost" satisfied by deleting the LightGBM path measures a different
    experiment than the one that was proposed.
    """
    present = called_names(child) | imported_modules(child)
    return [
        f"{name!r} should have been kept, but nothing in the result calls or imports it"
        for name in keep
        if name and name not in present
    ]


def check_addition(child: ast.Module, add: list[str]) -> list[str]:
    """The named new thing must actually appear.

    Catches a no-op delta that claims a technique — the card would credit a
    technique the code never used.
    """
    present = called_names(child) | imported_modules(child)
    return [
        f"{name!r} was supposed to be added, but the result never calls or imports it"
        for name in add
        if name and name not in present
    ]


def check_combination(child: ast.Module, combine: list[str]) -> list[str]:
    """For an ensemble claim, every named component must reach the output.

    "Added but unused" is the quietest failure here: the constructor is present,
    so `check_addition` passes, but the predictions are discarded and the score
    reflects the parent alone. Approximated by requiring each component to be
    called *and* some aggregation (`mean`, `average`, `sum`, …) to appear —
    without that, nothing combines them.
    """
    if len(combine) < 2:
        return []
    problems = check_addition(child, combine)
    aggregators = {"mean", "average", "nanmean", "sum", "vstack", "stack", "concatenate"}
    if not (called_names(child) & aggregators):
        problems.append(
            f"claimed to combine {combine}, but the result contains no aggregation "
            "(mean/average/sum/stack) — the components are computed and discarded"
        )
    return problems


def check_confinement(touched: list[str], limit: int = _WIDE_DELTA_FUNCTIONS) -> list[str]:
    """A wide delta is *flagged*, never refused.

    A second, uncredited change riding along is what makes attribution false —
    but a legitimate refactor also touches many functions, so blocking would
    reject real work. This lands on the evidence card the way `needs_review`
    does: a reader can discount a wide delta rather than being denied it.
    """
    if len(touched) <= limit:
        return []
    return [
        f"delta touches {len(touched)} functions ({', '.join(touched[:6])}…); "
        "attribution credits one technique with the whole gain, so a second "
        "change riding along would be credited to the first"
    ]


def check_delta_consistency(
    parent_source: str,
    child_source: str,
    *,
    keep: list[str] | None = None,
    add: list[str] | None = None,
    combine: list[str] | None = None,
) -> ConsistencyReport:
    """Compare what the delta did against what the hypothesis claimed.

    ``keep`` / ``add`` / ``combine`` come from the hypothesis, not from reading
    the diff — the point is to test the code against an independent claim. A
    hypothesis that says nothing checkable yields an empty report rather than a
    fabricated verdict.
    """
    report = ConsistencyReport()
    child_tree = _parse(child_source, "the proposed result")
    if child_tree is None:
        report.ok = False
        report.violations.append("the result does not parse as Python")
        return report

    parent_tree = _parse(parent_source, "the parent") if parent_source else None
    if parent_tree is not None:
        report.touched_functions = touched_functions(parent_tree, child_tree)
        report.flags.extend(check_confinement(report.touched_functions))

    report.violations.extend(check_preservation(child_tree, keep or []))
    report.violations.extend(check_addition(child_tree, add or []))
    report.violations.extend(check_combination(child_tree, combine or []))
    report.ok = not report.violations
    return report
