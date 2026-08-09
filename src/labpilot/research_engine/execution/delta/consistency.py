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


#: Named aggregations. Necessary but nowhere near sufficient — see below.
_AGGREGATORS = frozenset(
    {"mean", "average", "nanmean", "median", "sum", "vstack", "stack", "concatenate"}
)


def _has_arithmetic_blend(tree: ast.Module) -> bool:
    """Is anything combined by arithmetic rather than by a named aggregator?

    Checking only for `mean`/`stack` rejected four of five correct ensembles.
    The miss that matters is the **weighted** blend — ``0.6 * a.predict(X) +
    0.4 * b.predict(X)`` — which is the standard technique and never calls an
    aggregator. Each false violation costs a re-ask, and steps are the scarce
    resource in a campaign, so a check that fires on correct code is expensive
    in exactly the case it should be silent.

    A blend is an arithmetic expression drawing on **two or more distinct
    non-parameter names**. Parameters are excluded because ``m.predict(X) * 2``
    references two names (``m`` and ``X``) while combining nothing — counting
    ``X`` would let a scaled single model pass as an ensemble, which is the
    false negative this whole check exists to prevent.
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        params = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)} | {
            a.arg for a in (args.vararg, args.kwarg) if a
        }
        for inner in ast.walk(node):
            if not isinstance(inner, ast.BinOp):
                continue
            names = {
                sub.id
                for sub in ast.walk(inner)
                if isinstance(sub, ast.Name) and sub.id not in params
            }
            if len(names) >= 2:
                return True
    return False


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


def unreachable_functions(tree: ast.Module) -> set[str]:
    """Locally-defined functions that nothing in this module calls.

    Empty for a module with no entry point, where "nothing calls it here" says
    only that the caller lives elsewhere.
    """
    if not _has_entry_point(tree):
        return set()
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    return defined - called_names(tree)


def strip_unreachable(tree: ast.Module) -> ast.Module:
    """The module with its dead top-level functions removed.

    For asking what the code *does*, as opposed to what it contains. A failed
    experiment leaves its edit in the workspace, so the parent of the next
    experiment can carry code that has never run — and a question answered over
    the whole file would count it.
    """
    dead = unreachable_functions(tree)
    if not dead:
        return tree
    kept = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in dead)
    ]
    return ast.Module(body=kept, type_ignores=list(tree.type_ignores))


def _has_entry_point(tree: ast.Module) -> bool:
    """Does this module run something of its own when executed?

    True when a locally-defined function is called from module level — directly
    or inside the `if __name__ == "__main__":` guard, which is how every
    generated `pipeline/train.py` ends.

    Without this, `check_reachability` would condemn any module that defines
    functions for someone else to call, which is most of them.
    """
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not defined:
        return False
    for node in tree.body:
        statements = node.body if isinstance(node, ast.If) else [node]
        for statement in statements:
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id in defined
                ):
                    return True
    return False


def check_reachability(child: ast.Module, touched: list[str]) -> list[str]:
    """A delta whose every edited function is uncalled did not change anything.

    Measured on rogii 2026-08-09, on the first delta the adapter produced
    against the real pipeline: thirty-four correct lines of rolling-window
    features written into `engineer_features`, which is defined on line 45 and
    which `main()` never calls. It parsed, it applied, and it could not run.

    That one was caught by accident. `DeltaBriefAgent` had claimed
    `added=['engineer_features']` — the enclosing function rather than the
    contribution — so `check_addition` looked for a symbol nothing calls and
    reported a violation. Fix the brief to name what the change introduces, as
    it should, and the claim becomes `['rolling', 'groupby']`; both appear in
    the new code; every check passes; and the dead function goes unmentioned.

    So the guard cannot live in the claim. Whether the change is reachable is a
    fact about the file, not about what anyone said it would do, and it is the
    one question that separates "the experiment ran" from "the experiment
    produced a number".

    Conservative in three ways, because a violation costs a re-ask and a
    re-ask spent on a correct experiment is a step a campaign does not get back:

    * it asks only of files that **run themselves** — a module with no entry
      point is a library, where an uncalled function is called by someone else
      and "unreachable" is not a fact this file can establish;
    * **every** touched function must be uncalled, so a delta that edits three
      functions and leaves one helper dead still passes;
    * it asks whether the name is called *anywhere*, not whether it is reachable
      from the entry point, which would also condemn a function called only by
      another dead one. Fewer true positives, no false ones.
    """
    if not touched or not _has_entry_point(child):
        return []
    reachable = called_names(child)
    dead = [name for name in touched if name not in reachable]
    if len(dead) != len(touched):
        return []
    listed = ", ".join(repr(name) for name in sorted(dead))
    return [
        f"the delta only changed {listed}, which nothing in the result calls — "
        "the change cannot execute, so any score it produces measures the parent"
    ]


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
    if not (called_names(child) & _AGGREGATORS or _has_arithmetic_blend(child)):
        problems.append(
            f"claimed to combine {combine}, but the result contains no aggregation "
            "(no mean/stack call and no arithmetic blend) — the components are "
            "computed and discarded"
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
        # Independent of the claim, and that is the point — see
        # `check_reachability`. A better claim would have hidden the defect
        # this catches, so it cannot be derived from one.
        report.violations.extend(check_reachability(child_tree, report.touched_functions))

    report.violations.extend(check_preservation(child_tree, keep or []))
    report.violations.extend(check_addition(child_tree, add or []))
    report.violations.extend(check_combination(child_tree, combine or []))
    report.ok = not report.violations
    return report
