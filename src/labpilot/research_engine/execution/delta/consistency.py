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
import copy
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
    #: The subset of `violations` produced by checks that need no claim from
    #: the author — "did the code change" and "can it run". Tracked as a field
    #: rather than recovered by matching the sentences, because the consumer
    #: that keeps them when nothing was claimed was doing exactly that, and a
    #: copy edit to either message would have silently switched it off.
    claim_free_violations: list[str] = field(default_factory=list)

    def record(self, violations: list[str], *, needs_claim: bool = True) -> None:
        """Add violations, remembering which needed no claim."""
        self.violations.extend(violations)
        if not needs_claim:
            self.claim_free_violations.extend(violations)

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


def _referenced_definitions(tree: ast.Module) -> set[str]:
    """Names that are *defined here and used where that definition is visible*.

    Python's own rule, rather than a set membership test. A reference resolves
    to a definition when it sits in that definition's scope or in one nested
    inside it — so a helper defined and used inside one function counts, and a
    parameter that merely shares a name with some other function's private
    helper does not.

    This replaces four rounds of the same oscillation on PR #117. Matching any
    name at any depth let a nested dead helper collide with an unrelated
    parameter; restricting to top-level definitions then stopped recognising a
    callback defined inside the function that uses it — the exact idiom round
    one was about. Both are the same mistake, which is answering a question
    about *scope* with a flat list of names. There is no third setting of that
    dial that is right; the dial was the problem.
    """
    found: set[str] = set()

    def definitions_in(body: list[ast.stmt]) -> set[str]:
        return {
            node.name
            for node in body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }

    def visit(node: ast.AST, visible: frozenset[str]) -> None:
        """Walk one scope. `visible` already includes what this scope defines."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Load) and child.id in visible:
                    found.add(child.id)
                continue

            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                # Decorators, base classes and default values are evaluated
                # where the definition *sits*, not inside the scope it opens —
                # so they are visited here and the new scope's own body is
                # visited separately. Descending into `child` wholesale visited
                # them a second time with the inner scope merged in, and never
                # visited defaults in the right one at all. Reported on PR #117.
                enclosing = [*child.decorator_list, *getattr(child, "bases", [])]
                enclosing.extend(_default_values(child))
                for expression in enclosing:
                    visit(ast.Expression(body=expression), visible)
                visit_scope(child.body, visible, node=child)
                continue

            if isinstance(child, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
                # A comprehension has its own scope in Python 3 — unlike a
                # `for` loop, whose variable leaks — so its targets shadow
                # anything of the same name outside it. Reported on PR #117: a
                # never-called `def x()` was reported present because an
                # unrelated `[x for x in range(3)]` bound the same name.
                visit(child, visible - _comprehension_targets(child))
                continue

            visit(child, visible)

    def visit_scope(body: list[ast.stmt], visible: frozenset[str], *, node=None) -> None:
        # A binding shadows the enclosing scope for the *whole* scope, not from
        # the assignment onwards: `blend = 2` anywhere in a function makes every
        # `blend` in it local, and a top-level `def blend` is then unreachable
        # from inside. That is one rule, and the comprehension case above is the
        # same rule — which is why neither is a special case for a name the
        # checks happen to care about.
        shadowed = _local_bindings(body) | _parameter_names(node)
        inner = (visible - shadowed) | definitions_in(body)
        for statement in body:
            visit(ast.Module(body=[statement], type_ignores=[]), inner)

    visit_scope(tree.body, frozenset())
    return found


def _comprehension_targets(node: ast.expr) -> set[str]:
    """Names a comprehension binds in its own scope."""
    return {
        name.id
        for generator in node.generators
        for name in ast.walk(generator.target)
        if isinstance(name, ast.Name)
    }


def _parameter_names(node: ast.AST | None) -> set[str]:
    args = getattr(node, "args", None)
    if not isinstance(args, ast.arguments):
        return set()
    every = (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
    return {arg.arg for arg in every if arg is not None}


def _local_bindings(body: list[ast.stmt]) -> set[str]:
    """Names this scope binds, ignoring the scopes nested inside it.

    Nested functions, classes and comprehensions open their own scopes, so what
    they bind is theirs and not a shadow here. `global`/`nonlocal` do the
    reverse — they declare that an assignment binds elsewhere, so the name is
    not shadowed locally.
    """
    bound: set[str] = set()
    declared_elsewhere: set[str] = set()

    def collect(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                ast.FunctionDef
                | ast.AsyncFunctionDef
                | ast.ClassDef
                | ast.ListComp
                | ast.SetComp
                | ast.DictComp
                | ast.GeneratorExp,
            ):
                continue
            if isinstance(child, ast.Global | ast.Nonlocal):
                declared_elsewhere.update(child.names)
                continue
            if isinstance(child, ast.Import | ast.ImportFrom):
                bound.update(alias.asname or alias.name.split(".")[0] for alias in child.names)
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                bound.add(child.id)
            elif isinstance(child, ast.alias | ast.excepthandler) and getattr(child, "name", None):
                if isinstance(child, ast.excepthandler):
                    bound.add(child.name)
            collect(child)

    for statement in body:
        collect(ast.Module(body=[statement], type_ignores=[]))
    return bound - declared_elsewhere


def _default_values(node: ast.AST) -> list[ast.expr]:
    """A function's default argument expressions, if it has any."""
    args = getattr(node, "args", None)
    if not isinstance(args, ast.arguments):
        return []
    return [d for d in (*args.defaults, *args.kw_defaults) if d is not None]


def present_names(tree: ast.Module) -> set[str]:
    """Every symbol this module has available: called, imported, or defined-and-used.

    One definition, because keeping several in step failed twice on PR #117 —
    each round switched some consumers and left others behind. A consumer that
    calls this cannot be the one forgotten.

    A bare name counts only when it resolves to a definition *visible from
    where it is used* (see `_referenced_definitions`), which is the callback
    idiom these checks exist for without the name collisions that four rounds
    of flat-set matching produced in both directions.
    """
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    return called_names(tree) | imported_modules(tree) | attributes | _referenced_definitions(tree)


def bound_names(tree: ast.Module) -> set[str]:
    """Names bound as values, ignoring attribute access.

    `referenced_names` folds in every `Attribute.attr`, which is right for "is
    this symbol used at all" and wrong for "is this *import* used": an unrelated
    `df.time` made `import time` look alive, so a dead helper's import survived
    pruning. Reported on PR #117.
    """
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


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


def executable_signature(tree: ast.Module) -> str:
    """The module's behaviour, with everything that cannot affect it removed.

    Comments and formatting never reach the AST at all; docstrings do, so they
    are stripped here. Two files with the same signature run identically.
    """
    clone = copy.deepcopy(tree)
    for node in ast.walk(clone):
        if not isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
    return ast.dump(clone)


def check_effect(parent: ast.Module, child: ast.Module) -> list[str]:
    """A delta has to change what the code *does*.

    Measured on rogii 2026-08-09, on H-015's third and final attempt. Handed
    the LightGBM dtype error as its retry reason, aider edited the module
    docstring — *"adds Decision Tree"* to *"adds rolling features"* — and
    changed nothing else. Every existing check passed it:

    * `touched_functions` compares function bodies, and a module docstring is
      not in one, so it reported nothing touched;
    * `check_reachability` needs a touched function before it has an opinion;
    * `check_addition` passed because `rolling` and `groupby` had been added by
      an earlier attempt and were still there;
    * `aider_no_edit` never fired, because there *was* an edit.

    So an attempt that could not possibly change the result was recorded as a
    consistent delta, and it consumed the hypothesis's last attempt. The
    hypothesis was then retired as failed three times, when it had really been
    tested twice.

    This is the same failure as the dead function and the dead parent, arriving
    through the one door those two do not cover: not *unreachable* code, but
    *identical* code. All three reduce to one rule — a change that cannot alter
    behaviour is not an experiment — and each needed its own check because each
    is invisible to the others.
    """
    if executable_signature(parent) != executable_signature(child):
        return []
    return [
        "the delta changed no executable code — only comments, docstrings or "
        "formatting differ, so the result behaves exactly like its parent"
    ]


def referenced_names(tree: ast.Module, *, ignoring: str = "") -> set[str]:
    """Every name the module mentions as a value, plus attribute names.

    Mentions, not calls. A function handed to something else runs perfectly
    well without ever being the target of a `Call` node — `df.apply(helper,
    axis=1)`, `.transform(helper)`, `sorted(key=helper)`, a decorator, a
    dispatch dict. Reported on PR #117 and reproduced: a real behaviour-changing
    delta to a `df.apply` callback came back
    `the delta only changed 'helper', which nothing in the result calls`, and
    row-wise feature engineering is precisely the idiom this system's own
    codegen writes.

    **Load context only.** `ast.Name` covers assignment as well as use, and
    `called_names` — which these checks used before — could only ever see a
    `Call.func`, so the imprecision was invisible until they switched. Reported
    on PR #117: `for rolling in range(3)` binds `rolling`, which made a
    hypothesis proposing a real rolling-window feature read as already
    implemented, and an unrelated `for mean in [...]` made a delta that
    discards both predictions pass as an ensemble. Being bound is not being
    used.

    A `Name` node covers direct calls too, since `helper()` puts one in
    `Call.func`. Being mentioned without running is a far cheaper mistake here
    than a false violation, which costs a re-ask on a correct experiment.

    `ignoring` drops one function's own body from the scan, so a function that
    only calls itself does not vouch for its own reachability. Reported on PR
    #118: `def helper(): return helper()`, never wired into `main()`, counted
    its own recursive call and escaped the check built to catch exactly that.
    """
    scope: ast.AST = tree
    if ignoring:
        scope = ast.Module(
            body=[
                node
                for node in tree.body
                if not (
                    isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    and node.name == ignoring
                )
            ],
            type_ignores=list(tree.type_ignores),
        )
    found = {
        node.id
        for node in ast.walk(scope)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    found |= {node.attr for node in ast.walk(scope) if isinstance(node, ast.Attribute)}
    return found


def unreachable_functions(tree: ast.Module) -> set[str]:
    """Top-level functions nothing running can reach.

    Reachability *walked from the entry point*, not "is the name mentioned".
    Mentions were the first answer and each case they missed was reported on
    PR #117: a self-call vouched for its own function, a mutually recursive
    pair vouched for each other, and `defined` collected functions at *any*
    nesting depth while the caller could only remove top-level ones — so a dead
    helper nested inside a live function made `strip_unreachable`'s fixpoint
    loop spin forever. That is a hard hang on the path that runs before every
    non-retry aider call.

    So: seed with what module-level code references, then follow each reachable
    function into what *it* references. A cycle nothing outside it enters is
    never seeded and stays dead at any depth, and only top-level definitions are
    ever named, so the caller can always act on the answer.

    Reported again on PR #118 from the other side: excluding one function's
    own body let a mutually recursive pair vouch for each other, which the
    walk answers without a special case.

    Nested functions are deliberately out of scope: they live and die with the
    function enclosing them, which this already judges.

    Empty for a module with no entry point, where "nothing reaches it here"
    says only that the caller lives elsewhere.
    """
    if not _has_entry_point(tree):
        return set()
    defined = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    if not defined:
        return set()

    # Decorators execute at import, whether or not the function they wrap is
    # ever called, so they seed reachability alongside module-level statements.
    # Reported on PR #117.
    seed_body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in defined:
            seed_body.extend(ast.Expr(value=d) for d in node.decorator_list)
            continue
        seed_body.append(node)
    seeds = ast.Module(body=seed_body, type_ignores=[])
    reachable = {name for name in defined if name in referenced_names(seeds)}
    frontier = list(reachable)
    while frontier:
        node = defined[frontier.pop()]
        # The whole node, not just its body: default arguments and decorators
        # reference names too, and a reachable function's decorator is itself
        # reachable.
        for name in referenced_names(ast.Module(body=[node], type_ignores=[])):
            if name in defined and name not in reachable:
                reachable.add(name)
                frontier.append(name)
    return set(defined) - reachable


def strip_unreachable(tree: ast.Module) -> ast.Module:
    """The module with its dead top-level functions removed.

    For asking what the code *does*, as opposed to what it contains. A failed
    experiment leaves its edit in the workspace, so the parent of the next
    experiment can carry code that has never run — and a question answered over
    the whole file would count it.
    """
    if not _has_entry_point(tree):
        # A module that runs nothing of its own cannot establish that anything
        # in it is dead — neither a function nor the import it uses. Same
        # precondition as `unreachable_functions`, applied to both halves so a
        # library's unused import is not pruned on a guess.
        return tree

    live = tree
    # To a fixpoint. One pass removes only the *leaves* of a dead chain: an
    # unreachable wrapper still references the helper it calls, so the helper
    # looks live until the wrapper is gone. Reported on PR #117 — a two-level
    # chain kept `rolling` and `groupby` "present", which is the rogii false
    # retirement one indirection deeper.
    while True:
        dead = unreachable_functions(live)
        if not dead:
            break
        live = ast.Module(
            body=[
                node
                for node in live.body
                if not (
                    isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in dead
                )
            ],
            type_ignores=list(live.type_ignores),
        )
    return _without_unused_imports(live)


def _without_unused_imports(tree: ast.Module) -> ast.Module:
    """Drop imports nothing remaining references.

    Stripping dead function *bodies* left their imports behind, and
    `imported_modules` counts an import whether or not anything uses it — so
    `import catboost as cb`, used only inside a function `main()` never calls,
    still answered "the parent already has catboost". The import half of the
    same dead-code question. Reported on PR #117.
    """
    used = bound_names(tree)
    kept: list[ast.stmt] = []
    changed = False
    for node in tree.body:
        if not isinstance(node, ast.Import | ast.ImportFrom):
            kept.append(node)
            continue
        live_aliases = [
            alias for alias in node.names if (alias.asname or alias.name.split(".")[0]) in used
        ]
        if not live_aliases:
            changed = True
            continue
        if len(live_aliases) != len(node.names):
            changed = True
        replacement = copy.copy(node)
        replacement.names = live_aliases
        kept.append(replacement)
    # Compared by content, not by statement count. `import pandas as pd, numpy
    # as np` with only `pd` used trims to one alias but still yields one
    # statement, so a count check called it unchanged and threw the trimmed
    # version away — `numpy` survived and a hypothesis adding it was judged
    # already implemented. Reported on PR #117.
    if not changed:
        return tree
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
    dead = [name for name in touched if name not in referenced_names(child, ignoring=name)]
    if len(dead) != len(touched):
        return []
    listed = ", ".join(repr(name) for name in sorted(dead))
    return [
        f"the delta only changed {listed}, which nothing in the result mentions — "
        "the change cannot execute, so any score it produces measures the parent"
    ]


def check_preservation(child: ast.Module, keep: list[str]) -> list[str]:
    """Things the hypothesis says to *keep* must still be called.

    Catches substitution disguised as addition: "ensemble LightGBM with
    CatBoost" satisfied by deleting the LightGBM path measures a different
    experiment than the one that was proposed.
    """
    present = present_names(child)
    return [
        f"{name!r} should have been kept, but nothing in the result references it"
        for name in keep
        if name and name not in present
    ]


def check_addition(child: ast.Module, add: list[str]) -> list[str]:
    """The named new thing must actually appear.

    Catches a no-op delta that claims a technique — the card would credit a
    technique the code never used.
    """
    # `referenced_names`, not `called_names`, for the reason `check_reachability`
    # already uses it: a function handed to `df.apply` runs without ever being a
    # `Call` target. Reported on PR #117 and left half-fixed — reachability was
    # switched and these three were not, so the *normal* path still failed:
    # `DeltaBriefAgent` always supplies an `add` claim, and a correctly wired
    # callback came back "never calls or imports it".
    present = present_names(child)
    return [
        f"{name!r} was supposed to be added, but the result never references it"
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
    # `present_names`, not `called_names`: `preds.apply(np.mean, axis=1)`
    # genuinely averages and never calls `mean` directly, so the aggregator
    # scan had the same callback blind spot the other checks did. Reported on
    # PR #117.
    if not (present_names(child) & _AGGREGATORS or _has_arithmetic_blend(child)):
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
        # First, because every other verdict is about *how* the code changed
        # and this asks whether it changed at all. A docstring-only delta
        # otherwise passes each of them on its own terms.
        report.record(check_effect(parent_tree, child_tree), needs_claim=False)
        # Independent of the claim, and that is the point — see
        # `check_reachability`. A better claim would have hidden the defect
        # this catches, so it cannot be derived from one.
        report.record(check_reachability(child_tree, report.touched_functions), needs_claim=False)

    report.record(check_preservation(child_tree, keep or []))
    report.record(check_addition(child_tree, add or []))
    report.record(check_combination(child_tree, combine or []))
    report.ok = not report.violations
    return report
