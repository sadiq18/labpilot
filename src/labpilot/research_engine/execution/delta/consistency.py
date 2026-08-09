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
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: A delta touching more than this many functions is *flagged*, not refused.
#: Chosen to catch "a second change rode along" without rejecting a real
#: refactor — which is why the outcome is a note on the card, not a rejection.
_WIDE_DELTA_FUNCTIONS = 5


@dataclass(frozen=True)
class ValidationSignals:
    """What the workspace declared about how this competition is validated.

    §5's fifth check needs to know where the validation region *is*, and the
    design question that stopped it for three milestones was how to say so
    without a curated list of function names — the
    curated-set-answering-an-open-world-question pattern this plan has rejected
    four times, most recently as the technique→symbol map that killed 1b's
    original derivation.

    Nothing needs curating, because the workspace has already declared it.
    `derive_validation_plan` reads the dataset profile and writes a
    `ValidationPlan` into `baseline_choice.json`: the scheme it chose and the
    columns validation must never see. Those are facts about *this* dataset,
    derived from its own shape, and already load-bearing — the codegen prompt is
    built from the same values.

    So the region is declared by the workspace **and** derived from the parent:
    the workspace names the scheme, and `validation_region` finds which of the
    parent's functions run it. Neither half is a list anyone maintains.

    **Only the scheme marks the region, and that was measured, not assumed.**
    On rogii's real 7-function `train.py`:

    | signal | region |
    |---|---|
    | scheme | `main`, `partition_suffix_holdout_split` |
    | + `group_key` | 5 of 7 functions |
    | `exclude_features` | 3 of 7, all of them *correct* feature code |

    The last row is the giveaway: a function mentioning an excluded column is
    usually the function excluding it. And `group_key` is a *column* — rogii
    groups by `file_stem_entity` for rolling features as readily as for the
    split. A scheme is a *procedure*, and only validation runs one. Six flags
    out of seven functions is a flag nobody reads, which is the failure M20
    exists for, so the wider signals are carried for the leakage check and not
    used to mark the region.

    `n_splits` and `holdout_fraction` are not signals at all. They are numbers:
    `0.5` and `5` appear in code that has nothing to do with validation.
    """

    scheme: str = ""
    exclude_features: tuple[str, ...] = ()

    @classmethod
    def from_baseline_choice(cls, choice: dict | None) -> ValidationSignals:
        """Read `baseline_choice.json`'s `validation` block, tolerating anything.

        A workspace without one yields empty signals, which flag nothing — the
        right answer, because a competition whose validation plan was never
        derived has no declared region for a delta to land in.
        """
        if not isinstance(choice, dict):
            # `(choice or {}).get(...)` assumed a dict for anything truthy, so a
            # `baseline_choice.json` holding a top-level list — or a string, or a
            # number — raised `AttributeError` past a caller catching
            # `(ValueError, TypeError)` and took the whole write with it.
            # Reported on PR #119, against a docstring promising to tolerate
            # anything.
            return cls()
        plan = choice.get("validation")
        if not isinstance(plan, dict):
            return cls()
        scheme = plan.get("scheme")
        excluded = plan.get("exclude_features")
        return cls(
            # A `str()` coercion turned `true` into `"True"` and `5` into `"5"`,
            # each of which then matched real function names by coincidence —
            # `is_true_positive_rate` for the first. A malformed value should
            # yield an empty region, which flags nothing, not a lucky substring.
            # Reported on PR #119.
            scheme=scheme if isinstance(scheme, str) else "",
            exclude_features=tuple(
                column
                for column in (excluded if isinstance(excluded, list) else [])
                if isinstance(column, str) and column
            ),
        )


def _word_parts(name: str) -> list[str]:
    """`GroupKFold` and `group_kfold` both split to `["group", "k", "fold"]`.

    Underscores and camel-case humps are the same boundary written two ways —
    a scheme is spelled one way in a config and another in code, and that
    difference is spelling, not meaning.
    """
    return [part.lower() for part in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", name)]


def _matches_scheme(name: str, scheme_parts: list[str]) -> bool:
    """Does `name` contain the scheme as a run of whole words?

    Not a substring test, which was the first version and was wrong on the
    default scheme. `kfold` is a substring of `sanity_check_folds` — "che**ck
    fold**s" — so a diagnostic that counts pre-existing folds joined the
    validation region, and so did anything with an `n_kfolds` parameter or a
    `benchmark_folds` local. Reported on PR #119.

    Word runs instead: `partition_suffix_holdout_split` contains the run
    `partition|suffix|holdout`; `sanity_check_folds` contains no run equal to
    `kfold`, and `n_kfolds` splits to `n|kfolds`, whose only run is `kfolds`.
    `GroupKFold` splits to `group|k|fold` and matches `kfold` on `k|fold`.
    """
    if not scheme_parts:
        return False
    parts = _word_parts(name)
    joined = "".join(scheme_parts)
    for start in range(len(parts)):
        run = ""
        for part in parts[start:]:
            run += part
            if run == joined:
                return True
            if len(run) > len(joined):
                break
    return False


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


def referenced_names(tree: ast.Module) -> set[str]:
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

    Self- and mutual recursion used to be handled here, by dropping one
    function's own body from the scan. That answered "does anything *other than
    this* mention it", which is still not reachability — a dead pair calling
    each other passed. Both cases now belong to `unreachable_functions`, which
    walks the call graph from the entry point and gets them for free. Reported
    on PR #118.
    """
    scope: ast.AST = tree
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

    True when a **top-level** function is called from **module level** —
    directly or inside the `if __name__ == "__main__":` guard, which is how
    every generated `pipeline/train.py` ends.

    Without this, `check_reachability` would condemn any module that defines
    functions for someone else to call, which is most of them.

    Both halves of that sentence have been wrong in turn, so both are now
    stated precisely.

    *Module level* used to be approximated by walking each top-level statement,
    `ast.walk` and all — which descends into a function body, so one helper
    calling another looked like an entry point. `runs_at_import` walks only what
    actually executes when the module is imported. A class body **does**: it
    runs the moment the `class` statement does, so `class Config: seed =
    set_seed(42)` is an entry point and skipping it alongside `def` was wrong.
    Its methods are `def`s and are skipped like any other. Reported on PR #118.

    *Top-level* used to be "defined anywhere", also via `ast.walk`, so a method
    named `fit` made a module-level `fit()` look locally defined — and
    sklearn-shaped code names methods `fit`, `predict` and `train` as a matter
    of course. Only a top-level `def` can satisfy a module-level call; anything
    else would be a `NameError` if the code ran. Reported on PR #118.
    """
    defined = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    if not defined:
        return False

    def runs_at_import(body: list[ast.stmt]) -> bool:
        for statement in body:
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                # Defining is not running. The body executes only if something
                # calls it, which is the question being asked.
                continue
            if isinstance(statement, ast.ClassDef):
                # A class body is not deferred — it executes immediately.
                if runs_at_import(statement.body):
                    return True
                continue
            nested = _control_flow_bodies(statement)
            if nested is not None:
                if runs_at_import(nested):
                    return True
                continue
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id in defined
                ):
                    return True
        return False

    return runs_at_import(tree.body)


def _control_flow_bodies(statement: ast.stmt) -> list[ast.stmt] | None:
    """The statements a compound statement guards, or None if it is not one.

    Enumerated from `ast`'s own node set rather than listed by hand: the hand
    list omitted `ast.Match`, so a module-level `match` fell through to the
    unguarded `ast.walk` below and reopened the false "runs at import" this
    rewrite closed for every other construct. Reported on PR #118. Anything
    carrying a statement body is compound, whatever it is called.
    """
    if isinstance(statement, ast.Match):
        return [inner for case in statement.cases for inner in case.body]
    fields = ("body", "orelse", "finalbody")
    if not any(isinstance(getattr(statement, name, None), list) for name in fields):
        return None
    nested: list[ast.stmt] = []
    for name in fields:
        nested.extend(getattr(statement, name, []) or [])
    for handler in getattr(statement, "handlers", []) or []:
        nested.extend(handler.body)
    return nested


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

    **The dead set comes from `unreachable_functions`, not from a second
    mechanism.** This asked "is the name mentioned anywhere else" for a while,
    which is a weaker question with its own bugs — a self-call vouched for its
    own function, then a mutually recursive pair vouched for each other, each
    fixed here one at a time while the sibling walk in this same file already
    answered all of them. Reported on PR #118: `A` calls `B`, `B` calls `A`,
    `main()` calls neither, and this returned no violation. Two primitives for
    one question means every bug gets found twice, so there is now one.

    The walk is also the *stronger* answer, and deliberately so. An earlier
    version of this docstring called entry-point reachability too aggressive
    because it "would also condemn a function called only by another dead one"
    — but such a function genuinely cannot run, so that is a true positive, not
    a false one.

    Conservative in two ways, because a violation costs a re-ask and a re-ask
    spent on a correct experiment is a step a campaign does not get back:

    * it asks only of files that **run themselves** — a module with no entry
      point is a library, where an uncalled function is called by someone else
      and "unreachable" is not a fact this file can establish;
    * **every** touched function the child still defines must be dead, so a
      delta that edits three functions and leaves one helper dead still passes.
    """
    if not touched or not _has_entry_point(child):
        return []
    # Only functions the child still defines can be judged. A delta that
    # *removes* a function puts a name in `touched` that can never appear in
    # any dead-set, and "every touched function is dead" was then unsatisfiable
    # — so a delta adding dead code alongside an unrelated removal passed
    # silently. Reported on PR #118. A removal is not dead code; it is the
    # absence of code, which `check_preservation` owns.
    present = {
        node.name for node in child.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    judged = [name for name in touched if name in present]
    if not judged:
        return []
    unreachable = unreachable_functions(child)
    dead = [name for name in judged if name in unreachable]
    if len(dead) != len(judged):
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


def _identifiers_used(node: ast.AST) -> set[str]:
    """Identifiers in a subtree — no string constants, no nested definition names.

    Deliberately not `referenced_names`, which is Load-context only. That
    policy exists because being *bound* is not being *used*, and it is right
    for "did the delta's claim happen". This asks a different question — what
    does this function name at all — and a scheme assigned to a local
    (`partition_suffix_holdout = 0.7`) is exactly the inlining the region check
    looks for. Two questions, two collectors, and this comment so the next
    change to either has a reason to read the other. Reported on PR #119.

    A nested definition puts its *parent* in the region, and that is deliberate
    rather than overlooked: `def compute_metrics(): def kfold_helper(): …`
    contains validation-shaped code, and the parent is the only unit a delta
    can be said to have landed in — `touched_functions` names top-level
    functions. Raised on PR #119 as contamination; it is the same rule that
    puts a wrapper in the region, and exempting it would reopen that.
    """
    found: set[str] = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name):
            found.add(inner.id)
        elif isinstance(inner, ast.Attribute):
            found.add(inner.attr)
        elif isinstance(inner, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.add(inner.name)
    return found


def _string_constants(node: ast.AST) -> set[str]:
    """String literals in a subtree.

    Split out rather than folded into `_identifiers_used` because only the
    leakage check wants them: an excluded column is referenced as `df["ANCC"]`
    far more often than as an identifier, while a scheme named in a string is
    *reporting* which scheme ran, not running one.
    """
    return {
        inner.value
        for inner in ast.walk(node)
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
    }


def validation_region(tree: ast.Module, signals: ValidationSignals) -> set[str]:
    """Top-level functions that run the validation scheme the workspace declared.

    Derived, not listed: `signals.scheme` names the procedure and this finds
    who performs it — the function named after it (`partition_suffix_holdout_split`)
    and any function that calls or mentions it (`main`). A workspace that
    declared no scheme has an empty region, which is the honest answer: there
    is no validation plan for a delta to disturb.

    Matched on a folded name, so `group_kfold` finds `GroupKFold` — the scheme
    is written one way in a config and another in code, and that difference is
    spelling, not meaning. Substring rather than equality because the split
    function is named *after* the scheme, not *as* it.

    Module-level statements are deliberately out of scope. The question is
    which *function* a delta landed in, and `touched_functions` answers in the
    same vocabulary.

    **The entry point calls the splitter, and calls everything else too.**
    `main` landed in the region on the first measurement for no better reason
    than that, and nearly every delta would then have carried a validation flag
    — a flag on everything is a flag nobody reads. So the module's entry point
    is exempt from reaching the region by delegation alone.

    Only the entry point, and only the one the `if __name__` guard names. The
    first version exempted *every* caller, which hid a wrapper doing
    consequential work of its own: `prepare_and_split` reseeds and reshuffles
    before delegating, which changes exactly which rows land in the holdout.
    The second exempted anything called at module level, which handed the same
    escape back to a notebook-shaped script with no `main()`. Both reported on
    PR #119.

    Identifiers only — see `_identifiers_used`. Naming the scheme in a string
    is reporting, not running: rogii's `main` writes
    `{"validation_scheme": "partition_suffix_holdout"}` into its metrics, and
    counting that kept the orchestrator in the region after the delegation rule
    had already taken it out.

    The limit worth stating: a delta that *inlines* a split under names that
    resemble nothing in the plan is not caught. This finds code that performs
    the declared scheme, and code that quietly performs a different one is the
    open edge. Confinement covers part of it — such a delta is usually wide —
    and the rest waits for a signal better than naming.
    """
    scheme_parts = _word_parts(signals.scheme)
    if not scheme_parts:
        return set()
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    region = {node.name for node in functions if _matches_scheme(node.name, scheme_parts)}
    delegators = _guarded_entry_points(tree)
    for node in functions:
        if node.name in region:
            continue
        beyond = _identifiers_used(node)
        if node.name in delegators:
            # The entry point calls everything, so a call into the region says
            # nothing about it. Nothing else gets that exemption: a wrapper that
            # reseeds or reorders before delegating changes which rows land in
            # the holdout, and excluding every caller hid it. Reported on
            # PR #119.
            beyond = beyond - region
        if any(_matches_scheme(name, scheme_parts) for name in beyond):
            region.add(node.name)
    return region


def _guarded_entry_points(tree: ast.Module) -> set[str]:
    """Functions called from the `if __name__ == "__main__":` guard.

    Narrower than "called at module level", which was the first version and was
    too generous by exactly the amount that reopened the bug it shipped
    alongside: in a notebook-shaped script with no `main()`, every top-level
    call is a module-level call, so `prepare_and_split(...)` became its own
    entry point and got the delegation exemption back. Reported on PR #119.

    The guard is the one construct that means *this is how the module starts*.
    A script without one grants the exemption to nobody, which is the safe
    direction: a wrapper stays in the region rather than escaping it.
    """
    called: set[str] = set()
    for statement in tree.body:
        if not isinstance(statement, ast.If) or not _is_main_guard(statement.test):
            continue
        for inner in ast.walk(statement):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                called.add(inner.func.id)
    return called


def _is_main_guard(test: ast.expr) -> bool:
    """`__name__ == "__main__"`, however the comparison is spelled."""
    if not isinstance(test, ast.Compare):
        return False
    operands = [test.left, *test.comparators]
    names = {node.id for node in operands if isinstance(node, ast.Name)}
    literals = {node.value for node in operands if isinstance(node, ast.Constant) and node.value}
    return "__name__" in names and "__main__" in literals


def check_validation_region(
    parent: ast.Module | None,
    child: ast.Module,
    touched: list[str],
    signals: ValidationSignals,
) -> list[str]:
    """§5's fifth check — a delta landing in the validation region is *flagged*.

    §8 names this the only risk it calls *the one that would hurt*: a delta may
    change validation logic, and a leaky score looks **better**, not worse, so
    neither the metric nor the leaderboard says anything is wrong. A hypothesis
    *about* validation is legitimate; one that changes validation while
    claiming to test a feature is a false result wearing a real number.

    Flagged and never refused, which is §8's own wording — *"the mitigation is
    detection, not prohibition"*. The region is inferred from names, and every
    check in this file that refused on inferred names has had to be walked back
    (see `redundancy.py`). A flag on the evidence card costs a reader a second
    look; a refusal costs a legitimate experiment.

    Asked of the parent **and** the child. A delta that edits the existing
    holdout construction is caught by the parent's region; a delta that
    *introduces* validation logic where there was none is caught by the
    child's, and that is the shape that would otherwise be invisible — nothing
    to preserve, nothing claimed, and a new split nobody asked for.

    **Silent on a first-ever write, deliberately.** With no parent there are no
    touched functions, so this returns nothing — and it should: writing the
    validation split is what a baseline *is*, and flagging every baseline for
    defining one would flag every baseline. Raised on PR #119 as the check being
    inert on the case it exists for; it is the opposite case. A baseline is not
    a delta, and the thing that *does* need saying about a first write —
    whether it trains on columns the test set will not carry — is
    `check_leakage_discipline`, which needs no parent and runs on every write.
    """
    if not touched or not signals.scheme:
        return []
    region = validation_region(child, signals)
    if parent is not None:
        region |= validation_region(parent, signals)
    landed = sorted(name for name in touched if name in region)
    if not landed:
        return []
    listed = ", ".join(repr(name) for name in landed)
    return [
        f"the delta changed {listed}, which the workspace's validation plan "
        f"({signals.scheme or 'unnamed scheme'}) runs through — a change to how "
        "the score is computed is not the experiment the hypothesis claimed, and "
        "a leaky split scores better rather than worse"
    ]


def check_leakage_discipline(child: ast.Module, signals: ValidationSignals) -> list[str]:
    """The excluded columns must be excluded by something in the file.

    F7: columns in `validation.exclude_features` must never reach the feature
    set. They are the columns the test set does not carry — `Geology`, `ANCC`
    and five others on rogii — so a model that trains on them scores well in
    validation and cannot score at all on the leaderboard.

    This was enforced structurally until M19 §2: the Jinja pack skipped
    `column in set(EXCLUDE_FEATURES)` when deriving features. Deleting the pack
    left one bullet in `code_engineer_system.md` and no check at all — an
    instruction to a model, which is the thing this milestone keeps learning
    not to rely on.

    What is checkable without guessing: a file that derives its features from
    the frame's columns and never mentions the exclusion cannot be applying it.
    That is an implication, not a heuristic — exclusion by name requires the
    name, or the key that holds the names.

    Two questions, in order.

    **Is an excluded column selected?** `df[["Geology", "GR"]]` keeps it, and
    that is a fact, not an inference. Flagged whatever else the file says.

    **Otherwise, is anything excluding them?** A file that derives features from
    the frame's own columns and never mentions every excluded column, nor
    `exclude_features`, cannot be applying the rule. Not flagged when it names
    them, reads the key from config, or selects features by explicit allowlist —
    the last because features chosen by name never touch the frame's columns, so
    the excluded ones are absent by construction.

    What stays beyond reach, and is not claimed: a file that mentions an
    excluded column in a log line while leaking it elsewhere, or one that
    assigns `exclude_features` and never applies it. Mentioning is weak
    evidence of excluding; it is the strongest available once direction has
    been ruled on, and both remaining shapes are visible to a reader the flag
    brings in.
    """
    if not signals.exclude_features:
        return []
    excluded = set(signals.exclude_features)

    # Direction first, because it is the only half that is unambiguous. A name
    # inside a column selection is a column being *kept*: `df[["Geology",
    # "GR"]]` says so outright. The first version asked only whether the file
    # mentioned the name at all, which read explicit inclusion as evidence of
    # exclusion — the exact leak class, scored as clean. Reported on PR #119.
    selected = _selected_columns(child) & excluded
    if selected:
        listed = ", ".join(repr(column) for column in sorted(selected))
        return [
            f"the code selects {listed} as a feature, and the validation plan "
            "excludes those columns because the test set does not carry them — a "
            "model trained on them scores well in validation and cannot score on "
            "the leaderboard"
        ]

    if not _derives_from_frame(child):
        return []
    accounted = _identifiers_used(child) | _string_constants(child)
    if "exclude_features" in accounted:
        return []
    # *Every* excluded column, not any of them. An intersection test passed a
    # file that dropped one of three and kept the other two. Reported on
    # PR #119.
    unmentioned = sorted(excluded - accounted)
    if not unmentioned:
        return []
    listed = ", ".join(repr(column) for column in unmentioned[:4])
    return [
        f"the code derives its features from the frame and never mentions {listed}"
        f"{'…' if len(unmentioned) > 4 else ''} or `exclude_features`, so nothing "
        "in it excludes the columns the test set does not carry"
    ]


#: Attributes that hand back a *frame's* own column set. A file reaching for one
#: of these is deriving features from whatever the data happens to hold, which
#: is the shape leakage exclusion exists for.
#:
#: Pandas-specific names only, because this matches an attribute without knowing
#: the receiver's type. `keys` and `items` were here for one round and had to
#: go: `Counter.items()` in an unrelated word-count function made a file that
#: never touches a DataFrame read as deriving features from one — and worse, it
#: defeated the *"explicit allowlist"* exemption, so a file correctly doing
#: `df[["GR", "MD"]]` was flagged for a `.items()` call elsewhere in the module.
#: Reported on PR #119. `df.keys()` is now a miss; a false positive that fires
#: on ordinary dict code is worse than a miss, because it is the one that gets
#: the check turned off.
_FRAME_COLUMN_SOURCES = frozenset({"columns", "select_dtypes", "dtypes"})


def _derives_from_frame(tree: ast.Module) -> bool:
    """Does this file build its feature set out of the frame's own columns?

    Not detected: implicit iteration, `[c for c in df if …]`, which needs to
    know `df` is a DataFrame and this does not. Stated rather than guessed at —
    a comprehension over a bare name is far too common to treat as a signal.
    """
    return bool(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in _FRAME_COLUMN_SOURCES
        }
    )


def _selected_columns(tree: ast.Module) -> set[str]:
    """Column names a *list* subscript reads out of a frame.

    `df[["a", "b"]]` and `df.loc[:, ["a", "b"]]` both select columns, and the
    second was invisible: a `.loc` slice is a `Tuple` holding the `List`, and
    the first version only looked one level down. Reported on PR #119.

    **List slices only.** A single-string subscript is far too ambiguous to
    read as selection — `df[df["Geology"] > 0]` filters rows, and the mask's
    inner `df["Geology"]` is a `Subscript(Load)` indistinguishable from a
    feature pick once `ast.walk` has separated it from the `Compare` it belongs
    to. That produced "the code selects 'Geology' as a feature" for code that
    selects no features at all — a false positive in the exact direction this
    check was rebuilt to remove. Reported on PR #119.

    The cost is a missed `X = df["Geology"]`, and it is worth paying: a
    multi-column selection is how a feature set is written, and a check that
    cries leak at row filtering is a check that gets switched off.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Load):
            continue
        found.update(_string_elements(node.slice))
    return found


def _string_elements(node: ast.expr) -> set[str]:
    """String constants in a (possibly nested) list or tuple literal."""
    if not isinstance(node, ast.List | ast.Tuple):
        return set()
    found: set[str] = set()
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            found.add(element.value)
        else:
            found |= _string_elements(element)
    return found


def check_delta_consistency(
    parent_source: str,
    child_source: str,
    *,
    keep: list[str] | None = None,
    add: list[str] | None = None,
    combine: list[str] | None = None,
    validation: ValidationSignals | None = None,
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
        # Through `record`, like every other violation. Appending straight to
        # the list skipped `claim_free_violations`, so `_observe_delta` — which
        # reads that list when nothing was claimed — saw an empty one and
        # reported nothing wrong about a result that does not parse. Reported
        # on PR #118. Nothing claimed this, so it is claim-free by definition.
        report.ok = False
        report.record(["the result does not parse as Python"], needs_claim=False)
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

    # §5's fifth check, and F7 alongside it. Both are flags: they land on the
    # evidence card so a reader can discount the result, and neither refuses,
    # because both infer a region from names and refusing on inferred names is
    # what this file has had to walk back every time it tried.
    signals = validation or ValidationSignals()
    report.flags.extend(
        check_validation_region(parent_tree, child_tree, report.touched_functions, signals)
    )
    report.flags.extend(check_leakage_discipline(child_tree, signals))

    report.record(check_preservation(child_tree, keep or []))
    report.record(check_addition(child_tree, add or []))
    report.record(check_combination(child_tree, combine or []))
    report.ok = not report.violations
    return report
