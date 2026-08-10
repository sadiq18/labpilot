"""M20 exit criterion 1: a guard ships with the failure it rejects.

Fifteen defects on 2026-08-08, **eight of them one shape** — a gate that tests
something easier than it promises, and passes. Every one read as correct. Four of
the nine defects in the 2026-08-07 log were guards that could never fire, and
each had been read and approved.

`AGENTS.md` has recorded the countermeasure since then — *feed a guard a real bad
record before trusting it* — and three of the eight gates above were written
**after** it. Advice does not hold. So this file enumerates the capabilities that
report pass/fail and requires each to have a test proving it can say **no**.

The link is a marker, `@pytest.mark.rejects("<capability>")`, rather than a
naming convention or a grep for `passed is False`. A grep finds tests that
mention failure; a marker is a claim someone had to make on purpose, and it names
which gate the claim is about.

The bar the marker asserts is **red-then-green**: a test that passes with and
without the fix has proven nothing. That part cannot be automated, and the
marker's docstring is where the author says they did it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_CAPABILITIES = Path("src/labpilot/research_engine/execution/capabilities")
_TESTS = Path("tests")

#: Capabilities that have no rejection test yet, with the reason each is still
#: here. Every entry is a gate nobody has proven can fail — which is exactly the
#: state M20 exists to leave. The list only shrinks; adding to it fails the test
#: below that guards its length.
_UNPROVEN: dict[str, str] = {}


#: Capabilities whose every `passed=` is a literal `True` — no path to a failing
#: verdict at all. Sharper than "untested": a test cannot prove rejection of
#: something the code is unable to do.
#:
#: **Empty, and it is meant to stay that way.** It held five on 2026-08-09, the
#: day M20 started, and all five were fixed rather than excused:
#:
#: * `runtime` substituted the local default for a runtime it could not resolve
#:   and reported "selected runtime local" — a campaign that asked for a GPU
#:   trained elsewhere, with a passing card;
#: * `workspace` reported `passed=True` while its own metadata said
#:   `download_skipped: no_kaggle_config`. *Skipped because asked to* and
#:   *skipped because unable* were both `None`, and the verdict read
#:   anything-but-False as done;
#: * `submission` wrote `id,prediction\n0,0` and then reported
#:   `passed=packaged.is_file()` — a verdict about a file it had just
#:   fabricated;
#: * `reporting` returned `passed=True` from four sites, each of which ends by
#:   writing a file, so the verdict answered "did I write something" while the
#:   step promised "this execution was reported on";
#: * `stub` always passed, which is what a stub is for — so it now declares
#:   `verifies = False` instead, and its card says nothing was checked.
_CANNOT_FAIL: dict[str, str] = {}


def _capability_names() -> dict[str, Path]:
    """Capabilities that *claim to verify*, by their registered name.

    A capability declaring `verifies = False` is excluded from both checks
    below. That is not a loophole — it is M20's other option, taken in the open:
    the verdict says the step ran, the card says nothing was checked, and a
    reviewer can see the claim being declined.
    """
    found: dict[str, Path] = {}
    for path in sorted(_CAPABILITIES.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "passed=" not in source or re.search(r"^\s{4}verifies\s*=\s*False", source, re.M):
            continue
        match = re.search(r'^\s{4}name\s*=\s*"([a-z_]+)"', source, re.M)
        if match:
            found[match.group(1)] = path
    return found


def test_declining_to_verify_is_declared_not_implied():
    """A capability that opts out has to say so on the class, where a reviewer
    reads it — and its evidence has to say so too, or the card is
    indistinguishable from one that checked something."""
    from labpilot.research_engine.execution.capabilities.stub import StubCapability

    assert StubCapability.verifies is False
    source = Path("src/labpilot/research_engine/execution/capabilities/stub.py").read_text()
    assert "stub_no_verification" in source


def _marked_capabilities() -> dict[str, list[str]]:
    """`@pytest.mark.rejects("x")` across the suite, by capability name."""
    marked: dict[str, list[str]] = {}
    for path in sorted(_TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                target = decorator.func
                if not (isinstance(target, ast.Attribute) and target.attr == "rejects"):
                    continue
                for argument in decorator.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        marked.setdefault(argument.value, []).append(f"{path}::{node.name}")
    return marked


def _capabilities_marked() -> set[str]:
    """Capability names, from markers in either form.

    `@pytest.mark.rejects("reporting:reflect")` names a verdict site and also
    covers the module for the coarser check below.
    """
    return {name.split(":", 1)[0] for name in _marked_capabilities()}


def test_every_capability_that_reports_a_verdict_can_be_shown_to_fail():
    """The criterion itself. A capability with no rejection test is a gate
    nobody has proven can say no — and on 2026-08-08 that described eight."""
    capabilities = _capability_names()
    marked = _capabilities_marked()

    assert capabilities, "no capabilities found — has the layout moved?"
    # `_CANNOT_FAIL` is excluded because no test can prove rejection of
    # something the code is unable to do — those are answered by
    # `test_no_capability_reports_a_verdict_it_cannot_withhold` instead.
    missing = sorted(set(capabilities) - marked - set(_UNPROVEN) - set(_CANNOT_FAIL))

    assert not missing, (
        "these report pass/fail with no test proving they can reject a real bad "
        f'artifact: {missing}. Write one, marked `@pytest.mark.rejects("<name>")`, '
        "and confirm it is red before the fix and green after."
    )


def test_the_unproven_list_only_shrinks():
    """A list that grows once per incident is the curated-set pattern wearing
    verification's clothes — `15-gates-must-fail.md` names it as a trap. This
    one is allowed to exist only while it is being emptied."""
    assert len(_UNPROVEN) <= 0, (
        f"{len(_UNPROVEN)} capabilities are still unproven: {sorted(_UNPROVEN)}. "
        "Lower this number as they are covered; never raise it."
    )


def test_no_capability_reports_a_verdict_it_cannot_withhold():
    """The milestone's title, asked of the code rather than of its tests.

    A capability whose every `passed=` is a literal `True` is a gate that cannot
    fail — and unlike an untested gate, no test can fix it, because rejection is
    not something the code is able to do.

    **This is per-file and syntactic, and that is a real limit**, named here
    rather than left for the next reviewer to find: one falsifiable verdict
    exempts its siblings in the same module, and an expression like
    `not result.get("skipped")` counts as "can fail" no matter whether any input
    reaches it. Both loopholes were used: `reporting` has four verdicts, and two
    of them — REFLECT and UPDATE_BELIEF, the two in every baseline plan — could
    not fail while the file passed this check. Reported on PR #120.

    `test_every_verdict_site_is_named_by_a_test` below closes the granularity
    half by keying on the verdict's `checks=` label rather than the module.
    """
    unable = {}
    for name, path in _capability_names().items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        verdicts = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "passed"
        ]
        if verdicts and all(
            isinstance(value, ast.Constant) and value.value is True for value in verdicts
        ):
            unable[name] = len(verdicts)

    unexpected = sorted(set(unable) - set(_CANNOT_FAIL))
    fixed = sorted(set(_CANNOT_FAIL) - set(unable))

    assert not unexpected, (
        f"these capabilities report a verdict they can never withhold: {unexpected}. "
        "Either give them a failing path or stop claiming they verified anything."
    )
    assert not fixed, f"these can fail now — remove them from _CANNOT_FAIL: {fixed}"


#: Labels inside a literal list that this resolver cannot read, collected as it
#: goes. Each is an element it dropped, and dropping one silently is the shape of
#: every blind spot this file has had.
#:
#: `code_engineering` has the only one: `checks=["write_code", "apply", origin,
#: "override"]`, where `origin` is `"llm"` / `"aider"` / `"last_resort"` — the
#: provenance of the proposal, not a gate. Recorded rather than exempted, and
#: `test_no_verdict_label_is_silently_dropped` holds the count so a *new* one
#: has to be looked at.
_UNREADABLE_LABELS: list[str] = []

#: Verdicts whose `checks=` could not be resolved at all, as `file:line — why`.
_UNRESOLVED_VERDICTS: list[str] = []


class _Unresolvable(Exception):
    """A verdict whose `checks=` this resolver cannot read.

    The reason every earlier blind spot was silent: unresolvable and absent were
    the same answer, so each new syntax quietly shrank the requirement instead of
    failing. `checks=checks` removed the whole of `workspace`; a literal holding
    a non-string element would lose that gate; a verdict outside a function would
    vanish. Now the resolver says so and `test_every_verdict_resolves_to_a_site`
    fails with the file and line.
    """


def _literal_strings(node: ast.expr) -> set[str]:
    """String labels in a list literal, or a concatenation of them.

    `ast.BinOp` is handled because `research_review` writes
    `["train_exists", "syntax"] + (["llm_note"] if llm_notes else [])` — three
    labels sitting in plain sight, which the first version discarded by falling
    through to the variable branch and naming the *function* instead. Two real
    gates, one site, one marker covering both: the per-module granularity this
    file exists to replace, re-created by a `+`. Reported reviewing PR #121.
    """
    if isinstance(node, ast.List | ast.Tuple):
        found: set[str] = set()
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                found.add(element.value)
            else:
                # A name, a call, a starred expression. Recorded, though it no
                # longer changes the answer: a multi-label call is named by its
                # function, so an unreadable element cannot move a site or lose
                # one. Kept because a *single*-label call whose one label is
                # unreadable would otherwise fall through quietly. Reported
                # reviewing PR #121, where `origin` turned out to be live.
                _UNREADABLE_LABELS.append(ast.unparse(element))
        return found
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_strings(node.left) | _literal_strings(node.right)
    if isinstance(node, ast.IfExp):
        return _literal_strings(node.body) | _literal_strings(node.orelse)
    raise _Unresolvable(f"cannot read {ast.unparse(node)!r}")


def _stamps_no_verification(name: str, scope: ast.AST) -> bool:
    """Does this function stamp `name` **unconditionally**?

    The declaration is invisible on a dynamic verdict otherwise: `_checks_at`
    returns the *function* name, which never contains `no_verification`, so a
    capability building its list conditionally could not declare that a branch
    verifies nothing however it behaved at run time. Reported reviewing PR #121.

    **Unconditionally**, because a *conditional* stamp means the verdict
    sometimes checks and sometimes does not — and a gate that can fail must
    still be shown to. `evaluation._compare` appends it inside
    `if not had_control:`: a baseline compared nothing, a run with a control
    that produced no gain is a real failure, and exempting the site from a
    rejection test on the strength of the first would lose the second. So only
    an append at the function's own top level counts, which is what a branch
    that *only* reports a state looks like.
    """
    body = getattr(scope, "body", [])
    for statement in body:
        for node in ast.walk(statement) if not isinstance(statement, ast.If) else ():
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == name
                and any(
                    isinstance(a, ast.Constant) and a.value == "no_verification" for a in node.args
                )
            ):
                return True
    return False


def _checks_at(call: ast.Call, scope: ast.AST | None) -> set[str] | None:
    """The site name of one verdict call, or None if it has no `checks`.

    **One verdict call is one site.** Named by its label when it carries exactly
    one, and by its enclosing function otherwise. Both halves were learned the
    hard way:

    * naming by *label* alone missed every capability that builds its list
      conditionally — the whole of `workspace` had never been enumerated, and
      `evaluation`'s compare verdict left the moment a stamp had to be appended
      to it. Silently, because a coverage test cannot miss what it cannot see;
    * naming *per label* over-counted in the other direction. `workspace` threads
      one list through `_ensure_data` and `_ensure_profile` and ends with twelve
      labels on a single `passed=` expression; `research_review` writes
      `["train_exists", "syntax"] + (["llm_note"] if llm_notes else [])` on one.
      Those are annotations on one gate, and demanding a rejection test each
      would be busywork that invites gaming.

    So the labels answer *how many gates is this*, not *what are they called*.
    A capability like `reporting`, which answers four questions from four calls
    with one label each, still gets four sites — which is the granularity this
    file exists for. Reported reviewing PR #121 across three rounds.

    `no_verification` is not a label; it rides along and is returned so the
    caller can see the declaration.
    """
    argument = {k.arg: k.value for k in call.keywords}.get("checks")
    if argument is None:
        return None
    if isinstance(argument, ast.Name):
        function = getattr(scope, "name", None)
        if not function:
            raise _Unresolvable("a verdict outside any function")
        site = {function}
        if _stamps_no_verification(argument.id, scope):
            site.add("no_verification")
        return site
    labels = _literal_strings(argument)
    stamped = "no_verification" in labels
    gates = labels - {"no_verification"}
    if len(gates) == 1:
        site = set(gates)
    else:
        function = getattr(scope, "name", None)
        if not function:
            raise _Unresolvable("a multi-label verdict outside any function")
        site = {function}
    if stamped:
        site.add("no_verification")
    return site


def _enclosing_functions(tree: ast.Module) -> dict[int, ast.AST]:
    owner: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for inner in ast.walk(node):
                owner.setdefault(id(inner), node)
    return owner


def _declared_non_verifying() -> dict[str, set[str]]:
    """Sites that stamp `no_verification` on their own evidence.

    M20's second option, per verdict rather than per capability. Some branches
    genuinely check nothing — *"no requirements file; skipped install"*, *"no
    unit tests; skipped"*, *"runtime job already active"*. Their `passed=True`
    is honest about the step and dishonest about the *card*, where it reads
    identically to a gate that looked and found nothing wrong.

    A rejection test cannot be written for them, so they say so instead, in the
    `checks` list, where a reader sees it. Read from the source rather than
    listed in this file: a curated list in a test is the pattern this milestone
    refuses, and it would drift from the code the moment a branch changed.
    """
    declared: dict[str, set[str]] = {}
    for name, path in _capability_names().items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            try:
                names = _checks_at(node, owner.get(id(node)))
            except _Unresolvable as why:
                _UNRESOLVED_VERDICTS.append(f"{path}:{node.lineno} — {why}")
                continue
            if not names or "no_verification" not in names:
                continue
            labels = names - {"no_verification"}
            if len(labels) != 1:
                # A stamp exempts **one** gate. The first version exempted every
                # label beside it, so stamping a two-label site — `evaluation`
                # already has `checks=["compare", "evidence_card"]` — would drop
                # both from the coverage requirement with nothing proving either
                # can reject. That is the *"one marker stands for four gates"*
                # defect this file was rewritten to fix, on the exemption path
                # instead of the marker path. Reported reviewing PR #121.
                #
                # Left unexempted rather than guessed at: the enumerator then
                # demands a test, which is the safe direction.
                continue
            declared.setdefault(name, set()).update(labels)
    return declared


def _verdict_sites() -> dict[str, set[str]]:
    """Every verdict, by capability and check label.

    A capability is not one gate. `reporting` answers four different questions —
    report, reflect, update belief, propose a hypothesis — and each has its own
    `checks=` label, its own inputs and its own way of being wrong. Counting the
    module as covered because *one* of them had a rejection test is how two
    unfailable gates shipped inside the change written to remove them.
    """
    sites: dict[str, set[str]] = {}
    for name, path in _capability_names().items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            if "passed" not in keywords:
                continue
            try:
                labels = _checks_at(node, owner.get(id(node)))
            except _Unresolvable as why:
                # Recorded, not skipped in silence — `test_every_verdict_resolves`
                # names the file and line so a form the resolver does not
                # understand fails loudly instead of shrinking the requirement.
                _UNRESOLVED_VERDICTS.append(f"{path}:{node.lineno} — {why}")
                continue
            if not labels:
                continue
            # The stamp is the declaration, not a gate.
            sites.setdefault(name, set()).update(labels - {"no_verification"})
    return sites


_DECLARED_NON_VERIFYING = _declared_non_verifying()


def test_every_verdict_resolves_to_a_site():
    """No verdict may be skipped because this resolver could not read it.

    The single change that closes four separate findings. Every blind spot this
    file has had was the same one — **unresolvable and absent were the same
    answer** — so each new syntax quietly shrank the requirement rather than
    failing: `checks=checks` removed the whole of `workspace`, a `+` collapsed
    two `research_review` gates into one, and a verdict outside a function would
    vanish entirely. Reported reviewing PR #121 across three rounds.
    """
    assert not _UNRESOLVED_VERDICTS, (
        "verdicts whose `checks=` could not be read, so they are not enumerated: "
        f"{_UNRESOLVED_VERDICTS}. Teach `_checks_at` the form, or write the labels "
        "as literals."
    )


def test_no_verdict_label_is_silently_dropped():
    """One element inside a literal list, dropped without a signal, is a gate
    leaving the requirement unnoticed — the same defect as the whole capability
    disappearing, one element down.

    Both are `code_engineering`'s `origin` — `checks=["write_code", "apply",
    origin, "override"]`, holding `"llm"` / `"aider"` / `"last_resort"`, which is
    provenance rather than a gate. Held at the count so a *new* one has to be
    looked at rather than discovered three rounds later.
    """
    assert len(_UNREADABLE_LABELS) <= 2, (
        f"labels this resolver cannot read: {_UNREADABLE_LABELS}. If it names a "
        "gate, write it as a literal; if it is provenance, say so here."
    )


#: Verdict sites with no rejection test yet, as `capability:check`.
#:
#: **Empty.** It held twenty when the enumerator was first keyed on the verdict
#: site rather than the module — twenty gates nobody had shown could say no,
#: hidden because one marker per capability had been counted as covering all of
#: them. Eight turned out to check nothing at all and now declare it on their own
#: evidence; the rest got a rejection test, each verified red-then-green.
#:
#: The count was wrong twice more before it settled. Discovery read only literal
#: `checks=[...]` lists, so `workspace` — which threads its list through helpers
#: — had never been enumerated at all, and `evaluation:compare` left the moment
#: a stamp had to be appended to it. Both silently: a coverage test cannot miss
#: what it cannot see. Reported reviewing PR #121.
#:
#: It only shrinks.
_UNPROVEN_SITES: set[str] = set()


def test_every_verdict_site_is_named_by_a_test():
    """The granularity fix. `@pytest.mark.rejects("capability:check")` names one
    verdict; the bare `"capability"` form stays valid for single-gate modules.

    Reported on PR #120: keying on the module let one marker stand for four
    gates, and two of the four could not fail at all.
    """
    marked = set(_marked_capabilities())
    sites = _verdict_sites()
    # The bare `"capability"` form covers a module with **one** verdict, where
    # there is no ambiguity about which gate it names. It does not cover a module
    # with four. Written this way after the first version accepted the bare form
    # everywhere and passed while proving nothing — the same shape as the gates
    # it enumerates, in the enumerator.
    missing = sorted(
        f"{capability}:{check}"
        for capability, checks in sites.items()
        for check in checks
        if f"{capability}:{check}" not in marked
        and not (len(checks) == 1 and capability in marked)
        and check not in _DECLARED_NON_VERIFYING.get(capability, set())
    )

    unexpected = sorted(set(missing) - _UNPROVEN_SITES)
    assert not unexpected, (
        f"verdict sites with nothing proving they can reject: {unexpected}. "
        'Mark a test `@pytest.mark.rejects("capability:check")`.'
    )


def test_a_marker_names_a_capability_that_exists():
    """A marker pointing at a renamed or deleted capability silently stops
    proving anything, which is the failure mode of the thing it enforces."""
    capabilities = _capability_names()

    orphans = {
        name: sites
        for name, sites in _marked_capabilities().items()
        if name.split(":", 1)[0] not in capabilities
    }

    assert not orphans, f"`rejects` markers naming no capability: {orphans}"
