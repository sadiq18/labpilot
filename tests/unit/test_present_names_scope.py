"""`present_names` resolves scope, and this enumerates what that means.

Five review rounds on PR #117 turned on one question — when does a bare name in
the parent count as "this symbol is here?" — and each round moved a dial:

* any name at any depth  → a dead nested helper collided with an unrelated
  parameter, and a hypothesis was retired for work nothing computed;
* Load context only      → `print(rolling)` is a load, so a loop variable still
  satisfied a claim about rolling windows;
* a defined function     → same flat set, same collision;
* top-level only         → a callback defined *inside* the function that uses
  it stopped counting, which is the idiom round one was about;
* scope resolution, v1    → comprehensions shared their enclosing scope, so a
  loop variable named after a top-level function satisfied a claim about it,
  and a decorator on a nested `def` was read twice — once in the enclosing
  scope and once inside the function's own.

There is no correct setting of that dial, because the question is about scope
and a flat set of names cannot express one. `_referenced_definitions` resolves
references the way Python does: a reference matches a definition when it sits
in that definition's scope or one nested inside it.

This file is the matrix rather than another example. Each row is a scope
relationship, and every past round's trigger is one of them — so a future
change to the resolver fails here before it reaches a reviewer.
"""

from __future__ import annotations

import ast

import pytest

from labpilot.research_engine.execution.delta.consistency import present_names

_ENTRY = '\n\nif __name__ == "__main__":\n    main()\n'


def _present(body: str) -> set[str]:
    return present_names(ast.parse(body + _ENTRY))


# (label, source, name, expected) — `main` always exists so the module runs.
_CASES = [
    (
        "top-level function used at top level",
        "def helper():\n    return 1\n\n\ndef main():\n    return helper()\n",
        "helper",
        True,
    ),
    (
        "top-level function handed to a callback",
        "import pandas as pd\n\n\ndef helper(r):\n    return r\n\n\n"
        "def main():\n    return pd.DataFrame().apply(helper, axis=1)\n",
        "helper",
        True,
    ),
    (
        "nested function used inside its own scope",
        "import pandas as pd\n\n\ndef engineer(df):\n"
        "    def _row(r):\n        return r\n"
        "    return df.apply(_row, axis=1)\n\n\n"
        "def main():\n    return engineer(None)\n",
        "_row",
        True,
    ),
    (
        "nested function, name used only in an unrelated scope",
        "def owner():\n    def rolling(x):\n        return x\n    return 1\n\n\n"
        "def other(rolling, factor):\n    return rolling * factor\n\n\n"
        "def main():\n    owner()\n    other(1, 2)\n",
        "rolling",
        False,
    ),
    (
        "loop variable that is never a definition",
        "def engineer(df):\n    for rolling in range(3):\n"
        "        print(rolling)\n    return df\n\n\n"
        "def main():\n    return engineer(None)\n",
        "rolling",
        False,
    ),
    (
        "parameter that is never a definition",
        "def scale(mean, factor):\n    return mean * factor\n\n\n"
        "def main():\n    return scale(1, 2)\n",
        "mean",
        False,
    ),
    (
        "an imported symbol",
        "import lightgbm as lgb\n\n\ndef main():\n    return lgb\n",
        "lgb",
        True,
    ),
    (
        "an attribute call",
        "def main(df):\n    return df.groupby('p').rolling(5)\n",
        "rolling",
        True,
    ),
    (
        "a name that appears nowhere",
        "def main():\n    return 1\n",
        "catboost",
        False,
    ),
    (
        "comprehension target shadows a top-level def",
        "def rolling():\n    return 1\n\n\n"
        "def main():\n    return [rolling for rolling in range(3)]\n",
        "rolling",
        False,
    ),
    (
        "a comprehension can still reference the outer name",
        "def rolling(x):\n    return x\n\n\n"
        "def main():\n    return [rolling(i) for i in range(3)]\n",
        "rolling",
        True,
    ),
    (
        "a decorator on a nested def resolves in the enclosing scope",
        "def cache(fn):\n    return fn\n\n\n"
        "def main():\n    @cache\n    def inner():\n        return 1\n    return inner()\n",
        "cache",
        True,
    ),
    (
        "a decorator does not resolve against the decorated body's locals",
        "def main():\n    def inner():\n        cache = 1\n        return cache\n"
        "    return inner()\n",
        "cache",
        False,
    ),
    (
        "a default value resolves in the enclosing scope",
        "def window():\n    return 7\n\n\n"
        "def main():\n    def inner(n=window):\n        return n\n    return inner()\n",
        "window",
        True,
    ),
    (
        "a class base resolves in the enclosing scope",
        "class Base:\n    pass\n\n\n"
        "def main():\n    class Child(Base):\n        pass\n    return Child()\n",
        "Base",
        True,
    ),
    (
        "a method body does not satisfy a claim about a top-level name",
        "def blend():\n    return 1\n\n\n"
        "class Model:\n    def fit(self):\n        blend = 2\n        return blend\n\n\n"
        "def main():\n    return Model().fit()\n",
        "blend",
        False,
    ),
]


@pytest.mark.parametrize(
    ("source", "name", "expected"),
    [(source, name, expected) for _label, source, name, expected in _CASES],
    ids=[label for label, _s, _n, _e in _CASES],
)
def test_scope_matrix(source: str, name: str, expected: bool) -> None:
    assert (name in _present(source)) is expected


def test_a_definition_does_not_leak_into_a_sibling_scope():
    """The property behind half the matrix: two functions can each define the
    same private name without either satisfying a claim about the other's."""
    source = (
        "def a():\n    def shared():\n        return 1\n    return shared()\n\n\n"
        "def b(shared):\n    return shared\n\n\n"
        "def main():\n    a()\n    b(1)\n"
    )

    # `a` genuinely uses its own `shared`, so the name is present — via `a`,
    # not via `b`'s parameter. Removing `a`'s use must remove it entirely.
    assert "shared" in _present(source)

    without_use = source.replace("    return shared()", "    return 1")
    assert "shared" not in _present(without_use)
