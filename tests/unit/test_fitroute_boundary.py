"""fitroute must stay extractable to its own package.

`docs/smart-router/DESIGN.md` §13.1 says v0.1 lives in this repo so its first
consumer can exercise it, and that extraction stays "a directory move". That
promise is worth exactly as much as the test enforcing it — "extract later"
usually fails because the code quietly grows host-shaped assumptions.
"""

from __future__ import annotations

import ast
import pkgutil
from pathlib import Path

import pytest

import fitroute

_ROOT = Path(fitroute.__file__).resolve().parent
_MODULES = sorted(m.name for m in pkgutil.iter_modules([str(_ROOT)]))


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which stays inside the package.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_package_is_not_empty():
    assert _MODULES, "no modules found — the boundary test would pass vacuously"


@pytest.mark.parametrize("module", _MODULES)
def test_fitroute_module_does_not_import_labpilot(module):
    offenders = _imported_roots(_ROOT / f"{module}.py") & {"labpilot"}
    assert not offenders, (
        f"fitroute/{module}.py imports {sorted(offenders)} — extraction would no "
        "longer be a directory move. Pass what it needs in as an argument "
        "(see CredentialResolver) rather than importing the host."
    )


def test_init_does_not_import_labpilot():
    assert "labpilot" not in _imported_roots(_ROOT / "__init__.py")


def test_fitroute_uses_only_the_standard_library_and_pydantic():
    """Keeping the dependency surface tiny is what makes it adoptable.

    The stdlib set comes from `sys.stdlib_module_names` rather than a list
    maintained here. A hand-written list fails on the first stdlib module
    fitroute happens to use next — it flagged `re` — which trains the reader to
    extend the allowlist reflexively, and that is how a real third-party import
    would eventually be waved through.
    """
    import sys

    allowed = {"fitroute", "pydantic", "__future__"}
    for module in _MODULES:
        extra = _imported_roots(_ROOT / f"{module}.py") - allowed - sys.stdlib_module_names
        assert not extra, f"fitroute/{module}.py adds dependency {sorted(extra)}"


def test_the_boundary_test_still_catches_a_real_dependency(tmp_path):
    """Guards the guard: `sys.stdlib_module_names` must not admit everything."""
    import sys

    probe = tmp_path / "probe.py"
    probe.write_text("import requests\nimport numpy as np\n", encoding="utf-8")
    extra = _imported_roots(probe) - {"fitroute", "pydantic", "__future__"} - sys.stdlib_module_names
    assert extra == {"requests", "numpy"}
