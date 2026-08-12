"""A new consumer of metrics must not re-invent "comparable" and lose a guard.

The defect this exists to catch, twice observed:

* rogii 2026-08-07 — seven of fifteen evidence cards were built from runs that
  never trained a model, and a stub's `cv_accuracy` 0.5 was compared against a
  real run's `cv_rmse` 194.80. `evidence/builder.py::is_placeholder_metrics`
  was added to stop it.
* M11 promotion — `rank_candidates` grew its own numeric/NaN/bool filter and
  omitted the placeholder check, so a dry-run branch won its cohort. Every
  unit test passed, because every fixture wrote metrics as a bare float and
  none could express the `status` marker a real placeholder run carries.

Both are the same shape: a new caller derives its own notion of a usable score
instead of asking the one definition. These tests find such a caller rather
than listing today's, so the next one fails here instead of in review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from labpilot.research_engine.evidence.builder import PLACEHOLDER_STATUSES
from labpilot.research_engine.shared.experiments.scoring import comparable_metric_value

SRC = Path(__file__).resolve().parents[2] / "src" / "labpilot"

#: Building this view of a run's metrics means "I am about to compare them".
_COMPARISON_MARKER = "metrics_as_experiment"

#: Any one of these means the module asked the shared definition rather than
#: rolling its own.
_GUARDS = {"is_placeholder_metrics", "comparable_metric_value"}


def _modules_calling(marker: str) -> list[Path]:
    """Every module under src/ that calls `marker`, found by walking the AST."""
    found: list[Path] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == marker
            ):
                found.append(path)
                break
    return found


def test_every_module_that_compares_metrics_refuses_placeholders() -> None:
    """Discovery, not a list: a module added tomorrow is checked too."""
    callers = _modules_calling(_COMPARISON_MARKER)
    # If this is empty the test has stopped testing anything — the marker was
    # renamed and the invariant silently lapsed.
    assert callers, f"no caller of {_COMPARISON_MARKER}; the marker moved"

    unguarded = []
    for path in callers:
        source = path.read_text(encoding="utf-8")
        if not any(guard in source for guard in _GUARDS):
            unguarded.append(path.relative_to(SRC).as_posix())
    assert unguarded == [], (
        f"these compare run metrics without refusing placeholder runs: {unguarded}. "
        f"Call comparable_metric_value() instead of writing another numeric filter."
    )


@pytest.mark.parametrize("status", sorted(PLACEHOLDER_STATUSES))
def test_comparable_metric_value_refuses_every_placeholder_status(status: str) -> None:
    """Driven off the frozenset, so a marker added there is covered here.

    A guard that hardcodes one status passes a single-value test and still
    admits the other.
    """
    assert comparable_metric_value({"status": status, "cv_rmse": 0.5}, "cv_rmse") is None


def test_comparable_metric_value_accepts_an_ordinary_score() -> None:
    """The other half — a guard that refuses everything also passes the above."""
    assert comparable_metric_value({"cv_rmse": 194.80}, "cv_rmse") == 194.80


@pytest.mark.parametrize(
    "metrics",
    [
        {"cv_rmse": float("nan")},
        {"cv_rmse": float("inf")},
        {"cv_rmse": True},
        {"cv_rmse": "0.5"},
        {"cv_rmse": None},
        {},
        None,
        "not-a-dict",
    ],
)
def test_comparable_metric_value_refuses_what_cannot_be_ranked(metrics) -> None:
    assert comparable_metric_value(metrics, "cv_rmse") is None
