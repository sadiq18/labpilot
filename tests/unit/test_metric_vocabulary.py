"""One vocabulary, matched exactly, with direction owned by the metric.

Two defects measured on disk 2026-08-13 motivate every test here:

* `playground-series-s6e7/competition.json` carries
  `{"name": "balanced_accuracy_score", "key": "accuracy"}` — a real competition
  whose metric is silently mis-mapped to a neighbour.
* `metric_names_match("cv_mse", "mean_squared_error")` was False, so a campaign
  targeting the competition's own stated metric could never fire its stop.

The substring matching that produced the second is why matching here is exact:
`"mse" in "rmse"` is True, and the hint tuple this replaces survived that only by
listing `rmse` first.
"""

from __future__ import annotations

import numpy as np
import pytest

from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
    _METRICS,
    MEASUREMENT_PREFIXES,
    bare_probe_keys,
    cv_probe_keys,
    cv_search_order,
    direction_of,
    is_scorable,
    known_keys,
    maximize_from_spec,
    metrics_for,
    metrics_for_problem_type,
    normalize_direction,
    normalize_metric_key,
    scorable_keys,
)
from labpilot.research_engine.intelligence.competition.models import ProblemType

# --- the two live defects ---------------------------------------------------


def test_the_competitions_own_metric_string_resolves() -> None:
    """`mean_squared_error` matched nothing, so an `mse` target could never fire."""
    assert normalize_metric_key("mean_squared_error") == "mse"
    assert normalize_metric_key("Mean Squared Error") == "mse"


def test_balanced_accuracy_is_not_accuracy() -> None:
    """The live mis-map. Naming it correctly is the fix; scoring it is separate."""
    assert normalize_metric_key("balanced_accuracy_score") == "balanced_accuracy"
    assert normalize_metric_key("Balanced Accuracy Score") == "balanced_accuracy"


# --- exact matching, and the substring trap it exists to avoid ---------------


@pytest.mark.parametrize(
    ("left", "right"),
    [("mse", "rmse"), ("rmse", "rmsle"), ("mse", "rmsle"), ("auc", "accuracy")],
)
def test_distinct_metrics_never_collapse(left: str, right: str) -> None:
    """Substring matching made these interchangeable in one direction or another."""
    assert normalize_metric_key(left) != normalize_metric_key(right)


@pytest.mark.parametrize("raw", ["RMSE", "rmse", "  Rmse  ", "root-mean-squared-error"])
def test_spelling_and_separators_do_not_matter(raw: str) -> None:
    assert normalize_metric_key(raw) == "rmse"


@pytest.mark.parametrize("prefix", MEASUREMENT_PREFIXES)
def test_a_measurement_qualifier_is_stripped(prefix: str) -> None:
    """`cv_rmse` and `lb_rmse` are one metric read in two places."""
    assert normalize_metric_key(f"{prefix}rmse") == "rmse"


def test_only_one_qualifier_is_stripped() -> None:
    """Stripping repeatedly would turn a column named `test_score` into `score`."""
    assert normalize_metric_key("cv_cv_rmse") is None


# --- unknown is an answer ---------------------------------------------------


@pytest.mark.parametrize("raw", ["wrmsse", "quadratic_weighted_kappa", "", "  ", None])
def test_an_unknown_metric_resolves_to_none(raw: str | None) -> None:
    """m5's WRMSSE is real and unmapped. None must propagate, never a default."""
    assert normalize_metric_key(raw) is None


def test_direction_is_never_guessed() -> None:
    """A guessed direction inverts every conclusion drawn from it — see
    `direction.py`, written after all fifteen rogii cards were built as though
    MSE were maximised."""
    assert direction_of("wrmsse") is None
    assert direction_of(None) is None


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("rmse", "minimize"),
        ("mse", "minimize"),
        ("mae", "minimize"),
        ("rmsle", "minimize"),
        ("logloss", "minimize"),
        ("accuracy", "maximize"),
        ("balanced_accuracy", "maximize"),
        ("auc", "maximize"),
        ("f1", "maximize"),
    ],
)
def test_direction_belongs_to_the_metric(key: str, expected: str) -> None:
    assert direction_of(key) == expected
    # and it is reachable through a measured key, not only the bare one
    assert direction_of(f"cv_{key}") == expected


# --- the coupling that keeps the table honest -------------------------------


def _score(key: str) -> float:
    """Call the real `compute_metric` with inputs valid for this metric."""
    from labpilot.research_engine.execution.metrics import compute_metric

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.6, 0.8])
    if key in {"rmse", "mse", "mae", "rmsle"}:
        return compute_metric(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 2.9]), key)
    return compute_metric(y_true, y_pred, key, y_proba=y_proba)


@pytest.mark.parametrize("key", sorted(scorable_keys()))
def test_every_scorable_key_can_actually_be_scored(key: str) -> None:
    """The registry must not promise a number the pipeline cannot produce.

    `metrics_for_problem_type` feeds the baseline selector, which tells codegen
    which `cv_<key>` to emit — and the generated `train.py` calls `compute_metric`,
    which raises on anything outside its dispatch. A metric marked scorable here
    and absent there is a `ValueError` inside model-written code.
    """
    assert isinstance(_score(key), float)


def test_a_named_but_unscorable_metric_is_declared_as_such() -> None:
    """`balanced_accuracy` has no branch in `compute_metric`. Naming it correctly
    is still strictly better than mis-mapping it to `accuracy`."""
    from labpilot.research_engine.execution.metrics import compute_metric

    assert is_scorable("balanced_accuracy") is False
    assert "balanced_accuracy" in known_keys()
    assert "balanced_accuracy" not in scorable_keys()
    with pytest.raises(ValueError, match="Unsupported metric"):
        compute_metric(np.array([0, 1]), np.array([0, 1]), "balanced_accuracy")


# --- structural invariants --------------------------------------------------


def test_no_alias_is_claimed_by_two_metrics() -> None:
    """An ambiguous alias makes resolution depend on table order — the exact
    fragility five rounds of PR #117 kept returning to."""
    seen: dict[str, str] = {}
    for metric in _METRICS:
        for alias in metric.aliases:
            assert alias not in seen, f"{alias!r} claimed by {seen.get(alias)} and {metric.key}"
            seen[alias] = metric.key


def test_every_key_is_its_own_alias() -> None:
    """So a canonical key always round-trips."""
    for metric in _METRICS:
        assert metric.key in metric.aliases
        assert normalize_metric_key(metric.key) == metric.key


def test_no_alias_needs_normalising() -> None:
    """Aliases are stored already-slugged, or a lookup silently misses."""
    for metric in _METRICS:
        for alias in metric.aliases:
            assert alias == alias.lower()
            assert " " not in alias and "-" not in alias


def test_cv_priorities_are_unique() -> None:
    """Ties would make the search order depend on declaration order."""
    priorities = [m.cv_priority for m in _METRICS]
    assert len(priorities) == len(set(priorities))


# --- behaviour preserved for the sets this replaces -------------------------


@pytest.mark.parametrize(
    ("problem_type", "expected"),
    [
        (ProblemType.TABULAR_CLASSIFICATION, {"accuracy", "auc", "logloss", "f1"}),
        (ProblemType.TEXT_CLASSIFICATION, {"accuracy", "auc", "logloss", "f1"}),
        (ProblemType.IMAGE_CLASSIFICATION, {"accuracy", "auc", "logloss", "f1"}),
        (ProblemType.TABULAR_REGRESSION, {"rmse", "mse", "mae", "rmsle"}),
    ],
)
def test_problem_type_sets_match_the_hardcoded_ones(problem_type, expected) -> None:
    """Derived, not restated — and identical to `SUPPORTED_METRICS_BY_PROBLEM_TYPE`
    today, so adopting the registry changes no behaviour. `balanced_accuracy` is
    deliberately absent: it is named but not scorable."""
    assert set(metrics_for_problem_type(problem_type)) == expected
    assert set(metrics_for_problem_type(problem_type.value)) == expected


def test_scorable_keys_match_the_canonical_set_being_replaced() -> None:
    assert set(scorable_keys()) == {
        "accuracy",
        "auc",
        "logloss",
        "f1",
        "rmse",
        "mse",
        "mae",
        "rmsle",
    }


def test_cv_search_order_preserves_the_existing_priority() -> None:
    """`_primary_cv_keyed` probes balanced_accuracy before accuracy, accuracy
    before roc_auc, and roc_auc before rmse. Moving the list must not reorder it."""
    order = cv_search_order()
    assert order.index("balanced_accuracy") < order.index("accuracy")
    assert order.index("accuracy") < order.index("auc")
    assert order.index("auc") < order.index("rmse")


# --- applicability is derived from shape, not from a task label -------------


def test_no_metric_entry_names_a_task() -> None:
    """The scaling property, asserted structurally.

    `problem_types` on every metric made adding a task type an O(metrics x tasks)
    edit, against a `ProblemType` closed at five values — no ranking, detection,
    segmentation, forecasting, audio or RL. A metric declares what it needs of
    the data; nothing declares which task it belongs to.
    """
    for metric in _METRICS:
        assert not hasattr(metric, "problem_types")
        assert metric.target_kind in {"continuous", "discrete", "any"}


def test_a_new_kind_of_problem_needs_no_registry_edit() -> None:
    """A forecasting objective has no `ProblemType` member and never will need
    one here: its truth is continuous, so the continuous metrics apply."""
    assert metrics_for(target_kind="continuous") == {"rmse", "mse", "mae", "rmsle"}
    assert metrics_for(target_kind="discrete") == {"accuracy", "auc", "logloss", "f1"}


def test_a_group_metric_is_offered_only_when_there_are_groups(monkeypatch) -> None:
    """Ranking and retrieval score per query, not per row, so such a metric must
    not be offered for a dataset with no groups to score over.

    Injected into the real registry and read back through `metrics_for`, because
    asserting the predicate by hand would test my arithmetic rather than the
    filter.
    """
    from dataclasses import replace

    from labpilot.research_engine.intelligence.competition import metric_vocabulary as mv

    ndcg = replace(
        mv._METRICS[1], key="ndcg", target_kind="discrete", requires_groups=True
    )
    monkeypatch.setattr(mv, "_METRICS", (*mv._METRICS, ndcg))

    assert "ndcg" not in mv.metrics_for(target_kind="discrete")
    assert "ndcg" in mv.metrics_for(target_kind="discrete", has_groups=True)
    # a per-row metric stays available either way
    assert "accuracy" in mv.metrics_for(target_kind="discrete", has_groups=True)


def test_an_unrecognised_problem_type_yields_nothing_rather_than_guessing() -> None:
    """The selector then falls back to its own default and says so, instead of
    this pretending to know what an unmapped task needs."""
    assert metrics_for_problem_type("audio_classification") == frozenset()
    assert metrics_for_problem_type("unknown") == frozenset()


# --- A3: the registry is the only metric list -------------------------------


def test_no_literal_metric_list_survives_outside_this_module() -> None:
    """The A3 exit criterion, asserted structurally rather than by memory.

    Five modules each kept their own spelling of "the metrics we know", and they
    disagreed: the parser's substring hints mapped `mean_squared_error` to
    nothing the selector would accept, and `budgets` carried a second copy of the
    measurement prefixes. A sixth copy is one edit away at any time, so the
    check has to be automatic.
    """
    import pathlib
    import re

    known = {"accuracy", "auc", "logloss", "f1", "rmse", "mse", "mae", "rmsle"}
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "labpilot"
    this_module = "metric_vocabulary.py"
    literal = re.compile(r"[{(\[]\s*((?:\"[a-z0-9_]+\"\s*,\s*){2,}\"[a-z0-9_]+\")\s*,?\s*[})\]]")

    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == this_module:
            continue
        for match in literal.finditer(path.read_text(encoding="utf-8")):
            names = set(re.findall(r'"([a-z0-9_]+)"', match.group(1)))
            # Every member a canonical key. `objective.py`'s morphological hints
            # ("error", "loss", "gain", "precision") overlap this vocabulary
            # without being a metric list — they orient a metric *no* catalogue
            # knows, which is the case the registry cannot serve.
            if len(names & known) >= 3 and names <= known:
                offenders.append(f"{path.relative_to(root)}: {sorted(names)}")

    assert not offenders, "metric list outside the registry:\n" + "\n".join(offenders)


def test_a_name_and_its_own_abbreviation_resolve_to_one_metric() -> None:
    """`Root Mean Squared Error (RMSE)` is how Kaggle actually spells it — and
    exact alias matching alone could not read it, which broke house-prices."""
    assert normalize_metric_key("Root Mean Squared Error (RMSE)") == "rmse"
    assert normalize_metric_key("Area Under the ROC Curve (AUC)") == "auc"


def test_a_gloss_that_contradicts_its_name_resolves_to_nothing() -> None:
    """Preferring the left half because it is written first is the positional
    tie-break this module exists to remove."""
    assert normalize_metric_key("Accuracy (RMSE)") is None


def test_a_composite_metric_is_not_read_as_the_one_it_contains() -> None:
    """MCRMSE is column-wise RMSE and is a *different* number. The substring
    matcher this replaced answered `rmse` here, which is the whole bug class:
    `"mse" in "rmse"` is also True, and the old table survived only by listing
    `rmse` first."""
    assert normalize_metric_key("Mean Columnwise RMSE (MCRMSE)") is None
    assert normalize_metric_key("mean_columnwise_rmse") is None


def test_the_probe_order_covers_the_spellings_runs_actually_write() -> None:
    """`cv_roc_auc` is what the classification template writes; `cv_auc` is what
    the canonical key would suggest. Both must be found."""
    keys = cv_probe_keys()

    assert "cv_roc_auc" in keys and "cv_auc" in keys
    assert keys.index("cv_balanced_accuracy") < keys.index("cv_accuracy") < keys.index("cv_rmse")


def test_cross_validated_and_bare_spellings_are_offered_separately() -> None:
    """Review finding. Returning both from one function put every bare
    catalogued key ahead of the caller's generic `cv_*` sweep, so a single-split
    `mae` displaced the `cv_wrmsse` a campaign was actually judged on. The split
    is what lets the caller sequence them around that sweep.
    """
    assert all(k.startswith("cv_") for k in cv_probe_keys())
    assert not any(k.startswith("cv_") for k in bare_probe_keys())
    assert len(cv_probe_keys()) == len(bare_probe_keys())
    assert bare_probe_keys().index("accuracy") < bare_probe_keys().index("rmse")


# --- one spelling of "which way is better" ---------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("min", "minimize"),
        ("Minimize", "minimize"),
        ("MIN", "minimize"),
        ("minimise", "minimize"),
        ("max", "maximize"),
        ("Maximize", "maximize"),
        ("sideways", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_every_spelling_direction_py_reads_is_accepted(raw, expected) -> None:
    """Review finding. `direction.py` has always read any `min*`/`max*` prefix,
    so a `Literal` that rejected them made the model stricter than the reader —
    and a hand-written `configs/competitions/<slug>.yaml` saying `direction: min`
    raised out of `CompetitionParser`."""
    assert normalize_direction(raw) == expected


def test_an_unreadable_direction_becomes_unknown_rather_than_an_error() -> None:
    """Never a guess, never a crash: a misspelled direction is exactly the case
    where the registry and the probe should get to answer instead."""
    assert normalize_direction("lower_is_better") == "unknown"


def test_the_registry_outranks_a_contract_that_contradicts_it() -> None:
    from labpilot.research_engine.intelligence.competition.models import MetricSpec

    assert maximize_from_spec(MetricSpec(name="rmse", key="rmse", direction="maximize")) is False
    assert maximize_from_spec(MetricSpec(name="auc", key="auc", direction="minimize")) is True


def test_a_stated_direction_decides_a_key_the_registry_cannot_know() -> None:
    from labpilot.research_engine.intelligence.competition.models import MetricSpec

    unknown_key = MetricSpec(name="wellbore misfit", key="wellbore_misfit", direction="minimize")
    assert maximize_from_spec(unknown_key) is False


def test_nothing_knowable_is_none_rather_than_maximize() -> None:
    """`None` is what lets the caller keep its own default and say so. Answering
    `True` here is the silent-maximize this whole layer exists to remove."""
    from labpilot.research_engine.intelligence.competition.models import MetricSpec

    assert maximize_from_spec(None) is None
    assert maximize_from_spec(MetricSpec(name="wellbore misfit", key="wellbore_misfit")) is None
