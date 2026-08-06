"""Turn a plan's `technique` into something the template path can render.

Per design §8.7 this resolves *identity*, not intent: the LLM path already
receives the hypothesis triad and does not consult this module. What it
produces is the discrete recipe switches a Jinja gate can act on, plus an
honest status for anything it cannot map.

Four outcomes, and the distinctions between them are the point:

``none``            no technique on the plan — render exactly as today (N5).
``applied``         mapped to an executable recipe, preconditions satisfied.
``not_applicable``  a real technique, wrong data — e.g. target encoding with no
                    categorical columns. Recorded so reflection does not read
                    the flat result as "the technique did not help".
``candidate``       not in the executable set. **Not a rejection.** The run
                    proceeds on its description via LLM codegen; the label is
                    surfaced for review so a genuinely new method is never
                    silently discarded.
``rejected``        provably not a technique — a record reference like
                    ``hyp:H-010``. The only outcome that asserts "this is junk".

**On F7 (leakage).** This module does *not* implement the design's F7 rule.
F7 rejects a recipe whose **input columns** intersect
``validation.exclude_features``; recipes here declare no input columns, so the
check is not expressible yet. Exclusion is enforced one level down, in the
templates: `tabular_regression_partitioned` skips
``column in set(EXCLUDE_FEATURES)`` when deriving features, which is what keeps
``TVT``/``ANCC`` out on rogii. Any new gate must follow that pattern. An earlier
version of this file intersected exclude_features with *recipe names*, which
could never fire on a real column list — a guard that looked protective and was
not, exactly the class of defect this milestone keeps finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from labpilot.research_engine.execution.technique.models import TechniqueSpec
from labpilot.research_engine.execution.technique.registry import gated_recipes, get_technique
from labpilot.research_engine.shared.labels import is_record_reference

_NUMERIC_DTYPES = ("int", "float", "double", "decimal")


@dataclass(frozen=True)
class TechniqueResolution:
    """What the executor will actually do about a plan's technique."""

    requested: str = ""
    canonical: str | None = None
    status: str = "none"  # none | applied | not_applicable | candidate | rejected
    reason: str = ""
    feature_recipes: list[str] = field(default_factory=list)
    model_params: dict[str, Any] = field(default_factory=dict)
    model_family: str | None = None

    @property
    def changes_rendering(self) -> bool:
        return bool(self.feature_recipes or self.model_params or self.model_family)


def _columns(profile: dict[str, Any]) -> list[dict[str, Any]]:
    cols = profile.get("columns")
    return [c for c in cols if isinstance(c, dict)] if isinstance(cols, list) else []


def _feature_columns(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Columns a recipe could derive from — target and id are not features."""
    target = str(profile.get("target_column") or "")
    ident = str(profile.get("id_column") or "")
    return [
        c
        for c in _columns(profile)
        if str(c.get("name") or "") not in {target, ident} and not c.get("is_target")
    ]


def _has_numeric(profile: dict[str, Any]) -> bool:
    return any(
        any(tok in str(c.get("dtype") or "").lower() for tok in _NUMERIC_DTYPES)
        for c in _feature_columns(profile)
    )


def _has_categorical(profile: dict[str, Any]) -> bool:
    """Non-numeric feature columns. Deliberately conservative: a low-cardinality
    integer *could* be categorical, but guessing it is how a recipe gets applied
    to something it cannot help, which then reads as the technique failing."""
    return any(
        not any(tok in str(c.get("dtype") or "").lower() for tok in _NUMERIC_DTYPES)
        for c in _feature_columns(profile)
    )


def _is_partitioned(profile: dict[str, Any], choice: Any) -> bool:
    return bool(profile.get("partitioned") or getattr(choice, "partitioned", False))


def _precondition_met(req: str, profile: dict[str, Any], choice: Any) -> bool:
    if req == "numeric_columns":
        return _has_numeric(profile)
    if req == "categorical_columns":
        return _has_categorical(profile)
    if req == "partitioned":
        return _is_partitioned(profile, choice)
    # An unrecognised precondition must not silently pass: that would let a
    # technique through on a check nobody implemented.
    return False


def _template_name(choice: Any) -> str:
    return str(getattr(choice, "template_name", "") or "")


def requested_technique(plan_meta: dict[str, Any], hyp_fields: dict[str, Any]) -> str:
    """The technique this plan is testing, using the precedence the LLM prompt
    already uses (`capability.py`): explicit technique, then the top of the
    stack, then a combo entry."""
    for source in (plan_meta, hyp_fields):
        value = str(source.get("technique") or "").strip()
        if value:
            return value
    for source in (plan_meta, hyp_fields):
        for key in ("technique_stack", "combo_techniques"):
            items = [str(x).strip() for x in (source.get(key) or []) if str(x).strip()]
            if items:
                return items[-1]
    return ""


def prompt_technique_fields(
    resolution: TechniqueResolution,
    plan_meta: dict[str, Any],
    hyp_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What the codegen prompt should say about the technique.

    Record references are stripped from *all three* fields, not just the scalar
    one. rogii's measured case was `technique`, but `technique_stack` carried
    `["vit", "hyp:H-010"]` on the same plans — asking a model to implement a
    hypothesis ID is no better inside a list. The hypothesis triad already
    carries the real intent, so dropping a meaningless label loses nothing.
    """
    hyp_fields = hyp_fields or {}

    def _clean(key: str) -> list[str]:
        for source in (plan_meta, hyp_fields):
            items = [str(x).strip() for x in (source.get(key) or []) if str(x).strip()]
            if items:
                return [item for item in items if not is_record_reference(item)]
        return []

    scalar = None if resolution.status == "rejected" else (resolution.requested or None)
    return {
        "technique": scalar,
        "technique_stack": _clean("technique_stack"),
        "combo_techniques": _clean("combo_techniques"),
    }


def resolve_technique(
    plan_meta: dict[str, Any],
    hyp_fields: dict[str, Any] | None = None,
    *,
    choice: Any = None,
    profile: dict[str, Any] | None = None,
) -> TechniqueResolution:
    """Map a plan's technique onto renderable recipes, or say why not."""
    hyp_fields = hyp_fields or {}
    profile = profile or {}

    requested = requested_technique(plan_meta, hyp_fields)
    if not requested:
        return TechniqueResolution(status="none", reason="plan declares no technique")

    if is_record_reference(requested):
        return TechniqueResolution(
            requested=requested,
            status="rejected",
            reason=f"{requested!r} is a record reference, not a technique",
        )

    spec: TechniqueSpec | None = get_technique(requested)
    if spec is None:
        return TechniqueResolution(
            requested=requested,
            status="candidate",
            reason=(
                f"{requested!r} has no deterministic recipe; codegen implements it "
                "from the hypothesis description"
            ),
        )

    problem_type = str(getattr(choice, "problem_type", "") or "")
    if spec.applies_to and problem_type and problem_type not in spec.applies_to:
        return TechniqueResolution(
            requested=requested,
            canonical=spec.name,
            status="not_applicable",
            reason=f"{spec.name} does not apply to problem type {problem_type!r}",
        )

    unmet = [r for r in spec.requires if not _precondition_met(r, profile, choice)]
    if unmet:
        return TechniqueResolution(
            requested=requested,
            canonical=spec.name,
            status="not_applicable",
            reason=f"{spec.name} requires {', '.join(sorted(unmet))}, absent from this dataset",
        )

    # A recipe the chosen template cannot act on must not be reported as
    # applied. `lag_features` on `tabular_regression_partitioned` resolves
    # cleanly, is passed to the renderer, and the template — which has zero
    # gates — ignores it. Recording that as `applied` would put "the technique
    # ran and did nothing" into research memory, which is the false negative
    # this milestone exists to prevent, one level up.
    template = _template_name(choice)
    if spec.feature_recipes and template:
        missing = sorted(set(spec.feature_recipes) - gated_recipes(template))
        if missing:
            return TechniqueResolution(
                requested=requested,
                canonical=spec.name,
                status="not_applicable",
                reason=(
                    f"template {template!r} has no gate for {missing}; "
                    "the recipe path cannot execute this technique yet"
                ),
            )

    return TechniqueResolution(
        requested=requested,
        canonical=spec.name,
        status="applied",
        reason=f"{spec.name} resolved to {spec.feature_recipes or spec.model_family or 'params'}",
        feature_recipes=list(spec.feature_recipes),
        model_params=dict(spec.model_params),
        model_family=spec.model_family,
    )
