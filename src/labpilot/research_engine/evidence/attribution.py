"""Deterministic prior-based technique attribution (no extra runs)."""

from __future__ import annotations


def attribute_techniques(
    techniques: list[str],
    *,
    cv_gain: float,
    belief_priors: dict[str, float] | None = None,
) -> dict[str, float]:
    """Split ``cv_gain`` across techniques using belief confidence priors.

    Single technique → full credit. Empty list → {}.
    Weights proportional to max(0.05, prior_confidence); normalized to sum to cv_gain.
    """
    techs = [t.strip() for t in techniques if str(t).strip()]
    if not techs:
        return {}
    if len(techs) == 1:
        return {techs[0]: float(cv_gain)}

    priors = belief_priors or {}
    weights: list[float] = []
    for name in techs:
        conf = float(priors.get(name, 0.5))
        weights.append(max(0.05, conf))
    total_w = sum(weights) or 1.0
    return {
        name: float(cv_gain) * (w / total_w) for name, w in zip(techs, weights, strict=True)
    }
