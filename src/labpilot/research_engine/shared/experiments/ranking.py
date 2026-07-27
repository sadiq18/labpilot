"""Deterministic hypothesis ranking (Milestone 2, Plan 6).

Scores the proposed-hypothesis backlog — no LLM, no auto-execution.
"""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.intelligence.competition.models import CompetitionSpec
from labpilot.research_engine.shared.experiments.graph import ExperimentGraph, build_graph
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.knowledge import KnowledgeBase, normalize_technique
from labpilot.research_engine.shared.experiments.models import (
    Experiment,
    Hypothesis,
    HypothesisStatus,
    KnowledgeEffect,
    RankedCandidate,
)


class RankingWeights:
    def __init__(
        self,
        *,
        expected_gain: float = 2.0,
        implementation_cost: float = 0.5,
        gpu_cost: float = 0.5,
        risk: float = 1.0,
        novelty: float = 0.5,
    ) -> None:
        self.expected_gain = expected_gain
        self.implementation_cost = implementation_cost
        self.gpu_cost = gpu_cost
        self.risk = risk
        self.novelty = novelty


def resolve_primary_metric_key(runs_dir: Path, competition: str, graph: ExperimentGraph) -> str | None:
    """Pick primary metric key from competition.json (+ cv_ prefix), else first shared key."""
    for run_id in graph.nodes:
        path = runs_dir / run_id / "competition.json"
        if not path.is_file():
            continue
        try:
            spec = CompetitionSpec.model_validate_json(path.read_text())
        except (OSError, ValueError):
            continue
        if spec.evaluation_metric is None or not spec.evaluation_metric.key:
            continue
        key = spec.evaluation_metric.key
        cv_key = f"cv_{key}"
        for candidate in (cv_key, key):
            if any(candidate in exp.metrics for exp in graph.nodes.values()):
                return candidate
        return cv_key
    # Fall back: most common metric key across experiments
    counts: dict[str, int] = {}
    for exp in graph.nodes.values():
        for metric in exp.metrics:
            counts[metric] = counts.get(metric, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def rank_candidates(
    competition: str,
    runs_dir: Path,
    knowledge_dir: Path,
    *,
    weights: RankingWeights | None = None,
    default_expected_gain: float = 0.0,
    cheap_tags: set[str] | None = None,
    risk_kb_bonus: float = 0.25,
) -> list[RankedCandidate]:
    """Rank proposed hypotheses for a competition. Empty list if none proposed."""
    weights = weights or RankingWeights()
    cheap = {normalize_technique(tag) for tag in (cheap_tags or _DEFAULT_CHEAP_TAGS)}

    store = HypothesisStore(knowledge_dir, competition)
    hypotheses = store.list(status=HypothesisStatus.PROPOSED)
    if not hypotheses:
        return []

    graph = build_graph(runs_dir, competition, knowledge_dir=knowledge_dir)
    kb = KnowledgeBase(knowledge_dir, competition)
    metric_key = resolve_primary_metric_key(runs_dir, competition, graph) or "cv_accuracy"

    dimensions: list[dict[str, float]] = []
    for hypothesis in hypotheses:
        dimensions.append(
            _score_dimensions(
                hypothesis,
                graph=graph,
                kb=kb,
                metric_key=metric_key,
                cheap_tags=cheap,
                default_expected_gain=default_expected_gain,
                risk_kb_bonus=risk_kb_bonus,
            )
        )

    gains = [d["expected_gain"] for d in dimensions]
    gpus = [d["gpu_cost_seconds"] for d in dimensions]
    norm_gain = _min_max(gains)
    norm_gpu = _min_max(gpus)

    ranked: list[RankedCandidate] = []
    for hypothesis, dims, ng, ngpu in zip(hypotheses, dimensions, norm_gain, norm_gpu, strict=True):
        score = (
            weights.expected_gain * ng
            - weights.implementation_cost * dims["implementation_cost"]
            - weights.gpu_cost * ngpu
            - weights.risk * dims["risk"]
            + weights.novelty * dims["novelty"]
        )
        ranked.append(
            RankedCandidate(
                hypothesis=hypothesis,
                expected_gain=dims["expected_gain"],
                implementation_cost=dims["implementation_cost"],
                gpu_cost_seconds=dims["gpu_cost_seconds"],
                risk=dims["risk"],
                novelty=dims["novelty"],
                score=score,
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


_DEFAULT_CHEAP_TAGS = frozenset(
    {
        "hyperparameter",
        "hyperparams",
        "tune",
        "tuning",
        "loss",
        "scheduler",
        "features",
        "feature-engineering",
        "learning_rate",
        "num_leaves",
        "n_estimators",
        "target_encoding",
        "log_numeric",
    }
)


def _score_dimensions(
    hypothesis: Hypothesis,
    *,
    graph: ExperimentGraph,
    kb: KnowledgeBase,
    metric_key: str,
    cheap_tags: set[str],
    default_expected_gain: float,
    risk_kb_bonus: float,
) -> dict[str, float]:
    tags = {normalize_technique(tag) for tag in hypothesis.tags if tag.strip()}
    matched_gains: list[float] = []
    best_positive_conf = 0.0
    for tag in tags:
        entry = kb.get(tag, metric_key)
        if entry is None:
            continue
        matched_gains.append(entry.delta_estimate * entry.confidence)
        if entry.effect == KnowledgeEffect.IMPROVES and entry.confidence > best_positive_conf:
            best_positive_conf = entry.confidence
    expected_gain = max(matched_gains) if matched_gains else default_expected_gain

    if tags & cheap_tags:
        implementation_cost = 0.2
    elif not tags:
        implementation_cost = 0.5
    else:
        implementation_cost = 0.85

    gpu_cost_seconds = _estimate_gpu_cost(graph, tags)
    risk = 1.0 - float(hypothesis.confidence)
    if best_positive_conf > 0:
        risk = max(0.0, risk - risk_kb_bonus * best_positive_conf)
    novelty = _novelty(graph, tags)

    return {
        "expected_gain": float(expected_gain),
        "implementation_cost": float(implementation_cost),
        "gpu_cost_seconds": float(gpu_cost_seconds),
        "risk": float(risk),
        "novelty": float(novelty),
    }


def _experiment_tags(exp: Experiment) -> set[str]:
    tags = {normalize_technique(recipe) for recipe in exp.feature_recipes}
    tags.update(normalize_technique(key) for key in exp.model_params)
    if exp.template_name:
        tags.add(normalize_technique(exp.template_name))
    return tags


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def _novelty(graph: ExperimentGraph, tags: set[str]) -> float:
    if not graph.nodes:
        return 1.0
    if not tags:
        return 0.5
    best = 0.0
    for exp in graph.nodes.values():
        best = max(best, _jaccard(tags, _experiment_tags(exp)))
    return max(0.0, 1.0 - best)


def _estimate_gpu_cost(graph: ExperimentGraph, tags: set[str]) -> float:
    timed = [
        exp for exp in graph.nodes.values() if exp.runtime_seconds is not None and exp.runtime_seconds > 0
    ]
    if not timed:
        return 0.0
    if tags:
        overlaps = [
            exp
            for exp in timed
            if _jaccard(tags, _experiment_tags(exp)) > 0.0
        ]
        if overlaps:
            return sum(float(exp.runtime_seconds or 0.0) for exp in overlaps) / len(overlaps)
    return sum(float(exp.runtime_seconds or 0.0) for exp in timed) / len(timed)


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]
