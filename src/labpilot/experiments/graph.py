import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from labpilot.baseline.selector import BaselineChoice
from labpilot.competition.models import CompetitionSpec
from labpilot.experiments.models import Experiment
from labpilot.improvement.models import (
    ImprovementPlan,
    load_improvement_plan,
    load_training_overrides,
)
from labpilot.orchestrator.manifest import RunManifest, StageStatus, load_manifest
from labpilot.tracking.store import ExperimentStore

logger = logging.getLogger(__name__)

# Mirrors `Pipeline.handlers`' keys in orchestrator/pipeline.py — duplicated
# here (rather than imported) to avoid a circular import: pipeline.py needs
# this module for `capture_git_commit()`. Used as the "total" side of
# `Experiment.progress` when a run's own config didn't override
# `pipeline.stages`.
_ALL_PIPELINE_STAGES: tuple[str, ...] = (
    "parse_competition",
    "download_data",
    "profile_dataset",
    "generate_brief",
    "select_baseline",
    "generate_code",
    "train_model",
    "evaluate_cv",
    "generate_submission",
    "export_kernel",
    "upload_submission",
    "log_experiment",
    "write_reflection",
    "write_report",
)

# Well-known artifact paths, relative to a run directory, matching
# ARCHITECTURE.md's Run Artifact Layout. `Experiment.artifacts` is computed
# by checking which of these exist at read time, rather than trusted from
# `experiment/record.json` (which is written mid-pipeline and would miss
# later-stage artifacts like reflection.md/report.html).
_ARTIFACT_CANDIDATES: tuple[str, ...] = (
    "competition.json",
    "profile.json",
    "profile.md",
    "brief.md",
    "baseline_choice.json",
    "training_overrides.json",
    "improvement_plan.json",
    "config.json",
    "pipeline/train.py",
    "pipeline/config.yaml",
    "models",
    "oof.csv",
    "metrics.json",
    "submission.csv",
    "kernel",
    "submission_result.json",
    "training.log",
    "experiment/record.json",
    "reflection.md",
    "report.html",
)


def capture_git_commit(cwd: Path | None = None) -> str | None:
    """Best-effort `git rev-parse HEAD`. Never raises — `None` outside a git
    checkout, if `git` isn't on `PATH`, or on any other failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git_commit capture failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _scan_artifacts(run_dir: Path) -> list[str]:
    return [
        str(run_dir / relative)
        for relative in _ARTIFACT_CANDIDATES
        if (run_dir / relative).exists()
    ]


def _load_config_snapshot(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return {}


def _load_baseline_choice(run_dir: Path) -> BaselineChoice | None:
    path = run_dir / "baseline_choice.json"
    if not path.is_file():
        return None
    try:
        return BaselineChoice.model_validate_json(path.read_text())
    except (OSError, ValueError) as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None


def _load_metrics(run_dir: Path) -> dict[str, float]:
    record = ExperimentStore(run_dir).load()
    if record and record.metrics:
        return record.metrics
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            raw = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            return {}
        return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    return {}


def _load_public_score(run_dir: Path) -> float | None:
    path = run_dir / "submission_result.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    score = data.get("public_score")
    return float(score) if isinstance(score, (int, float)) else None


def _compute_runtime_seconds(manifest: RunManifest) -> float | None:
    stage = manifest.stage("train_model")
    if stage is None or stage.started_at is None or stage.finished_at is None:
        return None
    return (stage.finished_at - stage.started_at).total_seconds()


def _compute_progress(manifest: RunManifest, config_snapshot: dict[str, Any]) -> str:
    configured_stages = config_snapshot.get("pipeline", {}).get("stages") or list(
        _ALL_PIPELINE_STAGES
    )
    total = len(configured_stages)
    done_statuses = {StageStatus.COMPLETED, StageStatus.SKIPPED}
    completed = sum(1 for stage in manifest.stages if stage.status in done_statuses)
    return f"{completed}/{total} stages"


def _compute_description(
    *,
    competition: str,
    parent_id: str | None,
    template_name: str | None,
    improvement_plan: ImprovementPlan | None,
    improvement_strategy: str | None,
    hypothesis_prediction: str | None = None,
) -> str:
    if hypothesis_prediction:
        return hypothesis_prediction
    if parent_id is None:
        if template_name:
            return f"{template_name} baseline for {competition}"
        return f"baseline for {competition}"
    if improvement_plan is not None and improvement_plan.rationale:
        return improvement_plan.rationale
    plan_strategy = improvement_plan.strategy if improvement_plan else None
    strategy = improvement_strategy or plan_strategy or "unknown"
    return f"{strategy} iteration on {parent_id}"


def _load_hypothesis_prediction(
    knowledge_dir: Path, competition: str, hypothesis_id: str | None
) -> str | None:
    if not hypothesis_id:
        return None
    # Local import avoids a circular dependency with hypothesis.py importing graph.
    from labpilot.experiments.hypothesis import HypothesisStore

    hypothesis = HypothesisStore(knowledge_dir, competition).get(hypothesis_id)
    if hypothesis is None:
        return None
    prediction = hypothesis.prediction.strip()
    return prediction or None


def assemble_experiment(
    run_dir: Path, *, knowledge_dir: Path | None = None
) -> Experiment:
    """Assemble one `Experiment` from a run directory's existing artifacts.

    Single-run assembly only — `children_ids` is always `[]` here, since
    this function has no visibility into sibling runs. Callers that need the
    full parent/child graph (`build_graph`) patch it in afterward.
    """
    manifest = load_manifest(run_dir)
    config_snapshot = _load_config_snapshot(run_dir)
    baseline_choice = _load_baseline_choice(run_dir)
    overrides = load_training_overrides(run_dir)
    improvement_plan = load_improvement_plan(run_dir)

    parent_id = manifest.metadata.get("parent_run_id")
    template_name = baseline_choice.template_name if baseline_choice else None
    problem_type = baseline_choice.problem_type if baseline_choice else None
    hypothesis_id = manifest.metadata.get("hypothesis_id")

    resolved_knowledge = knowledge_dir
    if resolved_knowledge is None:
        raw = config_snapshot.get("knowledge_dir")
        resolved_knowledge = Path(raw) if raw else Path("knowledge")

    reflection_path = run_dir / "reflection.md"
    report_path = run_dir / "report.html"

    return Experiment(
        id=manifest.run_id,
        competition=manifest.competition,
        status=manifest.status.value,
        progress=_compute_progress(manifest, config_snapshot),
        description=_compute_description(
            competition=manifest.competition,
            parent_id=parent_id,
            template_name=template_name,
            improvement_plan=improvement_plan,
            improvement_strategy=manifest.metadata.get("improvement_strategy"),
            hypothesis_prediction=_load_hypothesis_prediction(
                resolved_knowledge, manifest.competition, hypothesis_id
            ),
        ),
        parent_id=parent_id,
        children_ids=[],
        iteration=int(manifest.metadata.get("iteration", 0)),
        hypothesis_id=hypothesis_id,
        git_commit=manifest.metadata.get("git_commit"),
        template_name=template_name,
        problem_type=problem_type,
        model_params=overrides.model_params,
        feature_recipes=overrides.feature_recipes,
        metrics=_load_metrics(run_dir),
        public_score=_load_public_score(run_dir),
        runtime_seconds=_compute_runtime_seconds(manifest),
        config_snapshot=config_snapshot,
        artifacts=_scan_artifacts(run_dir),
        reflection_path=str(reflection_path) if reflection_path.is_file() else None,
        report_path=str(report_path) if report_path.is_file() else None,
        created_at=manifest.created_at,
    )


def _pick_best(candidates: list[Experiment], metric_key: str, maximize: bool) -> Experiment | None:
    scored = [candidate for candidate in candidates if metric_key in candidate.metrics]
    if not scored:
        return None
    if maximize:
        return max(scored, key=lambda exp: exp.metrics[metric_key])
    return min(scored, key=lambda exp: exp.metrics[metric_key])


class ExperimentGraph:
    """Parent/child view over every `Experiment` in one competition.

    Not persisted — rebuilt from disk on every `build_graph()` call.
    """

    def __init__(
        self, competition: str, nodes: dict[str, Experiment], *, maximize: bool = True
    ) -> None:
        self.competition = competition
        self.nodes = nodes
        self.maximize = maximize

    @property
    def roots(self) -> list[Experiment]:
        return [exp for exp in self.nodes.values() if exp.parent_id is None]

    def children(self, run_id: str) -> list[Experiment]:
        exp = self.nodes.get(run_id)
        if exp is None:
            return []
        return [self.nodes[child_id] for child_id in exp.children_ids if child_id in self.nodes]

    def ancestors(self, run_id: str) -> list[Experiment]:
        """Root-ward walk: immediate parent first, root last."""
        result: list[Experiment] = []
        current = self.nodes.get(run_id)
        while current is not None and current.parent_id is not None:
            parent = self.nodes.get(current.parent_id)
            if parent is None:
                break
            result.append(parent)
            current = parent
        return result

    def descendants(self, run_id: str) -> list[Experiment]:
        """Leaf-ward walk: every experiment reachable from `run_id`, any order."""
        result: list[Experiment] = []
        stack = list(self.children(run_id))
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(self.children(node.id))
        return result

    def best_path(self, metric_key: str) -> list[Experiment]:
        """Root-to-leaf path maximizing (or minimizing) `metric_key` at each branch point."""
        path: list[Experiment] = []
        current = _pick_best(self.roots, metric_key, self.maximize)
        while current is not None:
            path.append(current)
            current = _pick_best(self.children(current.id), metric_key, self.maximize)
        return path

    def to_tree_text(self, metric_key: str | None = None) -> str:
        best_ids = {exp.id for exp in self.best_path(metric_key)} if metric_key else set()
        lines: list[str] = []
        roots = sorted(self.roots, key=lambda exp: exp.created_at)
        for index, root in enumerate(roots):
            self._render_node(root, lines, "", index == len(roots) - 1, metric_key, best_ids)
        return "\n".join(lines) if lines else "(no experiments)"

    def _render_node(
        self,
        exp: Experiment,
        lines: list[str],
        prefix: str,
        is_last: bool,
        metric_key: str | None,
        best_ids: set[str],
    ) -> None:
        connector = "└── " if is_last else "├── "
        score = ""
        if metric_key and metric_key in exp.metrics:
            score = f" ({exp.metrics[metric_key]:.4f})"
        marker = " *" if exp.id in best_ids else ""
        lines.append(f"{prefix}{connector}{exp.id} [{exp.status}]{score}{marker}")

        children = sorted(self.children(exp.id), key=lambda child: child.created_at)
        extension = "    " if is_last else "│   "
        for index, child in enumerate(children):
            self._render_node(
                child, lines, prefix + extension, index == len(children) - 1, metric_key, best_ids
            )


def _resolve_metric_direction(runs_dir: Path, run_ids: list[str]) -> bool:
    """`True` = maximize. Reads the first available `competition.json`;
    defaults to maximize when none is found or the spec has no metric."""
    for run_id in run_ids:
        path = runs_dir / run_id / "competition.json"
        if not path.is_file():
            continue
        try:
            spec = CompetitionSpec.model_validate_json(path.read_text())
        except (OSError, ValueError):
            continue
        if spec.evaluation_metric is not None:
            return spec.evaluation_metric.direction != "minimize"
    return True


def build_graph(
    runs_dir: Path, competition: str, *, knowledge_dir: Path | None = None
) -> ExperimentGraph:
    """Scan `runs_dir` for every run belonging to `competition` and assemble
    the full parent/child graph."""
    nodes: dict[str, Experiment] = {}
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            if not (run_dir / "manifest.json").is_file():
                continue
            manifest = load_manifest(run_dir)
            if manifest.competition != competition:
                continue
            nodes[manifest.run_id] = assemble_experiment(
                run_dir, knowledge_dir=knowledge_dir
            )

    children_map: dict[str, list[str]] = {run_id: [] for run_id in nodes}
    for exp in nodes.values():
        if exp.parent_id and exp.parent_id in children_map:
            children_map[exp.parent_id].append(exp.id)
    for run_id, child_ids in children_map.items():
        if child_ids:
            nodes[run_id] = nodes[run_id].model_copy(update={"children_ids": sorted(child_ids)})

    maximize = _resolve_metric_direction(runs_dir, list(nodes))
    return ExperimentGraph(competition=competition, nodes=nodes, maximize=maximize)
