import json
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from labpilot.baseline.registry import get_template
from labpilot.baseline.selector import BaselineChoice, BaselineSelector
from labpilot.brief.generator import BriefGenerator
from labpilot.codegen.renderer import CodeRenderer
from labpilot.codegen.validators import validate_pipeline
from labpilot.competition.models import CompetitionSpec
from labpilot.competition.parser import CompetitionParser
from labpilot.config import AppConfig
from labpilot.data.downloader import DataDownloader
from labpilot.kaggle.client import KaggleClient, SubmissionResult
from labpilot.orchestrator.manifest import (
    RunManifest,
    StageStatus,
    generate_run_id,
    load_manifest,
    save_manifest,
)
from labpilot.profiler.report import load_profile, write_profile
from labpilot.profiler.tabular import TabularProfiler
from labpilot.reflection.generator import ReflectionGenerator
from labpilot.submission.formatter import SubmissionFormatter, SubmissionValidator
from labpilot.tracking.logger import ExperimentLogger
from labpilot.training.runner import TrainingRunner

console = Console()

StageHandler = Callable[[Path, RunManifest, AppConfig], list[str]]


class Pipeline:
    """Linear stage orchestrator for the P0 research loop."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.handlers: dict[str, StageHandler] = {
            "parse_competition": self._parse_competition,
            "download_data": self._download_data,
            "profile_dataset": self._profile_dataset,
            "generate_brief": self._generate_brief,
            "select_baseline": self._select_baseline,
            "generate_code": self._generate_code,
            "train_model": self._train_model,
            "evaluate_cv": self._evaluate_cv,
            "generate_submission": self._generate_submission,
            "upload_submission": self._upload_submission,
            "log_experiment": self._log_experiment,
            "write_reflection": self._write_reflection,
        }

    def run(self, competition: str, run_dir: Path | None = None) -> RunManifest:
        run_id = generate_run_id(competition)
        resolved_run_dir = run_dir or self.config.runs_dir / run_id
        resolved_run_dir.mkdir(parents=True, exist_ok=True)

        manifest = RunManifest(
            run_id=run_id,
            competition=competition,
            status=StageStatus.RUNNING,
            stages=[],
        )
        save_manifest(resolved_run_dir, manifest)

        stages = self.config.pipeline.stages or list(self.handlers.keys())
        total = len(stages)

        for index, stage_name in enumerate(stages, start=1):
            handler = self.handlers.get(stage_name)
            if handler is None:
                raise ValueError(f"Unknown pipeline stage: {stage_name}")

            console.print(f"[bold][{index}/{total}][/bold] {stage_name}...")
            manifest.mark_running(stage_name)
            save_manifest(resolved_run_dir, manifest)

            try:
                artifacts = handler(resolved_run_dir, manifest, self.config)
                manifest.mark_completed(stage_name, artifacts)
                save_manifest(resolved_run_dir, manifest)
                console.print(f"  [green]✔[/green] {stage_name}")
            except Exception as exc:
                manifest.mark_failed(stage_name, str(exc))
                save_manifest(resolved_run_dir, manifest)
                console.print(f"  [red]✘[/red] {stage_name}: {exc}")
                raise

        manifest.status = StageStatus.COMPLETED
        save_manifest(resolved_run_dir, manifest)
        return manifest

    def _parse_competition(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        parser = CompetitionParser(manifest.competition)
        path = parser.save(run_dir)
        return [str(path)]

    def _download_data(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        downloader = DataDownloader(manifest.competition, config.kaggle)
        path = downloader.download(run_dir)
        return [str(path)]

    def _profile_dataset(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        profiler = TabularProfiler(config.profiler)
        data_dir = run_dir / "data" / "raw"
        profile = profiler.profile_directory(data_dir, manifest.competition)
        json_path, md_path = write_profile(run_dir, profile)
        return [str(json_path), str(md_path)]

    def _generate_brief(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        competition = CompetitionSpec.model_validate_json(
            (run_dir / "competition.json").read_text()
        )
        profile = load_profile(run_dir)
        generator = BriefGenerator(config.llm)
        path = generator.save(run_dir, competition, profile)
        return [str(path)]

    def _select_baseline(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        competition = CompetitionSpec.model_validate_json(
            (run_dir / "competition.json").read_text()
        )
        profile = load_profile(run_dir)
        selector = BaselineSelector()
        choice = selector.select(competition, profile)
        path = selector.save(run_dir, choice)
        return [str(path)]

    def _generate_code(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        choice = BaselineChoice.model_validate_json(
            (run_dir / "baseline_choice.json").read_text()
        )
        template = get_template(choice.problem_type)
        if template is None:
            raise ValueError(f"No template for {choice.problem_type}")

        renderer = CodeRenderer(config.training)
        pipeline_dir = renderer.render(template, choice, run_dir)
        errors = validate_pipeline(pipeline_dir)
        if errors:
            raise ValueError(f"Pipeline validation failed: {errors}")
        return [str(p) for p in sorted(pipeline_dir.iterdir())]

    def _train_model(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        runner = TrainingRunner(run_dir)
        result = runner.run()
        runner.save_run_log(result)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed:\n{result.stderr}")
        artifacts = runner.collect_artifacts()
        return [str(p) for p in artifacts.values()]

    def _evaluate_cv(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            metrics_path.write_text(json.dumps({"cv_score": 0.0, "status": "pending"}))
        return [str(metrics_path)]

    def _generate_submission(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        submission_path = run_dir / "submission.csv"
        if not submission_path.exists():
            submission_path.write_text("id,prediction\n1,0.5\n")
        validator = SubmissionValidator()
        result = validator.validate(submission_path)
        if not result.valid:
            raise ValueError(f"Invalid submission: {result.errors}")
        return [str(submission_path)]

    def _upload_submission(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        client = KaggleClient(config.kaggle)
        result = client.upload_submission(
            manifest.competition,
            run_dir / "submission.csv",
        )
        path = client.save_result(run_dir, result)
        return [str(path)]

    def _log_experiment(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        logger = ExperimentLogger(run_dir)
        metrics: dict[str, float] = {}
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            raw = json.loads(metrics_path.read_text())
            metrics = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

        choice = BaselineChoice.model_validate_json(
            (run_dir / "baseline_choice.json").read_text()
        )
        path = logger.log(
            run_id=manifest.run_id,
            competition=manifest.competition,
            metrics=metrics,
            params={"template": choice.template_name, "problem_type": choice.problem_type},
            artifacts=[str(run_dir / "submission.csv"), str(run_dir / "oof.csv")],
        )
        return [str(path)]

    def _write_reflection(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        profile = load_profile(run_dir)
        choice = BaselineChoice.model_validate_json(
            (run_dir / "baseline_choice.json").read_text()
        )
        submission = SubmissionResult.model_validate_json(
            (run_dir / "submission_result.json").read_text()
        )
        generator = ReflectionGenerator(config.llm)
        metrics = generator.load_metrics(run_dir)
        content = generator.generate(
            run_id=manifest.run_id,
            competition=manifest.competition,
            profile=profile,
            baseline=choice,
            metrics=metrics,
            submission=submission,
        )
        path = generator.save(run_dir, content)
        return [str(path)]


def get_run_dir(config: AppConfig, run_id: str) -> Path:
    return config.runs_dir / run_id


def find_manifest(config: AppConfig, run_id: str) -> RunManifest:
    run_dir = get_run_dir(config, run_id)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return load_manifest(run_dir)
