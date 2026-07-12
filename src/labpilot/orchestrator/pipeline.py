import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from labpilot.baseline.registry import get_template
from labpilot.baseline.selector import BaselineChoice, BaselineSelector
from labpilot.brief.generator import BriefGenerator
from labpilot.codegen.renderer import CodeRenderer
from labpilot.codegen.validators import validate_pipeline
from labpilot.competition.models import CompetitionSpec, ProblemType
from labpilot.competition.parser import CompetitionParser
from labpilot.config import AppConfig
from labpilot.data.downloader import DataDownloader
from labpilot.improvement.fork import fork_run
from labpilot.improvement.models import (
    load_training_overrides,
    save_improvement_plan,
    save_training_overrides,
)
from labpilot.improvement.planner import ImprovementPlanner
from labpilot.kaggle.client import KaggleClient, KaggleGateway, SubmissionResult
from labpilot.kaggle.urls import competition_submissions_url
from labpilot.kernel.exporter import export_kernel
from labpilot.llm.client import LLMClient, create_llm_client, resolve_llm_client
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
from labpilot.submission.formatter import SubmissionValidator
from labpilot.tracking.logger import ExperimentLogger
from labpilot.training.runner import TrainingRunner
from labpilot.runtimes.models import RuntimeRecord
from labpilot.runtimes.registry import get_runtime

logger = logging.getLogger(__name__)
console = Console()

StageHandler = Callable[[Path, RunManifest, AppConfig], list[str]]

# The two halves `research init` and `research build` split the full
# pipeline into: init resolves *what* to run (competition + data + brief),
# build actually runs it (baseline through reflection). `research run` still
# does both halves in one call for the default one-command experience.
INIT_STAGES = ["parse_competition", "download_data", "profile_dataset", "generate_brief"]
POST_CODEGEN_STAGES = {
    "train_model",
    "evaluate_cv",
    "generate_submission",
    "export_kernel",
    "upload_submission",
    "log_experiment",
    "write_reflection",
}


class Pipeline:
    """Linear stage orchestrator for the P0 research loop."""

    def __init__(
        self,
        config: AppConfig,
        kaggle_client: KaggleGateway | None = None,
        submit: bool = False,
        force_submit: bool = False,
        configs_dir: Path | None = None,
        llm_client: LLMClient | None = None,
        dry_run: bool = False,
        project_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.kaggle_client = kaggle_client or KaggleClient(config.kaggle)
        self.llm_client = llm_client if llm_client is not None else resolve_llm_client(config.llm)
        self.submit = submit
        self.force_submit = force_submit
        self.configs_dir = configs_dir
        self.dry_run = dry_run
        self.project_dir = project_dir
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
            "export_kernel": self._export_kernel,
            "upload_submission": self._upload_submission,
            "log_experiment": self._log_experiment,
            "write_reflection": self._write_reflection,
        }

    def run(self, competition: str, run_dir: Path | None = None) -> RunManifest:
        """Run every configured stage, start to finish, in one call."""
        manifest, resolved_run_dir, all_stages = self._start(competition, run_dir)
        return self._execute(resolved_run_dir, manifest, all_stages, all_stages)

    def init(self, competition: str, run_dir: Path | None = None) -> RunManifest:
        """Run only the init half: parse → download → profile → brief.

        Leaves the run in `partial` status, ready for `build()` (or
        `resume()`) to run the remaining stages once the resolved brief/
        baseline choice has been reviewed.
        """
        manifest, resolved_run_dir, all_stages = self._start(competition, run_dir)
        stages_to_run = [name for name in all_stages if name in INIT_STAGES]
        if not stages_to_run:
            raise ValueError("No init stages are configured; check config.pipeline.stages.")
        return self._execute(resolved_run_dir, manifest, stages_to_run, all_stages)

    def build(self, run_id: str) -> RunManifest:
        """Run the build half (baseline through reflection) of an already-`init`'d run."""
        return self._continue(run_id, require_done=INIT_STAGES)

    def resume(self, run_id: str) -> RunManifest:
        """Resume a run from its first failed/incomplete stage.

        Stages already `completed` or `skipped` are left as-is; everything
        else (`failed`, stuck `running` from a killed process, or never
        reached) is re-executed in pipeline order.
        """
        return self._continue(run_id)

    def improve(
        self,
        parent_run_id: str,
        strategy: str = "auto",
    ) -> RunManifest:
        """Fork a completed parent run, plan improvements, and re-run downstream stages."""
        parent_run_dir = (self.config.runs_dir / parent_run_id).resolve()
        if not (parent_run_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"Run not found: {parent_run_id}")

        parent_manifest = load_manifest(parent_run_dir)
        if parent_manifest.status != StageStatus.COMPLETED:
            raise ValueError(
                f"Parent run '{parent_run_id}' must be completed before improving "
                f"(status: {parent_manifest.status.value})."
            )

        planner = ImprovementPlanner(self.config.llm, self.llm_client)
        plan, overrides = planner.plan(
            parent_run_dir,
            parent_run_id,
            strategy=strategy,
            random_seed=self.config.training.random_seed,
        )

        child_run_id, child_run_dir = fork_run(
            parent_run_dir,
            self.config.runs_dir,
            parent_run_id=parent_run_id,
            improvement_strategy=strategy,
        )
        save_improvement_plan(child_run_dir, plan)
        save_training_overrides(child_run_dir, overrides)
        self._write_runtime_record(child_run_dir)

        child_manifest = load_manifest(child_run_dir)
        all_stages = self.config.pipeline.stages or list(self.handlers.keys())
        stages_to_run = plan.stages_to_run or [
            name for name in all_stages if name not in {s.name for s in child_manifest.stages}
        ]

        console.print(
            f"Improving '{parent_run_id}' → [cyan]{child_run_id}[/cyan] "
            f"(strategy={strategy}, iteration={child_manifest.metadata.get('iteration')})\n"
        )
        return self._execute(child_run_dir, child_manifest, stages_to_run, all_stages)

    def _start(self, competition: str, run_dir: Path | None) -> tuple[RunManifest, Path, list[str]]:
        run_id = generate_run_id(competition)
        # Resolved to absolute: the training stage runs the generated script
        # as a subprocess with its cwd set to the run's pipeline directory,
        # so a relative run_dir would be re-resolved against that new cwd.
        resolved_run_dir = (run_dir or self.config.runs_dir / run_id).resolve()
        resolved_run_dir.mkdir(parents=True, exist_ok=True)

        self._write_runtime_record(resolved_run_dir)

        manifest = RunManifest(
            run_id=run_id,
            competition=competition,
            status=StageStatus.RUNNING,
            stages=[],
        )
        save_manifest(resolved_run_dir, manifest)

        all_stages = self.config.pipeline.stages or list(self.handlers.keys())
        return manifest, resolved_run_dir, all_stages

    def _write_runtime_record(self, run_dir: Path) -> None:
        runtime_id = self.config.runtime.default_runtime
        runtime = get_runtime(runtime_id, self.config.runtime.runtimes_dir)
        provider = runtime.provider if runtime is not None else "local"
        record = RuntimeRecord(runtime_id=runtime_id, provider=provider, mode="local")
        (run_dir / "runtime.json").write_text(record.model_dump_json(indent=2))

    def _write_dry_run_summary(
        self,
        run_dir: Path,
        manifest: RunManifest,
        all_stages: list[str],
    ) -> None:
        skipped = [stage.name for stage in manifest.stages if stage.status == StageStatus.SKIPPED]
        summary = {
            "dry_run": True,
            "completed_through": "generate_code",
            "skipped_stages": skipped,
            "would_run_next": [
                name for name in all_stages if name in POST_CODEGEN_STAGES and name not in skipped
            ],
        }
        (run_dir / "dry_run.json").write_text(json.dumps(summary, indent=2))

    def _continue(self, run_id: str, require_done: list[str] | None = None) -> RunManifest:
        resolved_run_dir = (self.config.runs_dir / run_id).resolve()
        manifest = load_manifest(resolved_run_dir)
        all_stages = self.config.pipeline.stages or list(self.handlers.keys())

        done = {StageStatus.COMPLETED, StageStatus.SKIPPED}
        finished_names = {record.name for record in manifest.stages if record.status in done}

        # A `skipped` `upload_submission` means "nothing was uploaded" (the
        # prior call didn't pass `--submit`), not "this stage is finished" —
        # so if the caller now passes `--submit`, it must be re-run for real
        # instead of staying silently skipped forever.
        upload_record = manifest.stage("upload_submission")
        if (
            self.submit
            and upload_record is not None
            and upload_record.status == StageStatus.SKIPPED
        ):
            finished_names.discard("upload_submission")

        if require_done:
            missing = [
                name for name in require_done if name in all_stages and name not in finished_names
            ]
            if missing:
                raise ValueError(
                    f"Run '{run_id}' hasn't finished its init stage(s) yet: {missing}. "
                    "Run `research init --competition <slug>` first (or `research run` "
                    "for the full pipeline in one call)."
                )

        remaining = [name for name in all_stages if name not in finished_names]
        if not remaining:
            console.print(f"[green]Run '{run_id}' has nothing left to do.[/green]")
            return manifest

        console.print(
            f"Continuing '{run_id}' from stage [cyan]{remaining[0]}[/cyan] "
            f"({len(all_stages) - len(remaining)}/{len(all_stages)} already done).\n"
        )
        manifest.status = StageStatus.RUNNING
        save_manifest(resolved_run_dir, manifest)
        return self._execute(resolved_run_dir, manifest, remaining, all_stages)

    def _execute(
        self,
        resolved_run_dir: Path,
        manifest: RunManifest,
        stages: list[str],
        all_stages: list[str],
    ) -> RunManifest:
        competition = manifest.competition
        total = len(all_stages)

        for stage_name in stages:
            handler = self.handlers.get(stage_name)
            if handler is None:
                raise ValueError(f"Unknown pipeline stage: {stage_name}")

            if self.dry_run and stage_name in POST_CODEGEN_STAGES:
                manifest.mark_skipped(stage_name, [])
                save_manifest(resolved_run_dir, manifest)
                console.print(f"  [yellow]–[/yellow] {stage_name} (dry-run)")
                continue

            index = all_stages.index(stage_name) + 1
            console.print(f"[bold][{index}/{total}][/bold] {stage_name}...")
            if stage_name == "export_kernel":
                competition = self._load_competition(resolved_run_dir)
                if competition.submission_mode != "kernel":
                    manifest.mark_skipped(stage_name, [])
                    save_manifest(resolved_run_dir, manifest)
                    console.print(f"  [yellow]–[/yellow] {stage_name} (csv competition)")
                    continue
            if stage_name == "upload_submission" and not self.submit:
                competition = self._load_competition(resolved_run_dir)
                status = "kernel_ready" if competition.submission_mode == "kernel" else "not_submitted"
                result = SubmissionResult(
                    competition=manifest.competition,
                    submission_path=str(resolved_run_dir / "submission.csv"),
                    status=status,
                    message="Upload skipped; rerun with --submit to upload.",
                    submission_mode=competition.submission_mode,
                    submissions_url=competition.submissions_url
                    or competition_submissions_url(competition.slug),
                )
                result_path = KaggleClient.save_result(resolved_run_dir, result)
                manifest.mark_skipped(stage_name, [str(result_path)])
                save_manifest(resolved_run_dir, manifest)
                console.print(f"  [yellow]–[/yellow] {stage_name} (use --submit to enable)")
                continue

            manifest.mark_running(stage_name)
            save_manifest(resolved_run_dir, manifest)

            try:
                artifacts = handler(resolved_run_dir, manifest, self.config)
                manifest.mark_completed(stage_name, artifacts)
                save_manifest(resolved_run_dir, manifest)
                console.print(f"  [green]✔[/green] {stage_name}")
            except BaseException as exc:
                # Catches BaseException (not just Exception) so the manifest
                # never gets stuck showing "running" forever — some
                # dependencies (e.g. kaggle>=2.0 on auth failure) raise
                # SystemExit instead of a normal exception, and Ctrl-C
                # (KeyboardInterrupt) should also leave an honest record.
                manifest.mark_failed(stage_name, str(exc))
                save_manifest(resolved_run_dir, manifest)
                console.print(f"  [red]✘[/red] {stage_name}: {exc}")
                raise

        # Only claim the whole run is `completed` once every stage in the
        # full pipeline is finished — checked against the manifest's actual
        # state, not just whether this call's `stages` list happens to end
        # with the last stage name. That distinction matters for a targeted
        # re-run like `resume --submit` on an already-finished run: it only
        # (re-)executes `upload_submission`, but the run as a whole is still
        # complete once that single stage lands. `init()` still correctly
        # reads as `partial`, since stages after the brief were never run at
        # all and so have no manifest record yet.
        done = {StageStatus.COMPLETED, StageStatus.SKIPPED}
        finished_names = {record.name for record in manifest.stages if record.status in done}
        reached_the_end = bool(all_stages) and all(name in finished_names for name in all_stages)
        manifest.status = StageStatus.COMPLETED if reached_the_end else StageStatus.PARTIAL
        if self.dry_run:
            manifest.metadata["dry_run"] = True
            self._write_dry_run_summary(resolved_run_dir, manifest, all_stages)
        save_manifest(resolved_run_dir, manifest)
        return manifest

    def _parse_competition(
        self, run_dir: Path, manifest: RunManifest, config: AppConfig
    ) -> list[str]:
        parser = CompetitionParser(
            manifest.competition,
            configs_dir=self.configs_dir,
            metadata_fetcher=self.kaggle_client,
            llm_client=self.llm_client,
        )
        path = parser.save(run_dir)
        return [str(path)]

    def _download_data(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        downloader = DataDownloader(
            manifest.competition,
            config.kaggle,
            client=self.kaggle_client,
        )
        path = downloader.download(run_dir)
        return [str(path)]

    def _profile_dataset(
        self, run_dir: Path, manifest: RunManifest, config: AppConfig
    ) -> list[str]:
        competition = CompetitionSpec.model_validate_json(
            (run_dir / "competition.json").read_text()
        )
        profiler = TabularProfiler(config.profiler)
        data_dir = run_dir / "data" / "raw"
        profile = profiler.profile_directory(
            data_dir,
            manifest.competition,
            train_pattern=competition.train_file_pattern,
            test_pattern=competition.test_file_pattern,
            submission_pattern=competition.submission_file_pattern,
            llm_client=self.llm_client,
            competition_title=competition.title,
            competition_description=competition.description,
        )
        json_path, md_path = write_profile(run_dir, profile)
        return [str(json_path), str(md_path)]

    def _generate_brief(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        competition = CompetitionSpec.model_validate_json(
            (run_dir / "competition.json").read_text()
        )
        profile = load_profile(run_dir)
        generator = BriefGenerator(config.llm, self.llm_client)
        path = generator.save(run_dir, competition, profile)
        return [str(path)]

    def _select_baseline(
        self, run_dir: Path, manifest: RunManifest, config: AppConfig
    ) -> list[str]:
        competition = CompetitionSpec.model_validate_json(
            (run_dir / "competition.json").read_text()
        )
        profile = load_profile(run_dir)
        selector = BaselineSelector()
        choice = selector.select(competition, profile)
        path = selector.save(run_dir, choice)
        return [str(path)]

    def _generate_code(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        choice = BaselineChoice.model_validate_json((run_dir / "baseline_choice.json").read_text())
        template = get_template(choice.problem_type, template_name=choice.template_name)
        if template is None:
            raise ValueError(f"No template for {choice.problem_type}")

        overrides = load_training_overrides(run_dir)
        renderer = CodeRenderer(config.training)
        pipeline_dir = renderer.render(
            template,
            choice,
            run_dir,
            model_params=overrides.model_params or None,
            feature_recipes=overrides.feature_recipes or None,
            target_encoding_columns=overrides.target_encoding_columns or None,
            log_numeric_columns=overrides.log_numeric_columns or None,
        )
        errors = validate_pipeline(pipeline_dir)
        if errors:
            raise ValueError(f"Pipeline validation failed: {errors}")
        return [str(p) for p in sorted(pipeline_dir.iterdir())]

    def _train_model(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        runner = TrainingRunner(run_dir)
        result = runner.run()
        log_path = runner.save_run_log(result)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed:\n{result.stderr}")
        artifacts = runner.collect_artifacts()
        missing = {"models", "oof", "metrics"} - set(artifacts)
        if missing:
            raise RuntimeError(f"Training completed without required artifacts: {sorted(missing)}")
        return [str(log_path), *(str(path) for path in artifacts.values())]

    def _evaluate_cv(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError("Training did not produce metrics.json.")

        # The metric key to look for comes from the baseline choice (i.e. the
        # metric the selected template actually writes), not from the
        # competition's real evaluation metric — see
        # baseline.selector.DEFAULT_METRIC_BY_PROBLEM_TYPE.
        choice = BaselineChoice.model_validate_json((run_dir / "baseline_choice.json").read_text())
        expected_key = f"cv_{choice.metric_name}"
        metrics = json.loads(metrics_path.read_text())
        score = metrics.get(expected_key)
        if not isinstance(score, (int, float)):
            raise ValueError(f"metrics.json is missing numeric '{expected_key}'.")
        return [str(metrics_path)]

    def _generate_submission(
        self, run_dir: Path, manifest: RunManifest, config: AppConfig
    ) -> list[str]:
        submission_path = run_dir / "submission.csv"
        if not submission_path.exists():
            raise FileNotFoundError("Training did not produce submission.csv.")

        profile = load_profile(run_dir)
        if not profile.target_column:
            raise ValueError("Dataset profile is missing target_column.")
        choice = BaselineChoice.model_validate_json((run_dir / "baseline_choice.json").read_text())
        # Integer-label validation only makes sense for classification targets
        # that are *numerically* encoded (e.g. Titanic's 0/1 Survived, or a
        # digit-class target). Regression targets are continuous, and a
        # classification target can just as easily be string labels (e.g.
        # multi-class species names) — neither should be forced through an
        # integer check.
        target_column_profile = next(
            (column for column in profile.columns if column.name == profile.target_column),
            None,
        )
        target_is_numeric = bool(target_column_profile and target_column_profile.is_numeric)
        require_integer_target = (
            choice.problem_type
            in {
                ProblemType.TABULAR_CLASSIFICATION.value,
                ProblemType.TEXT_CLASSIFICATION.value,
                ProblemType.IMAGE_CLASSIFICATION.value,
            }
            and target_is_numeric
        )
        validator = SubmissionValidator()
        result = validator.validate(
            submission_path,
            expected_rows=profile.test_row_count,
            expected_columns=profile.submission_columns,
            target_column=profile.target_column,
            require_integer_target=require_integer_target,
        )
        if not result.valid:
            raise ValueError(f"Invalid submission: {result.errors}")
        return [str(submission_path)]

    def _export_kernel(
        self, run_dir: Path, manifest: RunManifest, config: AppConfig
    ) -> list[str]:
        competition = self._load_competition(run_dir)
        kernel_dir = export_kernel(
            run_dir,
            competition,
            username=self.config.kaggle.username or None,
        )
        return [str(kernel_dir / name) for name in ("run.py", "kernel-metadata.json")]

    def _upload_submission(
        self, run_dir: Path, manifest: RunManifest, config: AppConfig
    ) -> list[str]:
        competition = self._load_competition(run_dir)
        self._preflight_submission(competition)

        existing = self._load_submission_result(run_dir)
        if competition.submission_mode == "kernel":
            kernel_dir = run_dir / "kernel"
            if not kernel_dir.is_dir():
                raise FileNotFoundError(
                    "Kernel export not found. Re-run from export_kernel or rebuild the run."
                )
            retry_slug = None
            retry_version = None
            if existing is not None and existing.status == "kernel_pushed":
                retry_slug = existing.kernel_slug
                retry_version = existing.kernel_version
            result = self.kaggle_client.submit_via_kernel(
                manifest.competition,
                kernel_dir,
                output_file=competition.kernel_output_file,
                existing_kernel_slug=retry_slug,
                existing_kernel_version=retry_version,
            )
        else:
            result = self.kaggle_client.upload_submission(
                manifest.competition,
                run_dir / "submission.csv",
            )
        path = KaggleClient.save_result(run_dir, result)
        self._print_submission_links(result)
        return [str(path)]

    @staticmethod
    def _load_competition(run_dir: Path) -> CompetitionSpec:
        return CompetitionSpec.model_validate_json((run_dir / "competition.json").read_text())

    @staticmethod
    def _load_submission_result(run_dir: Path) -> SubmissionResult | None:
        path = run_dir / "submission_result.json"
        if not path.is_file():
            return None
        return SubmissionResult.model_validate_json(path.read_text())

    @staticmethod
    def _print_submission_links(result: SubmissionResult) -> None:
        if result.submissions_url:
            console.print(f"\n[bold]Submissions:[/bold] {result.submissions_url}")
        if result.kernel_url:
            console.print(f"[bold]Kernel:[/bold]      {result.kernel_url}")

    def _preflight_submission(self, competition: CompetitionSpec) -> None:
        if competition.submissions_disabled:
            raise ValueError(
                f"Submissions are disabled for '{competition.slug}' on Kaggle."
            )
        if competition.deadline:
            try:
                deadline = datetime.fromisoformat(competition.deadline.replace("Z", "+00:00"))
                if deadline.tzinfo is not None:
                    deadline = deadline.replace(tzinfo=None)
            except ValueError:
                logger.warning(
                    "Could not parse deadline %r for '%s'; skipping deadline check.",
                    competition.deadline,
                    competition.slug,
                )
            else:
                if deadline < datetime.now():
                    if self.force_submit:
                        logger.warning(
                            "Deadline for '%s' (%s) has passed; uploading anyway "
                            "because --force-submit was set.",
                            competition.slug,
                            competition.deadline,
                        )
                    else:
                        raise ValueError(
                            f"Competition '{competition.slug}' deadline ({competition.deadline}) "
                            "has already passed. Pass --force-submit with --submit to upload anyway."
                        )
        if competition.max_daily_submissions is not None:
            count = self.kaggle_client.count_todays_submissions(competition.slug)
            if count >= competition.max_daily_submissions:
                raise ValueError(
                    f"Daily submission quota reached for '{competition.slug}' "
                    f"({count}/{competition.max_daily_submissions})."
                )

    def _log_experiment(self, run_dir: Path, manifest: RunManifest, config: AppConfig) -> list[str]:
        logger = ExperimentLogger(run_dir)
        metrics: dict[str, float] = {}
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            raw = json.loads(metrics_path.read_text())
            metrics = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

        choice = BaselineChoice.model_validate_json((run_dir / "baseline_choice.json").read_text())
        overrides = load_training_overrides(run_dir)
        params: dict[str, Any] = {
            "template": choice.template_name,
            "problem_type": choice.problem_type,
        }
        if overrides.model_params:
            params["model_params"] = overrides.model_params
        if overrides.feature_recipes:
            params["feature_recipes"] = overrides.feature_recipes
        if manifest.metadata.get("parent_run_id"):
            params["parent_run_id"] = manifest.metadata["parent_run_id"]
            params["iteration"] = manifest.metadata.get("iteration", 0)
            params["improvement_strategy"] = manifest.metadata.get("improvement_strategy")

        path = logger.log(
            run_id=manifest.run_id,
            competition=manifest.competition,
            metrics=metrics,
            params=params,
            artifacts=[str(run_dir / "submission.csv"), str(run_dir / "oof.csv")],
        )
        return [str(path)]

    def _write_reflection(
        self, run_dir: Path, manifest: RunManifest, config: AppConfig
    ) -> list[str]:
        profile = load_profile(run_dir)
        choice = BaselineChoice.model_validate_json((run_dir / "baseline_choice.json").read_text())
        submission = SubmissionResult.model_validate_json(
            (run_dir / "submission_result.json").read_text()
        )
        generator = ReflectionGenerator(config.llm, self.llm_client)
        metrics = generator.load_metrics(run_dir)
        content = generator.generate(
            run_id=manifest.run_id,
            competition=manifest.competition,
            profile=profile,
            baseline=choice,
            metrics=metrics,
            submission=submission,
            run_dir=run_dir,
        )
        path = generator.save(run_dir, content, submission=submission)
        return [str(path)]


def get_run_dir(config: AppConfig, run_id: str) -> Path:
    return config.runs_dir / run_id


def find_manifest(config: AppConfig, run_id: str) -> RunManifest:
    run_dir = get_run_dir(config, run_id)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return load_manifest(run_dir)
