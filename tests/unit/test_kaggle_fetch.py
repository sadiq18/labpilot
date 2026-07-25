"""Spike + research fetch — offline Kaggle kernels/discussions ingest."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli import main as cli_main
from labpilot.config import KaggleConfig
from labpilot.kaggle.client import KaggleClient
from labpilot.research_engine.intelligence.fetch import KaggleFetchService
from labpilot.research_engine.intelligence.knowledge import KnowledgeStore
from labpilot.research_engine.intelligence.micro_agents.forum_analyzer import (
    ForumAnalyzerAgent,
)
from labpilot.common.micro_agents import StructuredContext

runner = CliRunner()


class FakeFetchApi:
    """In-memory stand-in for KaggleApi kernel/discussion surfaces."""

    def __init__(self) -> None:
        self.kernel_pages: dict[int, list[object]] = {}
        self.topics_pages: dict[int, list[object]] = {}
        self.topic_messages: dict[int, list[object]] = {}
        self.pulled: list[str] = []

    def kernels_list(self, **kwargs):
        page = int(kwargs.get("page") or 1)
        return list(self.kernel_pages.get(page, []))

    def kernels_pull(self, kernel, path, metadata=False, quiet=True):
        self.pulled.append(str(kernel))
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "train.py").write_text(
            "# Focal Loss + Mixup + EMA\n"
            "criterion = FocalLoss()\n"
            "use_mixup = True\n"
            "ema = True\n"
        )
        if metadata:
            (dest / "kernel-metadata.json").write_text(
                json.dumps({"id": kernel, "title": "fake"})
            )

    def competition_list_topics(self, competition, sort_by=None, page=None):
        page_n = int(page or 1)

        class _Resp:
            def __init__(self, topics):
                self.topics = topics

        return _Resp(list(self.topics_pages.get(page_n, [])))

    def competition_list_topic_messages(
        self, competition, topic_id, sort_by=None, page_size=None
    ):
        class _Resp:
            def __init__(self, messages):
                self.messages = messages

        return _Resp(list(self.topic_messages.get(int(topic_id), [])))


class _Obj:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _kernel(
    ref: str, title: str, votes: int = 1, public_score: float | None = None
) -> _Obj:
    owner, slug = ref.split("/", 1)
    return _Obj(
        id=hash(ref) % 10_000,
        ref=ref,
        title=title,
        author=owner,
        slug=slug,
        language="python",
        kernel_type="notebook",
        total_votes=votes,
        best_public_score=public_score,
        current_version_number=1,
        last_run_time=None,
    )


def _topic(topic_id: int, title: str, votes: int = 5) -> _Obj:
    return _Obj(
        id=topic_id,
        title=title,
        topic_url=f"https://www.kaggle.com/competitions/x/discussion/{topic_id}",
        author_name="alice",
        comment_count=2,
        votes=votes,
        post_date=None,
        is_sticky=False,
    )


def _msg(content: str, *, msg_id: int = 1) -> _Obj:
    return _Obj(
        id=msg_id,
        author_name="bob",
        votes=1,
        post_date=None,
        content=content,
        raw_markdown=content,
        is_deleted=False,
        is_pinned=False,
        replies=[],
    )


def _client(api: FakeFetchApi) -> KaggleClient:
    return KaggleClient(KaggleConfig(), api=api)


def test_unique_limit_stops_and_skips_existing(tmp_path: Path) -> None:
    api = FakeFetchApi()
    api.kernel_pages[1] = [
        _kernel("a/one", "One", 10, public_score=0.973),
        _kernel("a/two", "Two", 9),
        _kernel("a/three", "Three", 8),
    ]
    api.kernel_pages[2] = [_kernel("a/four", "Four", 7)]
    knowledge = tmp_path / "knowledge"
    service = KaggleFetchService(llm_client=None, kaggle=_client(api))

    first = service.fetch(
        "birdclef-2026",
        sources={"kernels"},
        kernel_sort="voteCount",
        limit=2,
        knowledge_dir=knowledge,
    )
    assert first.written == 2
    assert first.skipped_existing == 0
    assert len(first.artifact_ids) == 2
    assert all(aid.startswith("kaggle-kernel:") for aid in first.artifact_ids)

    # Limit counts NEW uniques — skips a/one+a/two, writes next two.
    second = service.fetch(
        "birdclef-2026",
        sources={"kernels"},
        kernel_sort="voteCount",
        limit=2,
        knowledge_dir=knowledge,
    )
    assert second.written == 2
    assert second.skipped_existing == 2
    assert set(second.artifact_ids) == {
        "kaggle-kernel:a/three",
        "kaggle-kernel:a/four",
    }

    third = service.fetch(
        "birdclef-2026",
        sources={"kernels"},
        kernel_sort="voteCount",
        limit=2,
        knowledge_dir=knowledge,
    )
    assert third.written == 0
    assert third.skipped_existing >= 4

    with KnowledgeStore(knowledge, "birdclef-2026") as store:
        artifact = store.get_artifact(first.artifact_ids[0])
        assert artifact is not None
        assert artifact.type.value == "repository"
        assert artifact.source == "kaggle"
        assert artifact.metadata.get("kind") == "kaggle_kernel"
        assert artifact.metadata.get("extraction_source") == "rule_engine"
        # Rule engine should pick up Mixup / Focal / EMA from fake train.py.
        labels = {t.lower() for t in artifact.techniques}
        assert labels & {"mixup", "focal loss", "ema"}
        # Public score (when the API exposes one) must survive into metadata.
        scored = store.get_artifact("kaggle-kernel:a/one")
        assert scored is not None
        assert scored.metadata.get("public_score") == 0.973
        unscored = store.get_artifact("kaggle-kernel:a/two")
        assert unscored is not None
        assert unscored.metadata.get("public_score") is None


def test_discussions_enrich_rule_engine(tmp_path: Path) -> None:
    api = FakeFetchApi()
    api.topics_pages[1] = [
        _topic(101, "Public LB is misleading"),
        _topic(102, "Dataset bug in labels"),
    ]
    api.topic_messages[101] = [
        _msg(
            "Found that public LB shake-up happens often. Avoid target leakage.",
            msg_id=1,
        )
    ]
    api.topic_messages[102] = [
        _msg("There is a dataset bug / labelling error on rare classes.", msg_id=2)
    ]
    knowledge = tmp_path / "knowledge"
    service = KaggleFetchService(llm_client=None, kaggle=_client(api))
    result = service.fetch(
        "birdclef-2026",
        sources={"discussions"},
        discussion_sort="top",
        limit=2,
        knowledge_dir=knowledge,
    )
    assert result.written == 2
    assert result.rule_engine_enriched == 2
    with KnowledgeStore(knowledge, "birdclef-2026") as store:
        art = store.get_artifact("kaggle-discussion:birdclef-2026:101")
        assert art is not None
        assert art.type.value == "discussion"
        assert art.source == "kaggle"
        extract = art.metadata.get("forum_extract") or {}
        assert extract.get("mistakes") or extract.get("lb_shakeups") or extract.get(
            "discoveries"
        )


def test_discussion_provider_unavailable_soft_fails(tmp_path: Path) -> None:
    class BoomApi(FakeFetchApi):
        def competition_list_topics(self, competition, sort_by=None, page=None):
            raise RuntimeError("auth failed")

    service = KaggleFetchService(llm_client=None, kaggle=_client(BoomApi()))
    result = service.fetch(
        "birdclef-2026",
        sources={"discussions"},
        limit=5,
        knowledge_dir=tmp_path / "knowledge",
    )
    assert result.written == 0
    assert any("unavailable" in note.lower() for note in result.notes)


def test_forum_agent_heuristics_without_preparsed_data() -> None:
    agent = ForumAnalyzerAgent(llm_client=None)
    out = agent.run(
        StructuredContext(
            text=(
                "Avoid this mistake: target leakage on folds.\n"
                "Found that Mixup improved rare classes.\n"
                "Dataset bug: corrupted wav files.\n"
            )
        )
    )
    assert out.mistakes or out.discoveries or out.dataset_bugs


def _plain(text: str) -> str:
    """Strip ANSI codes and collapse whitespace.

    Typer/rich renders --help inside a panel and folds long option tokens across
    lines at terminal widths that differ between local and CI. Collapsing
    whitespace makes flag assertions robust to that wrapping.
    """
    without_ansi = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"\s+", "", without_ansi)


def test_fetch_cli_help() -> None:
    result = runner.invoke(
        cli_main.app, ["fetch", "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"}
    )
    assert result.exit_code == 0
    plain = _plain(result.stdout)
    assert "--source" in plain
    assert "--limit" in plain
    assert "--sort" in plain


def test_paths_include_kernels_raw(tmp_path: Path) -> None:
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    paths = ResearchPaths(tmp_path / "knowledge", "birdclef-2026").ensure()
    assert (paths.raw_dir / "kernels").is_dir()
    assert (paths.raw_dir / "discussions").is_dir()
