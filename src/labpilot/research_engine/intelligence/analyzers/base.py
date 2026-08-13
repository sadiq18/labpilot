"""Analyzer plugin interface (design §3.2).

An analyzer turns an ``AnalyzeContext`` into ``ResearchArtifacts``. It must
soft-fail (return empty artifacts + notes) rather than raise, so one broken
source never takes down a whole ``research analyze`` run.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from labpilot.research_engine.intelligence.models import AnalyzeContext, ResearchArtifacts


@runtime_checkable
class Analyzer(Protocol):
    """One pluggable research-intelligence content type."""

    name: str
    # Stable id: "competition", "papers", "repositories", "experiments",
    # "dataset", "discussions", …
    default_enabled: bool  # DiscussionAnalyzer starts False until a provider ships

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        """Read cache / M2 / call providers. Soft-fail → empty artifacts + notes."""
        ...


class BaseAnalyzer:
    """Convenience base: sets ``name`` / ``default_enabled`` and an empty result.

    Subclasses override :meth:`analyze`. The ``Independence rule`` (§3.2) holds:
    an analyzer never calls another analyzer — only its own providers, caches,
    and read-only execution libraries.
    """

    name: str = ""
    default_enabled: bool = True

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:  # pragma: no cover - abstract
        raise NotImplementedError

    def _empty(self, *notes: str) -> ResearchArtifacts:
        return ResearchArtifacts(analyzer=self.name, notes=list(notes))

    def _maybe_attach_llm_client(self, context: AnalyzeContext | None = None) -> None:
        """Attach an optional LLM client, resolved from the *workspace's* config.

        One implementation, on the base, because there were three identical ones
        and the bug this fixes was that a change had to be made in all of them:
        `load_config()` reads only the package default, where `routing` is empty,
        so `build_gateway` returned None and every call fell through to the
        legacy provider pin — a workspace naming fourteen routable endpoints ran
        every campaign on `ollama`.

        `context` matters. `load_config_for_cwd()` with no arguments discovers
        the workspace by walking up from the *process* directory, and nothing in
        `labpilot` ever chdirs — so `research conduct --workspace /data/rogii`
        launched from a checkout finds no `labpilot.yaml`, falls back to the
        package default, and reproduces the same bug the workspace-aware loader
        was supposed to end. `knowledge_dir` sits inside the workspace, so
        starting the walk there finds the right one wherever the process lives.

        Never required: any failure leaves `llm_client` None, which every caller
        already treats as "no enrichment".
        """
        if getattr(self, "_llm_explicit", False) or getattr(self, "llm_client", None) is not None:
            return
        try:
            from labpilot.llm.client import resolve_llm_client
            from labpilot.workspace import load_config_for_cwd

            start = context.knowledge_dir if context is not None else None
            config, _ = load_config_for_cwd(start=start)
            self.llm_client = resolve_llm_client(config.llm)
        except Exception:
            self.llm_client = None
