"""``RepositoryAnalyzerAgent`` — structured card for a GitHub repository.

Flagship "Yes" pattern (design §2.4). Deterministic fetch happens upstream;
the agent turns repo text into a :class:`RepoExtract` (architecture, notable
components, files worth reading, integration difficulty). Never "summarize this
repository".
"""

from __future__ import annotations

from labpilot.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
from labpilot.research_engine.intelligence.micro_agents.artifacts import RepoExtract

_DIFFICULTIES = {"easy", "medium", "hard", "unknown"}


class RepositoryAnalyzerAgent(BaseMicroAgent):
    name = "RepositoryAnalyzerAgent"
    output_model = RepoExtract

    def system_prompt(self) -> str:
        return (
            "You analyze an ML GitHub repository. Respond ONLY with a JSON "
            'object: {"architecture": str, "components": [str], '
            '"files_worth_reading": [str], "techniques": [str], '
            '"integration_difficulty": "Easy"|"Medium"|"Hard"}. '
            "Identify the core architecture, interesting components, files a "
            "practitioner should read, and how hard it is to reuse."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Repository text:\n{context.text}"

    def _run_rule_engine(self, context: StructuredContext) -> RepoExtract:
        d = context.data
        difficulty = str(d.get("integration_difficulty", "unknown")).strip() or "unknown"
        if difficulty.lower() not in _DIFFICULTIES:
            difficulty = "unknown"
        techniques = coerce_str_list(d.get("techniques"))
        return RepoExtract(
            architecture=str(d.get("architecture", "")),
            components=coerce_str_list(d.get("components")) or techniques,
            files_worth_reading=coerce_str_list(d.get("files_worth_reading")),
            techniques=techniques,
            integration_difficulty=difficulty,
        )
