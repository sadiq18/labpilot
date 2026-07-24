"""Compare repository knowledge with local code and suggest transfer opportunities."""

from __future__ import annotations

from labpilot.research_engine.intelligence.repositories.models import (
    EffortEstimate,
    ExpectedGain,
    LocalCodeProfile,
    RepoKnowledge,
    TransferOpportunity,
)


class RepoDiffer:
    def compare(
        self,
        local: LocalCodeProfile,
        knowledge: list[RepoKnowledge],
    ) -> list[TransferOpportunity]:
        opportunities: list[TransferOpportunity] = []
        for repo in knowledge:
            opportunities.extend(self._compare_one(local, repo))
        return opportunities

    def _compare_one(
        self,
        local: LocalCodeProfile,
        remote: RepoKnowledge,
    ) -> list[TransferOpportunity]:
        groups = (
            ("loss", local.loss, remote.loss, EffortEstimate.MINUTES_20, ExpectedGain.MEDIUM),
            (
                "augmentation",
                local.augmentation,
                remote.augmentation,
                EffortEstimate.HOURS_1,
                ExpectedGain.MEDIUM,
            ),
            (
                "training trick",
                local.training_tricks,
                remote.training_tricks,
                EffortEstimate.MINUTES_20,
                ExpectedGain.LOW,
            ),
            (
                "architecture",
                local.architecture,
                remote.architecture,
                EffortEstimate.HOURS_4,
                ExpectedGain.MEDIUM,
            ),
        )
        output: list[TransferOpportunity] = []
        for label, local_values, remote_values, effort, gain in groups:
            local_keys = {_norm(value) for value in local_values}
            novel = [value for value in remote_values if _norm(value) not in local_keys]
            if not novel:
                continue
            choice = novel[0]
            baseline = ", ".join(local_values[:2]) or "not detected locally"
            summary = f"Uses {choice} instead of {baseline}"
            output.append(
                TransferOpportunity(
                    repo_id=remote.repo_id,
                    summary=summary,
                    deltas=[f"{label}: local={baseline}; remote={choice}"],
                    local_baseline=baseline,
                    remote_choice=choice,
                    effort=effort,
                    expected_gain=gain,
                    interesting_files=remote.interesting_files[:5],
                    hypothesis_hint=f"Test {choice} from {remote.full_name} against {baseline}.",
                )
            )
        return output[:3]


def _norm(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())
