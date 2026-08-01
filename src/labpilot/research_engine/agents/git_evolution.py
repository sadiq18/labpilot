"""Experiment git orchestration over GitTool (code history, not knowledge)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from labpilot.research_engine.git import (
    CODE_PATHS,
    CommitSnapshot,
    GitTool,
    open_git_tool,
)

_SAFE = re.compile(r"[^A-Za-z0-9._/-]+")


def research_branch_name(session_id: str, experiment_key: str) -> str:
    """Build ``research/<session>/<experiment>`` with safe path segments."""
    session = _SAFE.sub("-", session_id.strip()) or "local"
    key = _SAFE.sub("-", experiment_key.strip()) or "exp"
    return f"research/{session}/{key}"


def short_commit(commit: str | None, *, n: int = 7) -> str | None:
    if not commit:
        return None
    return commit[:n]


def snapshot_before_experiment(
    workspace_root: Path,
    *,
    session_id: str,
    experiment_key: str,
    message: str,
    git: GitTool | None = None,
) -> CommitSnapshot | None:
    """Create research branch + code-only commit via GitTool."""
    tool = git or open_git_tool(workspace_root)
    branch_name = research_branch_name(session_id, experiment_key)
    tool.create_branch(branch_name, checkout=True)
    snapshot = tool.commit(message, paths=CODE_PATHS)
    if snapshot is None:
        # Still return HEAD so experiments record an anchor hash.
        head = tool.get_commit("HEAD")
        if head is None:
            return None
        return head.model_copy(update={"branch": branch_name, "message": message})
    return snapshot.model_copy(update={"branch": branch_name})


def revert_to_commit(
    repo: Path,
    commit: str,
    *,
    paths: Sequence[str] = CODE_PATHS,
    git: GitTool | None = None,
) -> None:
    """Restore code paths to ``commit`` through GitTool."""
    tool = git or open_git_tool(repo)
    tool.checkout(commit, paths=list(paths))


def experiment_records_dir(workspace_root: Path) -> Path:
    return Path(workspace_root) / "experiment" / "by_id"


def write_experiment_git_record(
    workspace_root: Path,
    payload: dict[str, Any],
) -> Path:
    """Write latest record.json and indexed copies under experiment/by_id/."""
    root = Path(workspace_root)
    exp_dir = root / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    latest = exp_dir / "record.json"
    latest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    keys: list[str] = []
    for key in (
        payload.get("execution_id"),
        payload.get("experiment_id"),
        *(payload.get("aliases") or []),
    ):
        text = str(key or "").strip()
        if text and text not in keys:
            keys.append(text)

    indexed_dir = experiment_records_dir(root)
    indexed_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2) + "\n"
    for key in keys:
        (indexed_dir / f"{key}.json").write_text(body, encoding="utf-8")
    return latest


def find_experiment_record(
    workspace_root: Path,
    experiment_id: str,
) -> dict[str, Any] | None:
    """Lookup experiment record by execution id, experiment id, or alias."""
    root = Path(workspace_root)
    needle = experiment_id.strip()
    direct = experiment_records_dir(root) / f"{needle}.json"
    if direct.is_file():
        try:
            data = json.loads(direct.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            return data

    latest = root / "experiment" / "record.json"
    if latest.is_file():
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict) and _record_matches(data, needle):
            return data

    indexed = experiment_records_dir(root)
    if indexed.is_dir():
        for path in sorted(indexed.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and (
                path.stem == needle or _record_matches(data, needle)
            ):
                return data
    return None


def _record_matches(data: dict[str, Any], needle: str) -> bool:
    if needle in {
        str(data.get("execution_id") or ""),
        str(data.get("experiment_id") or ""),
    }:
        return True
    aliases = data.get("aliases") or []
    if isinstance(aliases, list) and needle in {str(a) for a in aliases}:
        return True
    branch = str(data.get("git_branch") or "")
    return bool(branch and branch.endswith(f"/{needle}"))
