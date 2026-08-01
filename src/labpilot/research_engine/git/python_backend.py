"""GitPython-backed GitTool with subprocess fallback for advanced commands."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence

from git import Actor, Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from labpilot.research_engine.git.models import (
    BranchInfo,
    CommitSnapshot,
    DiffSummary,
    GitLogEntry,
    GitStatus,
)

logger = logging.getLogger(__name__)

# Default code roots — never stage knowledge/runs/artifacts by default.
DEFAULT_CODE_PATHS: tuple[str, ...] = ("pipeline", "src", "configs", "tests")

_LABPILOT_ACTOR = Actor("LabPilot", "labpilot@localhost")


class GitPythonTool:
    """GitTool implementation using GitPython (+ CLI ``execute`` escape hatch)."""

    def __init__(self, root: Path, *, code_paths: Sequence[str] = DEFAULT_CODE_PATHS) -> None:
        self.root = Path(root).resolve()
        self.code_paths = tuple(code_paths)
        self._repo = self._open_or_init(self.root)

    @staticmethod
    def _open_or_init(root: Path) -> Repo:
        root.mkdir(parents=True, exist_ok=True)
        try:
            return Repo(root)
        except InvalidGitRepositoryError:
            repo = Repo.init(root)
            with repo.config_writer() as cw:
                cw.set_value("user", "name", "LabPilot")
                cw.set_value("user", "email", "labpilot@localhost")
            repo.git.commit("--allow-empty", "-m", "labpilot: bootstrap research workspace")
            return repo

    def status(self) -> GitStatus:
        repo = self._repo
        branch = None
        try:
            branch = repo.active_branch.name
        except TypeError:
            branch = repo.head.commit.hexsha[:7] if repo.head.is_valid() else None
        staged = [i.a_path for i in repo.index.diff("HEAD")]
        unstaged = [i.a_path for i in repo.index.diff(None)]
        untracked = list(repo.untracked_files)
        clean = not staged and not unstaged and not untracked
        return GitStatus(
            branch=branch,
            clean=clean,
            staged=[p for p in staged if p],
            unstaged=[p for p in unstaged if p],
            untracked=untracked,
        )

    def diff(self, *, staged: bool = False) -> DiffSummary:
        repo = self._repo
        if staged:
            diff_index = repo.index.diff("HEAD", create_patch=True)
        else:
            diff_index = repo.index.diff(None, create_patch=True)
        files: list[str] = []
        chunks: list[str] = []
        for item in diff_index:
            path = item.a_path or item.b_path
            if path:
                files.append(path)
            try:
                chunks.append(item.diff.decode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                continue
        return DiffSummary(files_changed=files, text="\n".join(chunks))

    def commit(
        self,
        message: str,
        *,
        paths: Sequence[str] | None = None,
    ) -> CommitSnapshot | None:
        repo = self._repo
        use_paths = list(paths) if paths is not None else list(self.code_paths)
        existing = [p for p in use_paths if (self.root / p).exists()]
        if existing:
            repo.index.add(existing)

        staged = list(repo.index.diff("HEAD"))
        if not staged:
            return self.get_commit("HEAD")

        try:
            commit = repo.index.commit(
                message,
                author=_LABPILOT_ACTOR,
                committer=_LABPILOT_ACTOR,
                skip_hooks=True,
            )
        except GitCommandError as exc:
            logger.debug("git commit failed: %s", exc)
            return self.get_commit("HEAD")

        files = list(commit.stats.files.keys())
        branch = None
        try:
            branch = repo.active_branch.name
        except TypeError:
            branch = None
        return CommitSnapshot(
            commit=commit.hexsha,
            message=message,
            files_changed=files,
            branch=branch,
        )

    def checkout(self, ref: str, *, paths: Sequence[str] | None = None) -> None:
        repo = self._repo
        if paths:
            existing = [p for p in paths if (self.root / p).exists()] or list(paths)
            repo.git.checkout(ref, "--", *existing)
            return
        repo.git.checkout(ref)

    def create_branch(self, name: str, *, checkout: bool = True) -> BranchInfo:
        repo = self._repo
        # Force-create/reset branch tip like checkout -B.
        branch = repo.create_head(name, force=True)
        if checkout:
            branch.checkout()
        commit = branch.commit.hexsha if branch.commit else None
        return BranchInfo(name=name, commit=commit)

    def log(self, *, max_count: int = 20) -> list[GitLogEntry]:
        repo = self._repo
        entries: list[GitLogEntry] = []
        for commit in repo.iter_commits(max_count=max_count):
            entries.append(
                GitLogEntry(
                    commit=commit.hexsha,
                    message=(commit.message or "").strip(),
                    author=str(commit.author),
                )
            )
        return entries

    def get_commit(self, commit_id: str = "HEAD") -> CommitSnapshot | None:
        repo = self._repo
        try:
            commit = repo.commit(commit_id)
        except (GitCommandError, ValueError):
            return None
        files = list(commit.stats.files.keys()) if commit.parents else []
        branch = None
        try:
            branch = repo.active_branch.name
        except TypeError:
            branch = None
        return CommitSnapshot(
            commit=commit.hexsha,
            message=(commit.message or "").strip(),
            files_changed=files,
            branch=branch,
        )

    def execute(self, *git_args: str) -> str:
        """Raw git CLI fallback for operations not modeled on the port."""
        result = subprocess.run(
            ["git", *git_args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        out = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            raise RuntimeError(out.strip() or f"git {' '.join(git_args)} failed")
        return out.strip()


def open_git_tool(root: Path, *, code_paths: Sequence[str] = DEFAULT_CODE_PATHS) -> GitPythonTool:
    """Factory for the default GitTool backend."""
    return GitPythonTool(root, code_paths=code_paths)
