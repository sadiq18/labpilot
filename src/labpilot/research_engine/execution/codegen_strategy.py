"""Which codegen path a run takes — read in one place.

`CodeEngineeringCapability` chooses between the delta path and the whole-file
path from `constraints["codegen_strategy"]`, and every caller that builds a
`TaskContext` has to supply it. That has now been the same bug three times on
PR #118: the capability's own fallback pinned `"whole_file"` after the default
moved, `research resume` never set the key at all, and the Conductor's
specialist path did not either. Each was found separately and fixed separately.

So the reader lives here, once, and the callers import it. A fourth call site
that forgets is caught by `test_every_task_context_sets_the_codegen_strategy`
rather than by a campaign quietly regenerating whole files for a week.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_codegen_strategy(config_path: Path | None) -> str:
    """`codegen.strategy` from a workspace config, or `CodegenConfig`'s default.

    Never raises: a campaign should not fail to produce code because a config
    file has a typo in an unrelated section. The fallback follows
    `CodegenConfig`'s own default rather than pinning a literal separately —
    two places naming a default is how they drift, and M19 §3 moved it.
    """
    from labpilot.config import CodegenConfig

    default = str(CodegenConfig().strategy)
    if config_path is None:
        return default
    try:
        from labpilot.config import load_config

        return str(load_config(config_path).codegen.strategy)
    except Exception as exc:  # noqa: BLE001 — config trouble must not stop a run
        logger.debug("codegen strategy unreadable, using %s: %s", default, exc)
        return default


def workspace_config_path(workspace: object) -> Path | None:
    """`configs/default.yaml` under a workspace, or None without one."""
    root = getattr(workspace, "root", None)
    return None if root is None else Path(root) / "configs" / "default.yaml"
