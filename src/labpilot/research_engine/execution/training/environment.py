"""How a generated training script gets the libraries it asks for.

Generated code is the one place in this system where the set of needed
dependencies is genuinely open. Measured on rogii 2026-08-07: the LLM wrote
``import catboost`` — a sound choice for tabular regression — and every run died
at line 19 because catboost is not a labpilot dependency. Eight consecutive
executions failed identically, twenty campaign steps produced zero evidence
cards, and nothing in the prompt had ever told the model what was installed.

The obvious fix, telling the model what it may use, was rejected: an allowlist
answers an open-world question, needs an owner, goes stale, and would have
excluded exactly the libraries worth trying. It also would not have made the
system *safe*, because a curated list is always stale against a package that
turns hostile after it was added.

So the script declares its own dependencies via PEP 723 inline metadata and runs
in an ephemeral environment:

    # /// script
    # dependencies = ["lightgbm>=4.0", "catboost"]
    # ///

`uv run --script` resolves that into a throwaway venv — measured at 144ms for
catboost plus 17 transitive packages on a warm cache — so the deps travel with
the artifact, the run is reproducible from `train.py` alone, and nothing
accumulates in labpilot's own environment.

**What contains the risk is isolation, not classification.** A package the model
named is untrusted code; the answer is to bound what it can reach rather than to
predict which names are safe. This module removes credentials from the child
environment and keeps it scoped to the run directory. Network isolation is *not*
implemented — there is no portable way to do it from a parent process, and
claiming otherwise would be worse than saying so. See `docs/` backlog: it needs
OS-level support (a sandbox profile on macOS, a namespace on Linux) or a
container, and until then a training run can reach the network.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

#: PEP 723 opening fence. The block must start at the beginning of a line.
_SCRIPT_BLOCK = re.compile(r"^# /// script\s*$", re.MULTILINE)

#: Dependency entries inside the block, e.g. ``#   "catboost>=1.2",``.
#: Both quote styles: PEP 723 metadata is TOML, single quotes are valid there,
#: and models emit them. `uv` installs either way, so a double-quote-only parser
#: would report no dependencies for a script that has them.
_DEP_ENTRY = re.compile(r"""^#\s*["']([^"']+)["']\s*,?\s*$""", re.MULTILINE)

#: Environment variables never passed to generated code. Prefix matching, because
#: provider keys arrive under names this list cannot enumerate ahead of time —
#: the same open-world problem as package names, handled the same way.
_SECRET_PREFIXES = (
    "OPENAI",
    "OPENROUTER",
    "ANTHROPIC",
    "GROQ",
    "GEMINI",
    "GOOGLE",
    "NVIDIA",
    "HF_",
    "HUGGINGFACE",
    "AWS",
    "AZURE",
    "GCP",
    "KAGGLE",
    "GITHUB",
)

_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "APIKEY")


def declares_dependencies(script: Path | str) -> bool:
    """Whether this script carries a PEP 723 block."""
    text = script.read_text(encoding="utf-8") if isinstance(script, Path) else str(script)
    return bool(_SCRIPT_BLOCK.search(text))


def declared_dependencies(script: Path | str) -> list[str]:
    """Dependency specifiers the script declares, in order.

    Currently used for logging only. Writing these onto the evidence card — so a
    result can be tied to the environment that produced it — is deferred, and
    saying otherwise here would send the next reader hunting for a write path
    that does not exist.
    """
    text = script.read_text(encoding="utf-8") if isinstance(script, Path) else str(script)
    match = _SCRIPT_BLOCK.search(text)
    if not match:
        return []
    end = text.find("# ///", match.end())
    block = text[match.end() : end if end != -1 else len(text)]
    return [m.group(1).strip() for m in _DEP_ENTRY.finditer(block)]


def uv_available() -> bool:
    return shutil.which("uv") is not None


def training_command(script: Path, *, python: str) -> list[str]:
    """Argv for running ``script``.

    Uses ``uv run --script`` only when the script actually declares dependencies
    *and* uv is present. A script with no PEP 723 block gains nothing from an
    ephemeral env and would lose access to labpilot's own environment, so the
    existing interpreter stays the default — which also keeps every template
    that predates this change working unchanged.
    """
    if declares_dependencies(script) and uv_available():
        return ["uv", "run", "--script", str(script)]
    if declares_dependencies(script):
        logger.warning(
            "%s declares dependencies but uv is not on PATH; running with the "
            "current interpreter, which will fail if anything declared is missing",
            script.name,
        )
    return [python, str(script)]


def is_secret_env(name: str) -> bool:
    upper = name.upper()
    return upper.startswith(_SECRET_PREFIXES) or any(m in upper for m in _SECRET_MARKERS)


def child_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the training subprocess, with credentials removed.

    Generated code is untrusted: it was written by a model, may pull packages
    nobody has reviewed, and has no business holding the operator's provider
    keys or Kaggle credentials. Stripping them costs nothing — a training script
    needs data on disk, not API access — and bounds what a hostile dependency
    can exfiltrate even though it cannot stop it reaching the network.
    """
    source = dict(os.environ if base is None else base)
    return {k: v for k, v in source.items() if not is_secret_env(k)}
