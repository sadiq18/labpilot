#!/usr/bin/env bash
# Enable LabPilot git hooks that sync docs to your local Obsidian vault.
#
# Usage (from repo root):
#   ./scripts/install-obsidian-sync-hook.sh
#   ./scripts/install-obsidian-sync-hook.sh --uninstall
#
# After install, the vault updates automatically on:
#   - git pull   (post-merge)
#   - git checkout <branch>  (post-checkout)
#
# Skip once:
#   LABPILOT_SKIP_OBSIDIAN_SYNC=1 git pull
#
# Custom vault path:
#   export OBSIDIAN_VAULT="$HOME/Documents/Obsidian Vault/autonomous-research-engineer"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$ROOT/scripts/githooks"

usage() {
  sed -n '2,18p' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--uninstall" ]]; then
  cd "$ROOT"
  git config --unset core.hooksPath 2>/dev/null || true
  echo "Removed core.hooksPath (Obsidian sync hooks disabled for this clone)."
  exit 0
fi

chmod +x "$ROOT/scripts/sync-obsidian-vault.sh"
chmod +x "$HOOKS_DIR"/post-merge "$HOOKS_DIR"/post-checkout

cd "$ROOT"
git config core.hooksPath scripts/githooks

echo "Obsidian sync hook installed for: $(git rev-parse --show-toplevel)"
echo "  hooks path: scripts/githooks"
echo "  vault:      ${OBSIDIAN_VAULT:-\$HOME/Documents/Obsidian Vault/autonomous-research-engineer}"
echo ""
echo "Running initial sync..."
"$ROOT/scripts/sync-obsidian-vault.sh" "$ROOT"
