#!/usr/bin/env bash
# Sync LabPilot docs into your local Obsidian vault (outside the git repo).
#
# Usage:
#   ./scripts/sync-obsidian-vault.sh
#   ./scripts/sync-obsidian-vault.sh /path/to/labpilot
#
# Default vault:
#   ~/Documents/Obsidian Vault/autonomous-research-engineer
#
# Auto-run via git hook:
#   ./scripts/install-obsidian-sync-hook.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ge 1 ]]; then
  REPO_ROOT="$(cd "$1" && pwd)"
fi

VAULT="${OBSIDIAN_VAULT:-$HOME/Documents/Obsidian Vault/autonomous-research-engineer}"
QUIET="${OBSIDIAN_SYNC_QUIET:-0}"

log() {
  if [[ "$QUIET" != "1" ]]; then
    echo "$@"
  fi
}

mkdir -p "$VAULT"/{milestones,architecture,plans}

cp "$REPO_ROOT/docs/MILESTONES.md" "$VAULT/milestones/"
cp "$REPO_ROOT/docs/milestones/"*.md "$VAULT/milestones/"
cp "$REPO_ROOT/docs/ARCHITECTURE.md" "$VAULT/architecture/"

if [[ -d "$REPO_ROOT/docs/plans" ]]; then
  cp "$REPO_ROOT/docs/plans/"*.md "$VAULT/plans/" 2>/dev/null || true
fi

cat > "$VAULT/LabPilot.md" <<'EOF'
# LabPilot

Obsidian reference for the **autonomous-research-engineer** project (LabPilot repo).

## Milestones

- [[milestones/MILESTONES]]
- [[milestones/COMPLETED]]
- [[milestones/TODO]]
- [[milestones/backlog]]

## Architecture

- [[architecture/ARCHITECTURE]]

## Plans

See `plans/` (sync from repo when `docs/plans/` exists).
EOF

cat > "$VAULT/README.md" <<EOF
# Autonomous Research Engineer (LabPilot)

Auto-synced from \`$(basename "$REPO_ROOT")\` at $(date -u +"%Y-%m-%d %H:%M UTC").

Vault path: \`$VAULT\`

Re-run manually: \`$REPO_ROOT/scripts/sync-obsidian-vault.sh\`

Enable auto-sync on \`git pull\`: \`$REPO_ROOT/scripts/install-obsidian-sync-hook.sh\`
EOF

log "Synced LabPilot docs to: $VAULT"
log "Open in Obsidian: Obsidian → Open folder as vault → select that path"
