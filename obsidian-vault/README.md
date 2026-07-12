# Obsidian vault — LabPilot

Open this folder as an Obsidian vault:

1. Obsidian → **Open folder as vault**
2. Select `obsidian-vault/` (this directory)
3. Start from **[[LabPilot]]** (home note)

## Contents

| Folder | Source |
|--------|--------|
| `plans/` | Cursor agent plans (`/opt/cursor/artifacts/plans/`) |
| `milestones/` | `docs/MILESTONES.md` + `docs/milestones/*` |
| `architecture/` | `docs/ARCHITECTURE.md` |

To refresh from the repo after doc updates:

```bash
cp docs/MILESTONES.md obsidian-vault/milestones/
cp docs/milestones/*.md obsidian-vault/milestones/
cp docs/ARCHITECTURE.md obsidian-vault/architecture/
```

Plans are copied manually when new agent plans are created.
