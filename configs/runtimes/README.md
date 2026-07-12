# Runtime registry

LabPilot runtimes describe **where** `train_model` will execute. In v1.0, only
configuration and validation are implemented — training still runs locally via
the built-in `local-default` runtime. Remote dispatch (`--remote-train`) is
deferred to the P2 execution milestone.

## Layout

```
configs/runtimes/
  local-default.yaml     # built-in default (also registered in code)
  my-kaggle-gpu.yaml     # user-registered runtimes
  examples/              # documented samples (not auto-loaded)
```

Project workspaces can override with `project/runtimes/*.yaml`. Later directories
shadow earlier ones by `id`.

## Common fields

| Field | Description |
|-------|-------------|
| `schema_version` | Config schema version (currently `1`) |
| `id` | Stable runtime id for `--runtime` (future) |
| `provider` | `local`, `kaggle_kernel`, `google_colab`, or `other` |
| `enabled` | Soft disable without deleting the file |
| `priority` | Scheduler preference (config-only in v1.0) |
| `labels` | Tags such as `gpu`, `free-tier`, `paid` |
| `artifacts` | Paths to sync back when remote execution lands |
| `poll` | `interval_seconds`, `timeout_seconds` for remote job polling |
| `quotas` | Local quota tracking config (not enforced in v1.0) |

## Providers

### `local`

Runs `pipeline/train.py` as a local subprocess (current default behavior).

### `kaggle_kernel`

Kaggle notebook/kernel runtime for future remote training. Reuses Kaggle
credentials from `.env` / `KaggleConfig`.

### `google_colab`

Google Colab runtime. Doctor checks auth env vars only — no OAuth in v1.0.

### `other`

Extensibility hook via `adapter: module.path:ClassName`. Schema validates;
execution adapters ship with P2.

## CLI

```bash
research runtime list
research runtime show --runtime local-default
research runtime register --provider kaggle_kernel --id kaggle-gpu-free
research runtime doctor
research runtime doctor --runtime colab-pro
```
