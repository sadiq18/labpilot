import itertools
import logging
from typing import Any

from labpilot.improvement.models import DEFAULT_TABULAR_MODEL_PARAMS

logger = logging.getLogger(__name__)

# Small grid (≤12 combos) for tabular LightGBM baselines.
TUNE_GRID: dict[str, list[Any]] = {
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 63],
    "n_estimators": [200, 300],
}


def default_tabular_params(*, random_seed: int) -> dict[str, Any]:
    params = dict(DEFAULT_TABULAR_MODEL_PARAMS)
    params["random_state"] = random_seed
    return params


def grid_combinations() -> list[dict[str, Any]]:
    keys = list(TUNE_GRID.keys())
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(TUNE_GRID[key] for key in keys)):
        combos.append(dict(zip(keys, values, strict=True)))
    return combos


def pick_tune_params(
    parent_params: dict[str, Any] | None,
    *,
    random_seed: int,
    combo_index: int | None = None,
) -> dict[str, Any]:
    """Pick the next grid point relative to the parent params."""
    base = default_tabular_params(random_seed=random_seed)
    if parent_params:
        base.update(
            {k: v for k, v in parent_params.items() if k in TUNE_GRID or k == "random_state"}
        )

    combos = grid_combinations()
    if combo_index is not None:
        chosen = combos[combo_index % len(combos)]
    else:
        parent_key = tuple(base.get(k) for k in TUNE_GRID)
        try:
            start = next(
                index
                for index, combo in enumerate(combos)
                if tuple(combo.get(k) for k in TUNE_GRID) == parent_key
            )
        except StopIteration:
            start = -1
        chosen = combos[(start + 1) % len(combos)]

    result = dict(base)
    result.update(chosen)
    result["random_state"] = random_seed
    logger.info("Selected tune params: %s", chosen)
    return result
