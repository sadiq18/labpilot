import logging

from labpilot.accessor.profiler.report import load_profile
from labpilot.accessor.profiler.tabular import ColumnProfile, DatasetProfile

logger = logging.getLogger(__name__)

TARGET_ENCODING_MIN_CARDINALITY = 10
LOG_NUMERIC_SKEW_RATIO = 10.0


def suggest_feature_recipes(profile: DatasetProfile) -> tuple[list[str], list[str], list[str]]:
    """Return (recipe_names, target_encoding_columns, log_numeric_columns)."""
    recipes: list[str] = []
    target_encoding_columns = _high_cardinality_categoricals(profile)
    log_numeric_columns = _skewed_numeric_columns(profile)

    if target_encoding_columns:
        recipes.append("target_encoding")
    if log_numeric_columns:
        recipes.append("log_numeric")

    logger.info(
        "Feature recipe suggestions: recipes=%s, target_encoding=%s, log_numeric=%s",
        recipes,
        target_encoding_columns,
        log_numeric_columns,
    )
    return recipes, target_encoding_columns, log_numeric_columns


def apply_recipes_from_profile(
    profile: DatasetProfile,
    requested_recipes: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve recipe names and column lists for a profile."""
    suggested, te_cols, log_cols = suggest_feature_recipes(profile)
    if requested_recipes is None:
        return suggested, te_cols, log_cols

    active: list[str] = []
    if "target_encoding" in requested_recipes and te_cols:
        active.append("target_encoding")
    if "log_numeric" in requested_recipes and log_cols:
        active.append("log_numeric")
    te_result = te_cols if "target_encoding" in active else []
    log_result = log_cols if "log_numeric" in active else []
    return active, te_result, log_result


def _high_cardinality_categoricals(profile: DatasetProfile) -> list[str]:
    if not profile.target_column:
        return []
    columns: list[str] = []
    for column in profile.columns:
        if column.name == profile.target_column:
            continue
        if column.is_numeric:
            continue
        if column.unique_count >= TARGET_ENCODING_MIN_CARDINALITY:
            columns.append(column.name)
    return columns


def _skewed_numeric_columns(profile: DatasetProfile) -> list[str]:
    if not profile.target_column:
        return []
    columns: list[str] = []
    for column in profile.columns:
        if column.name == profile.target_column:
            continue
        if not column.is_numeric:
            continue
        if _looks_skewed(column):
            columns.append(column.name)
    return columns


def _looks_skewed(column: ColumnProfile) -> bool:
    stats = column.stats
    minimum = stats.get("min")
    maximum = stats.get("max")
    mean = stats.get("mean")
    if minimum is None or maximum is None or mean is None:
        return False
    if minimum <= 0:
        return False
    if maximum <= minimum:
        return False
    if mean <= 0:
        return False
    range_ratio = maximum / max(minimum, 1e-9)
    mean_ratio = maximum / mean
    return range_ratio >= LOG_NUMERIC_SKEW_RATIO or mean_ratio >= LOG_NUMERIC_SKEW_RATIO


def recipes_for_run(run_dir) -> tuple[list[str], list[str], list[str]]:
    profile = load_profile(run_dir)
    return suggest_feature_recipes(profile)
