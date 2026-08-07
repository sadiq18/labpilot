"""Feature-engineering recipes extracted from papers / repos / kernels / forums."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

FEATURE_ENGINEERING_CATEGORY = "feature_engineering"

_FE_HINTS = (
    "feature",
    "target encod",
    "target_encod",
    "one-hot",
    "one hot",
    "one_hot",
    "label encod",
    "label_encod",
    "frequency encod",
    "frequency_encod",
    "interaction",
    "polynomial",
    "binning",
    "aggregat",
    "groupby",
    "group by",
    "lag feature",
    "lag_feature",
    "rolling",
    "tf-idf",
    "tfidf",
    "embedding",
    "normalize",
    "standardiz",
    "log1p",
    "feature engineer",
    "feature_engineer",
)


class FeatureRecipe(BaseModel):
    """Concrete new-feature transform (not a vague 'do FE' suggestion)."""

    name: str
    description: str = ""
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    transform: str = ""


def coerce_feature_recipes(value: object) -> list[FeatureRecipe]:
    """Normalize loose JSON / dict lists into ``FeatureRecipe`` objects."""
    if not value:
        return []
    if isinstance(value, FeatureRecipe):
        return [value]
    items = value if isinstance(value, (list, tuple)) else [value]
    recipes: list[FeatureRecipe] = []
    for item in items:
        if isinstance(item, FeatureRecipe):
            recipes.append(item)
            continue
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            recipes.append(
                FeatureRecipe(
                    name=name,
                    description=str(item.get("description") or ""),
                    inputs=[str(x) for x in (item.get("inputs") or []) if str(x).strip()],
                    outputs=[str(x) for x in (item.get("outputs") or []) if str(x).strip()],
                    transform=str(item.get("transform") or ""),
                )
            )
            continue
        text = str(item).strip()
        if text:
            recipes.append(FeatureRecipe(name=text[:80], description=text))
    return recipes


def recipe_technique_names(recipes: list[FeatureRecipe]) -> list[str]:
    """Technique labels derived from recipe names (deduped, order preserved)."""
    names: list[str] = []
    for recipe in recipes:
        label = recipe.name.strip()
        if label and label not in names:
            names.append(label)
    return names


def looks_like_feature_engineering(text: str) -> bool:
    lower = (text or "").lower()
    return any(hint in lower for hint in _FE_HINTS)


def heuristic_feature_recipes(text: str, *, limit: int = 5) -> list[FeatureRecipe]:
    """Light offline FE extraction from free text (paper/repo/forum)."""
    if not text or not looks_like_feature_engineering(text):
        return []
    recipes: list[FeatureRecipe] = []
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?\n])\s+", text)
        if len(s.strip()) >= 24
    ]
    for sentence in sentences:
        if not looks_like_feature_engineering(sentence):
            continue
        name = _name_from_sentence(sentence)
        if not name:
            continue
        if any(r.name.lower() == name.lower() for r in recipes):
            continue
        recipes.append(
            FeatureRecipe(
                name=name,
                description=sentence[:240],
                transform=sentence[:200],
            )
        )
        if len(recipes) >= limit:
            break
    return recipes


def recipes_to_metadata(recipes: list[FeatureRecipe]) -> list[dict[str, Any]]:
    return [r.model_dump(mode="json") for r in recipes]


def merge_feature_recipes(
    *groups: list[FeatureRecipe] | None,
) -> list[FeatureRecipe]:
    merged: list[FeatureRecipe] = []
    seen: set[str] = set()
    for group in groups:
        for recipe in group or []:
            key = recipe.name.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(recipe)
    return merged


def _name_from_sentence(sentence: str) -> str:
    lower = sentence.lower()
    for pattern, label in (
        (r"target\s*encod\w*", "target_encoding"),
        (r"one[-\s]?hot", "one_hot_encoding"),
        (r"label\s*encod\w*", "label_encoding"),
        (r"frequency\s*encod\w*", "frequency_encoding"),
        (r"tf[-\s]?idf", "tfidf"),
        (r"\bbinning\b", "binning"),
        (r"polynomial", "polynomial_features"),
        (r"interaction", "feature_interactions"),
        (r"rolling", "rolling_features"),
        (r"\blag\b", "lag_features"),
        (r"aggregat\w*", "aggregation_features"),
        (r"log1p", "log1p_transform"),
    ):
        if re.search(pattern, lower):
            return label
    # No scraped fallback. The previous one took the first word before
    # "feature" — `\b([A-Za-z][A-Za-z0-9_]{2,40})\s+feature` — which matches any
    # English word of three or more letters. "We added the features to the
    # model" minted a technique called `the`; measured on rogii, ten beliefs
    # existed for `the`, `add`, `built`, `computed`, `average`, `context`,
    # `model`, `neighbour`, `tangent` and `booster`, and the Conductor asked
    # the code engineer to implement `the`.
    #
    # A name we cannot stand behind is worse than no name: it becomes an
    # identity in the ledger, accrues beliefs, and competes for attention with
    # techniques that were actually measured. Per §8.7 the description is the
    # payload and identity is for the ledger, so an unrecognised recipe stays
    # generic and keeps its sentence as the description.
    return "feature_engineering"
