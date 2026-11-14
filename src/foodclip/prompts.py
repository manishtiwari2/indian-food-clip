"""Prompt templates for CLIP zero-shot classification.

Languages (columns of data/class_names.csv):
  english          English description of the dish   ("spiced flattened rice")
  hindi_roman      dish name in Latin script          ("poha")
  hindi_devanagari dish name in Devanagari            ("पोहा")
"""
from __future__ import annotations

from typing import Sequence

TEMPLATES: dict[str, list[str]] = {
    "plain": ["{}"],
    "food": ["a photo of {}, a type of food."],          # same template used for Food-101 (fair bias test)
    "indian": ["a photo of {}, a type of Indian food."],
    "ensemble": [
        "a photo of {}, a type of food.",
        "a photo of {}, a type of Indian food.",
        "a close-up photo of {}.",
        "a photo of a plate of {}.",
        "a photo of homemade {}.",
        "a photo of {} served in a restaurant.",
    ],
    # Fully Hindi sentence; only meaningful with Devanagari names and a multilingual text encoder.
    "hindi_native": ["{} की एक तस्वीर", "{}, एक भारतीय व्यंजन"],
}

LANGUAGES = ("english", "hindi_roman", "hindi_devanagari")


def build_prompts(names: Sequence[str], template: str) -> list[list[str]]:
    """One list of prompt strings per class."""
    if template not in TEMPLATES:
        raise KeyError(f"unknown template {template!r}; choose from {sorted(TEMPLATES)}")
    return [[t.format(n) for t in TEMPLATES[template]] for n in names]


def valid_combo(language: str, template: str) -> bool:
    """hindi_native templates only make sense with Devanagari names."""
    return template != "hindi_native" or language == "hindi_devanagari"


def food101_name(label: str) -> str:
    return label.replace("_", " ")
