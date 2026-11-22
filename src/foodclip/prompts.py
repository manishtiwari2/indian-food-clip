"""Prompt templates for CLIP zero-shot classification.

Languages (columns of data/class_names.csv):
  english          English description of the dish   ("spiced flattened rice")
  hindi_roman      dish name in Latin script          ("poha")
  hindi_devanagari dish name in Devanagari            ("पोहा")

Used by E2-E4 (scripts/run_zeroshot.py: languages, templates and the Indian vs Western bias test)
and E7 (scripts/finetune_clip.py builds the text embeddings that start the classifier head).
Report section: Methods, "CLIP zero-shot and prompt engineering".
"""
from __future__ import annotations

from typing import Sequence

# Template name -> list of sentences; "{}" is replaced by the class name. With more than one sentence
# (e.g. "ensemble"), the text embeddings of all sentences are averaged into one class embedding.
# Dict keyed by template name; each value is a list of sentence patterns.
TEMPLATES: dict[str, list[str]] = {
    "plain": ["{}"],  # the class name alone, no sentence
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

# The three name columns compared in E3.
# Tuple of column names in class_names.csv, in the order the languages are compared.
LANGUAGES = ("english", "hindi_roman", "hindi_devanagari")


def build_prompts(names: Sequence[str], template: str) -> list[list[str]]:
    """One list of prompt strings per class.

    Input: class names (in one language) and a template name from TEMPLATES. Output: a list with one
    inner list per class; each inner list has one filled sentence per template sentence, e.g.
    "plain" gives [["poha"], ...]. Why: CLIP's text encoder reads sentences, and the wording changes
    accuracy (that is experiment E4). Used by: scripts/run_zeroshot.py and scripts/finetune_clip.py."""
    # Stop early with a clear message if the template name is wrong.
    # Dict membership test on the template names.
    if template not in TEMPLATES:
        raise KeyError(f"unknown template {template!r}; choose from {sorted(TEMPLATES)}")
    # For every class name, fill it into every sentence of the template (str.format replaces "{}" with the name).
    return [[t.format(n) for t in TEMPLATES[template]] for n in names]


def valid_combo(language: str, template: str) -> bool:
    """hindi_native templates only make sense with Devanagari names.

    Returns False for the combination (Latin or English name inside a Hindi sentence), True otherwise,
    so run_zeroshot.py can skip meaningless language/template pairs."""
    # Only the hindi_native template is restricted; every other template works with every language.
    return template != "hindi_native" or language == "hindi_devanagari"


def food101_name(label: str) -> str:
    """Food-101 folder label in ("apple_pie"), readable class name for prompts out ("apple pie").

    Why: Food-101 folders use underscores, but CLIP was trained on normal text, so spaces read better.
    Used by: scripts/run_zeroshot.py for the Western side of the bias test."""
    # Replace every underscore with a space.
    return label.replace("_", " ")
