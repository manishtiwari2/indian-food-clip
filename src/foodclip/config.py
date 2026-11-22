"""Load configs/default.yaml and apply `key.sub=value` overrides from the CLI.

Used by every script, so by all experiments E1-E7. All paths, seeds and hyperparameters live in the YAML
file, not in code. Report: the experimental setup described in every Methods section comes from this config.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

# Repo root = two folders above src/foodclip/config.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _set_dotted(cfg: dict, dotted: str, raw: str) -> None:
    """Set one nested config value, e.g. dotted="cnn.epochs", raw="5" -> cfg["cnn"]["epochs"] = 5."""
    # Walk down the nested dicts, creating missing levels on the way.
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    # Parse the text as YAML so numbers and lists get the right Python type.
    node[keys[-1]] = yaml.safe_load(raw)  # "5" -> 5, "[1,2]" -> list, "x" -> "x"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """YAML path (default configs/default.yaml) and "key.sub=value" overrides in; config dict out."""
    # Read the YAML file into a dict and copy it, so changes never touch a shared object.
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    # Apply each command-line override; split only at the first "=".
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_dotted(cfg, key, value)
    return cfg


def resolve(path: str | Path) -> Path:
    """Relative paths are taken from the repo root, so scripts work from any cwd."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def base_parser(description: str) -> argparse.ArgumentParser:
    """Script description in; argument parser with the shared --config and --set options out."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    # --set takes any number of KEY=VALUE items, e.g. --set cnn.epochs=5 seed=1
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Parsed command-line args in; final config dict (file + overrides) out."""
    return load_config(args.config, args.set)
