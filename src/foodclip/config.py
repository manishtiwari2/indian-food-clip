"""Load configs/default.yaml and apply `key.sub=value` overrides from the CLI."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _set_dotted(cfg: dict, dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw)  # "5" -> 5, "[1,2]" -> list, "x" -> "x"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
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
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, args.set)
