"""Manual PufferLib training launcher."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ConfigDict = dict[str, Any]


@dataclass(frozen=True)
class RunConfig:
    env_name: str
    output_dir: Path
    puffer_overrides: ConfigDict


def load_run_config(path: Path) -> RunConfig:
    raw = _load_mapping_file(path)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    env_name = raw.get("env_name")
    if not isinstance(env_name, str) or not env_name:
        raise ValueError("Config must set a non-empty string `env_name`.")

    output_dir = Path(raw.get("output_dir", "runs"))
    puffer_overrides = raw.get("puffer", {})
    if not isinstance(puffer_overrides, dict):
        raise ValueError("Config key `puffer` must be a mapping when provided.")

    return RunConfig(
        env_name=env_name,
        output_dir=output_dir,
        puffer_overrides=puffer_overrides,
    )


def _load_mapping_file(path: Path) -> ConfigDict:
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_yaml(path)

    with path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return raw


def _load_simple_yaml(path: Path) -> ConfigDict:
    """Parse the tiny YAML subset used by configs/base.yaml when PyYAML is absent."""

    root: ConfigDict = {}
    stack: list[tuple[int, ConfigDict]] = [(-1, root)]

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"Unsupported YAML indentation at {path}:{line_number}")

        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line at {path}:{line_number}")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Empty YAML key at {path}:{line_number}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if raw_value == "":
            child: ConfigDict = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(raw_value)

    return root


def _parse_scalar(raw_value: str) -> str | int | float | bool | None:
    value = raw_value.strip("'\"")
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def merge_config(base: ConfigDict, overrides: ConfigDict) -> ConfigDict:
    merged = deepcopy(base)
    _deep_update(merged, overrides)
    return merged


def _deep_update(target: ConfigDict, update: ConfigDict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def build_puffer_args(config: RunConfig, pufferl: Any) -> ConfigDict:
    args = pufferl.load_config(config.env_name)
    if not isinstance(args, dict):
        raise TypeError("pufferl.load_config returned an unexpected non-dict value.")

    args = merge_config(args, config.puffer_overrides)
    args.setdefault("checkpoint_dir", str(config.output_dir / "checkpoints"))
    args.setdefault("log_dir", str(config.output_dir / "logs"))
    return args


def ensure_output_dirs(args: ConfigDict) -> None:
    for key in ("checkpoint_dir", "log_dir"):
        value = args.get(key)
        if isinstance(value, str) and value:
            Path(value).mkdir(parents=True, exist_ok=True)


def run_training(config_path: Path, dry_run: bool = False) -> int:
    config = load_run_config(config_path)

    if dry_run:
        preview = {
            "env_name": config.env_name,
            "output_dir": str(config.output_dir),
            "puffer_overrides": config.puffer_overrides,
        }
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    try:
        from pufferlib import pufferl
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PufferLib is not installed. Install PufferLib 4.0 from the current "
            "PufferTank/PufferLib source workflow before running training."
        ) from exc

    args = build_puffer_args(config, pufferl)
    ensure_output_dirs(args)
    pufferl.train(config.env_name, args=args)
    return 0
