"""Manual PufferLib training launcher."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
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
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        args = pufferl.load_config(config.env_name)
    finally:
        sys.argv = original_argv
    if not isinstance(args, dict):
        raise TypeError("pufferl.load_config returned an unexpected non-dict value.")

    args = merge_config(args, config.puffer_overrides)
    args.setdefault("checkpoint_dir", str(config.output_dir / "checkpoints"))
    args.setdefault("log_dir", str(config.output_dir / "logs"))
    args.setdefault("train", {})
    args["train"].setdefault("data_dir", str(config.output_dir / "pufferlib"))
    return args


def ensure_output_dirs(args: ConfigDict) -> None:
    for path in output_paths(args):
        path.mkdir(parents=True, exist_ok=True)


def output_paths(args: ConfigDict) -> list[Path]:
    paths: list[Path] = []
    for key in ("checkpoint_dir", "log_dir"):
        value = args.get(key)
        if isinstance(value, str) and value:
            paths.append(Path(value))
    train = args.get("train", {})
    if isinstance(train, dict) and isinstance(train.get("data_dir"), str):
        paths.append(Path(train["data_dir"]))
    return paths


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
    logs = pufferl.train(config.env_name, args=args)
    if isinstance(logs, list) and logs:
        write_returned_logs(config, args, logs)
    return 0


def run_sweep(
    config_path: Path,
    max_runs: int | None = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> int:
    config = load_run_config(config_path)

    if dry_run:
        preview = {
            "env_name": config.env_name,
            "output_dir": str(config.output_dir),
            "max_runs": max_runs,
            "puffer_overrides": config.puffer_overrides,
        }
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    with _maybe_suppress_output(quiet):
        try:
            from pufferlib import pufferl
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PufferLib is not installed. Install PufferLib 4.0 from the current "
                "PufferTank/PufferLib source workflow before running sweeps."
            ) from exc

        args = build_puffer_args(config, pufferl)
        args.setdefault("sweep", {})
        if max_runs is not None:
            args["sweep"]["max_runs"] = max_runs
        args["sweep"].setdefault("gpus", 1)
        args.setdefault("train", {})
        args["train"].setdefault("gpus", 1)
        ensure_output_dirs(args)
        pufferl.sweep(config.env_name, args=args)
    return 0


@contextmanager
def _maybe_suppress_output(enabled: bool):
    if not enabled:
        yield
        return

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with Path(os.devnull).open("w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def write_returned_logs(config: RunConfig, args: ConfigDict, logs: Any) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = config.output_dir / "logs" / config.env_name / f"{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "env_name": config.env_name,
        "config": _json_safe(args),
        "metrics": _logs_to_metric_series(logs),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _logs_to_metric_series(logs: Any) -> ConfigDict:
    if not isinstance(logs, list):
        return {}

    series: ConfigDict = {}
    for row in logs:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if isinstance(value, int | float | str | bool) or value is None:
                series.setdefault(str(key), []).append(value)
    return series


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return repr(value)
