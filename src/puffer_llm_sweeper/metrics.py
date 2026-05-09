"""PufferLib JSON log parsing and summary generation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    env_name: str | None
    config: JsonDict
    final_reward: float | None
    best_reward: float | None
    training_steps: float | None
    wall_time: float | None
    sps: float | None
    failure: bool
    failure_reason: str | None
    ppo_metrics: JsonDict

    def to_dict(self) -> JsonDict:
        return asdict(self)


def summarize_logs(logs_dir: Path) -> list[RunSummary]:
    if not logs_dir.exists():
        raise FileNotFoundError(f"Runs log directory does not exist: {logs_dir}")

    summaries: list[RunSummary] = []
    for path in sorted(logs_dir.rglob("*.json")):
        summaries.append(parse_log(path))
    return summaries


def parse_log(path: Path) -> RunSummary:
    with path.open("r", encoding="utf-8") as log_file:
        raw = json.load(log_file)
    if not isinstance(raw, dict):
        raise ValueError(f"PufferLib log must be a JSON object: {path}")

    metrics = raw.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    env_name = _extract_env_name(raw)
    rewards = _reward_series(raw, metrics)
    final_reward = _last_finite(rewards)
    best_reward = max(rewards) if rewards else None

    training_steps = _last_finite(_number_series(metrics.get("agent_steps")))
    wall_time = _last_finite(_number_series(metrics.get("uptime")))
    sps = _last_finite(_number_series(metrics.get("SPS")))
    ppo_metrics = _final_prefixed_metrics(
        metrics,
        prefixes=("loss/", "losses/", "perf/", "performance/"),
    )

    failure_reason = _failure_reason(final_reward, metrics)
    return RunSummary(
        run_id=path.stem,
        env_name=env_name,
        config=_config_snapshot(raw),
        final_reward=final_reward,
        best_reward=best_reward,
        training_steps=training_steps,
        wall_time=wall_time,
        sps=sps,
        failure=failure_reason is not None,
        failure_reason=failure_reason,
        ppo_metrics=ppo_metrics,
    )


def write_summary(summaries: list[RunSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [summary.to_dict() for summary in summaries],
        "num_runs": len(summaries),
        "num_failures": sum(1 for summary in summaries if summary.failure),
        "best_reward": _best_reward(summaries),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_env_name(raw: JsonDict) -> str | None:
    base = raw.get("base")
    if isinstance(base, dict) and isinstance(base.get("env_name"), str):
        return base["env_name"]
    value = raw.get("env_name")
    return value if isinstance(value, str) else None


def _target_metric_name(raw: JsonDict) -> str:
    sweep = raw.get("sweep")
    metric = sweep.get("metric") if isinstance(sweep, dict) else None
    if isinstance(metric, str) and metric:
        return f"env/{metric}"
    return "env/score"


def _reward_series(raw: JsonDict, metrics: JsonDict) -> list[float]:
    candidates = [
        "env/score",
        "environment/score",
        "environment/reward",
        "env/episode_length",
        _target_metric_name(raw),
    ]
    for name in candidates:
        rewards = _number_series(metrics.get(name))
        if rewards:
            return rewards
    return []


def _number_series(value: Any) -> list[float]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]

    numbers: list[float] = []
    for item in raw_values:
        if isinstance(item, int | float):
            number = float(item)
            if math.isfinite(number):
                numbers.append(number)
    return numbers


def _last_finite(values: list[float]) -> float | None:
    return values[-1] if values else None


def _final_prefixed_metrics(metrics: JsonDict, prefixes: tuple[str, ...]) -> JsonDict:
    selected: JsonDict = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.startswith(prefixes):
            continue
        selected[key] = _last_finite(_number_series(value))
    return selected


def _config_snapshot(raw: JsonDict) -> JsonDict:
    return {key: value for key, value in raw.items() if key != "metrics"}


def _failure_reason(final_reward: float | None, metrics: JsonDict) -> str | None:
    if final_reward is None:
        return "missing_reward_metric"
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.startswith(("loss/", "losses/")):
            continue
        raw_values = value if isinstance(value, list) else [value]
        for item in raw_values:
            if isinstance(item, int | float) and not math.isfinite(float(item)):
                return "non_finite_loss"
    return None


def _best_reward(summaries: list[RunSummary]) -> float | None:
    rewards = [summary.best_reward for summary in summaries if summary.best_reward is not None]
    return max(rewards) if rewards else None
