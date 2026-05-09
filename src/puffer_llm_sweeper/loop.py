"""Minimal constrained experiment loop controller."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from puffer_llm_sweeper.decisions import LlmDecision, decision_to_json
from puffer_llm_sweeper.metrics import RunSummary, summarize_logs, write_summary
from puffer_llm_sweeper.openrouter import OpenRouterClient
from puffer_llm_sweeper.runner import run_training


@dataclass(frozen=True)
class StopRules:
    max_iterations: int
    max_trials: int
    max_minutes: float
    target_reward: float | None
    no_improvement_iterations: int
    improvement_window: int
    improvement_epsilon: float
    max_failures: int


@dataclass(frozen=True)
class LoopResult:
    iterations: int
    trials: int
    stop_reason: str
    best_reward: float | None
    last_decision: LlmDecision | None


def run_loop(
    config_path: Path,
    logs_dir: Path,
    summary_path: Path,
    decision_path: Path,
    work_config_path: Path,
    rules: StopRules,
    live: bool = False,
    skip_training: bool = False,
) -> LoopResult:
    start_time = time.monotonic()
    trials = 0
    best_history: list[float | None] = []
    last_decision: LlmDecision | None = None
    current_config_path = initialize_work_config(config_path, work_config_path)

    for iteration in range(1, rules.max_iterations + 1):
        pre_stop = evaluate_stop_rules(
            summaries=[],
            best_history=best_history,
            iteration=iteration - 1,
            trials=trials,
            elapsed_minutes=(time.monotonic() - start_time) / 60,
            rules=rules,
        )
        if pre_stop:
            return _result(iteration - 1, trials, pre_stop, best_history, last_decision)

        if not skip_training:
            run_training(current_config_path)
            trials += 1

        summaries = summarize_logs(logs_dir) if logs_dir.exists() else []
        write_summary(summaries, summary_path)
        best_history.append(_best_reward(summaries))

        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        last_decision = OpenRouterClient().propose_decision(summary_payload, dry_run=not live)
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(decision_to_json(last_decision) + "\n", encoding="utf-8")
        apply_decision_to_config(current_config_path, last_decision, work_config_path)
        current_config_path = work_config_path

        post_stop = evaluate_stop_rules(
            summaries=summaries,
            best_history=best_history,
            iteration=iteration,
            trials=trials,
            elapsed_minutes=(time.monotonic() - start_time) / 60,
            rules=rules,
        )
        if post_stop:
            return _result(iteration, trials, post_stop, best_history, last_decision)
        if last_decision.action == "stop":
            return _result(iteration, trials, "llm_stop", best_history, last_decision)

    return _result(rules.max_iterations, trials, "max_iterations", best_history, last_decision)


def initialize_work_config(source_path: Path, work_config_path: Path) -> Path:
    if source_path == work_config_path:
        return source_path
    work_config_path.parent.mkdir(parents=True, exist_ok=True)
    work_config_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return work_config_path


def apply_decision_to_config(
    source_path: Path,
    decision: LlmDecision,
    output_path: Path,
) -> None:
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Loop config must be a mapping: {source_path}")

    puffer = raw.setdefault("puffer", {})
    if not isinstance(puffer, dict):
        raise ValueError("Loop config key `puffer` must be a mapping when provided.")
    train = puffer.setdefault("train", {})
    if not isinstance(train, dict):
        raise ValueError("Loop config key `puffer.train` must be a mapping when provided.")

    for name, range_config in decision.search_space_update.items():
        train[name] = _representative_value(range_config.min, range_config.max, range_config.scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")


def evaluate_stop_rules(
    summaries: list[RunSummary],
    best_history: list[float | None],
    iteration: int,
    trials: int,
    elapsed_minutes: float,
    rules: StopRules,
) -> str | None:
    if iteration >= rules.max_iterations:
        return "max_iterations"
    if trials >= rules.max_trials:
        return "max_trials"
    if elapsed_minutes >= rules.max_minutes:
        return "max_minutes"

    best_reward = _best_from_history(best_history)
    if rules.target_reward is not None and best_reward is not None and best_reward >= rules.target_reward:
        return "target_reward"

    failures = sum(1 for summary in summaries if summary.failure)
    if failures > rules.max_failures:
        return "too_many_failures"

    if _no_improvement(best_history, rules.no_improvement_iterations):
        return "no_improvement"
    if _low_improvement(best_history, rules.improvement_window, rules.improvement_epsilon):
        return "low_improvement"

    return None


def _result(
    iterations: int,
    trials: int,
    stop_reason: str,
    best_history: list[float | None],
    last_decision: LlmDecision | None,
) -> LoopResult:
    return LoopResult(
        iterations=iterations,
        trials=trials,
        stop_reason=stop_reason,
        best_reward=_best_from_history(best_history),
        last_decision=last_decision,
    )


def _best_reward(summaries: list[RunSummary]) -> float | None:
    rewards = [summary.best_reward for summary in summaries if summary.best_reward is not None]
    return max(rewards) if rewards else None


def _best_from_history(best_history: list[float | None]) -> float | None:
    rewards = [reward for reward in best_history if reward is not None]
    return max(rewards) if rewards else None


def _no_improvement(best_history: list[float | None], patience: int) -> bool:
    rewards = [reward for reward in best_history if reward is not None]
    if patience <= 0 or len(rewards) <= patience:
        return False
    best_before_window = max(rewards[: -patience])
    return max(rewards[-patience:]) <= best_before_window


def _low_improvement(best_history: list[float | None], window: int, epsilon: float) -> bool:
    rewards = [reward for reward in best_history if reward is not None]
    if window <= 1 or len(rewards) < window:
        return False
    recent = rewards[-window:]
    return max(recent) - min(recent) < epsilon


def _representative_value(min_value: float, max_value: float, scale: str) -> float:
    if scale == "log":
        return math.sqrt(min_value * max_value)
    return (min_value + max_value) / 2
