"""Minimal constrained experiment loop controller."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from puffer_llm_sweeper.decisions import LlmDecision, decision_to_json
from puffer_llm_sweeper.metrics import RunSummary, summarize_logs, write_summary
from puffer_llm_sweeper.openrouter import OpenRouterClient
from puffer_llm_sweeper.runner import run_sweep


MAX_TRIALS_PER_ITERATION = 10


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
    trials_per_iteration: int,
    run_dir: Path | None = None,
    live: bool = False,
    skip_training: bool = False,
) -> LoopResult:
    if run_dir is not None:
        logs_dir, summary_path, decision_path, work_config_path = loop_paths(run_dir)

    start_time = time.monotonic()
    trials = 0
    best_history: list[float | None] = []
    last_decision: LlmDecision | None = None
    current_config_path = initialize_work_config(config_path, work_config_path, run_dir=run_dir)

    for iteration in range(1, rules.max_iterations + 1):
        remaining_trials = rules.max_trials - trials
        batch_trials = _next_batch_trials(
            default_trials=trials_per_iteration,
            remaining_trials=remaining_trials,
            last_decision=last_decision,
        )
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
            run_sweep(current_config_path, max_runs=batch_trials)
            trials += batch_trials

        summaries = summarize_logs(logs_dir) if logs_dir.exists() else []
        write_summary(summaries, summary_path)
        best_history.append(_best_reward(summaries))

        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        budget = {
            "max_total_trials": rules.max_trials,
            "trials_used": trials,
            "trials_remaining": max(rules.max_trials - trials, 0),
            "max_trials_per_iteration": MAX_TRIALS_PER_ITERATION,
            "default_trials_per_iteration": trials_per_iteration,
        }
        last_decision = OpenRouterClient().propose_decision(
            summary_payload,
            dry_run=not live,
            budget=budget,
        )
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


def loop_paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        run_dir / "logs",
        run_dir / "summary.json",
        run_dir / "decision.json",
        run_dir / "loop_config.yaml",
    )


def initialize_work_config(
    source_path: Path,
    work_config_path: Path,
    run_dir: Path | None = None,
) -> Path:
    if source_path == work_config_path and run_dir is None:
        return source_path
    work_config_path.parent.mkdir(parents=True, exist_ok=True)

    if run_dir is None:
        work_config_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        return work_config_path

    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Loop config must be a mapping: {source_path}")
    _apply_run_dir(raw, run_dir)
    work_config_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
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
    sweep = puffer.setdefault("sweep", {})
    if not isinstance(sweep, dict):
        raise ValueError("Loop config key `puffer.sweep` must be a mapping when provided.")

    for name, range_config in decision.search_space_update.items():
        _set_nested_sweep_range(sweep, name, range_config.model_dump(mode="json"))

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


def _next_batch_trials(
    default_trials: int,
    remaining_trials: int,
    last_decision: LlmDecision | None,
) -> int:
    if remaining_trials <= 0:
        return 0
    requested = last_decision.suggested_trials if last_decision else default_trials
    requested = requested or default_trials
    return max(1, min(requested, MAX_TRIALS_PER_ITERATION, remaining_trials))


def _set_nested_sweep_range(sweep: dict, dotted_name: str, value: dict) -> None:
    parts = dotted_name.split(".")
    if len(parts) != 2:
        raise ValueError(f"Expected sweep update path like section.name, got: {dotted_name}")
    section_name, param_name = parts
    section = sweep.setdefault(section_name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Loop config key `puffer.sweep.{section_name}` must be a mapping.")
    section[param_name] = value


def _apply_run_dir(raw: dict, run_dir: Path) -> None:
    raw["output_dir"] = str(run_dir)
    puffer = raw.setdefault("puffer", {})
    if not isinstance(puffer, dict):
        raise ValueError("Loop config key `puffer` must be a mapping when provided.")

    puffer["log_dir"] = str(run_dir / "logs")
    puffer["checkpoint_dir"] = str(run_dir / "checkpoints")

    train = puffer.setdefault("train", {})
    if not isinstance(train, dict):
        raise ValueError("Loop config key `puffer.train` must be a mapping when provided.")
    train["data_dir"] = str(run_dir / "pufferlib")
