"""Minimal constrained experiment loop controller."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from puffer_llm_sweeper.decisions import (
    DecisionAction,
    LlmDecision,
    decision_to_json,
    parse_decision_json,
)
from puffer_llm_sweeper.metrics import (
    RunSummary,
    log_paths,
    summarize_log_paths,
    summarize_logs,
    write_summary,
)
from puffer_llm_sweeper.openrouter import InvalidDecisionResponse, OpenRouterClient


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
    journal_path: Path | None = None
    report_path: Path | None = None
    plot_path: Path | None = None


@dataclass(frozen=True)
class LoopArtifacts:
    logs_dir: Path
    summary_path: Path
    decision_path: Path
    work_config_path: Path
    journal_path: Path
    report_path: Path
    plot_path: Path
    iteration_summaries_dir: Path
    rejected_decisions_dir: Path


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    requested_trials: int
    completed_trials: int
    cumulative_trials: int
    iteration_best_reward: float | None
    best_reward_so_far: float | None
    num_failures: int
    run_ids: list[str]
    decision: dict[str, Any] | None
    decision_error: str | None
    training_error: str | None
    elapsed_minutes: float
    stop_reason: str | None
    iteration_summary_path: str


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
    artifacts = loop_artifacts(
        logs_dir=logs_dir,
        summary_path=summary_path,
        decision_path=decision_path,
        work_config_path=work_config_path,
    )
    _reset_loop_artifacts(artifacts)

    start_time = time.monotonic()
    trials = 0
    best_history: list[float | None] = []
    records: list[IterationRecord] = []
    last_decision: LlmDecision | None = None
    current_config_path = initialize_work_config(config_path, artifacts.work_config_path, run_dir=run_dir)

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
            return _finish_loop(
                iterations=iteration - 1,
                trials=trials,
                stop_reason=pre_stop,
                best_history=best_history,
                last_decision=last_decision,
                records=records,
                artifacts=artifacts,
                rules=rules,
                live=live,
                skip_training=skip_training,
            )

        before_logs = set(log_paths(artifacts.logs_dir))
        training_error = None
        if not skip_training:
            try:
                run_sweep_with_progress(
                    current_config_path,
                    max_runs=batch_trials,
                    logs_dir=artifacts.logs_dir,
                    before_logs=before_logs,
                    phase=iteration,
                    max_phases=rules.max_iterations,
                )
            except Exception as exc:  # Keep a report even when training fails.
                training_error = f"{type(exc).__name__}: {exc}"
        else:
            progress = ProgressPrinter()
            progress.write_line(f"Phase {iteration}/{rules.max_iterations}: skipped training")
            progress.close()

        after_logs = set(log_paths(artifacts.logs_dir))
        iteration_paths = sorted(after_logs - before_logs)
        iteration_summaries = summarize_log_paths(iteration_paths)
        completed_trials = len(iteration_summaries)
        trials += completed_trials

        summaries = summarize_logs(artifacts.logs_dir) if artifacts.logs_dir.exists() else []
        write_summary(summaries, artifacts.summary_path)
        iteration_summary_path = _iteration_summary_path(artifacts, iteration)
        write_summary(iteration_summaries, iteration_summary_path)
        best_history.append(_best_reward(iteration_summaries))

        post_training_stop = evaluate_stop_rules(
            summaries=summaries,
            best_history=best_history,
            iteration=iteration,
            trials=trials,
            elapsed_minutes=(time.monotonic() - start_time) / 60,
            rules=rules,
        )

        if training_error:
            record = _iteration_record(
                iteration=iteration,
                requested_trials=batch_trials,
                completed_trials=completed_trials,
                cumulative_trials=trials,
                iteration_summaries=iteration_summaries,
                best_history=best_history,
                decision=None,
                decision_error=None,
                training_error=training_error,
                elapsed_minutes=(time.monotonic() - start_time) / 60,
                stop_reason="training_error",
                iteration_summary_path=iteration_summary_path,
            )
            records.append(record)
            append_journal_record(artifacts.journal_path, record)
            return _finish_loop(
                iterations=iteration,
                trials=trials,
                stop_reason="training_error",
                best_history=best_history,
                last_decision=last_decision,
                records=records,
                artifacts=artifacts,
                rules=rules,
                live=live,
                skip_training=skip_training,
            )

        if post_training_stop and post_training_stop != "max_iterations":
            record = _iteration_record(
                iteration=iteration,
                requested_trials=batch_trials,
                completed_trials=completed_trials,
                cumulative_trials=trials,
                iteration_summaries=iteration_summaries,
                best_history=best_history,
                decision=None,
                decision_error=None,
                training_error=None,
                elapsed_minutes=(time.monotonic() - start_time) / 60,
                stop_reason=post_training_stop,
                iteration_summary_path=iteration_summary_path,
            )
            records.append(record)
            append_journal_record(artifacts.journal_path, record)
            return _finish_loop(
                iterations=iteration,
                trials=trials,
                stop_reason=post_training_stop,
                best_history=best_history,
                last_decision=last_decision,
                records=records,
                artifacts=artifacts,
                rules=rules,
                live=live,
                skip_training=skip_training,
            )

        summary_payload = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
        budget = {
            "max_total_trials": rules.max_trials,
            "trials_used": trials,
            "trials_remaining": max(rules.max_trials - trials, 0),
            "max_trials_per_iteration": MAX_TRIALS_PER_ITERATION,
            "default_trials_per_iteration": trials_per_iteration,
        }
        decision_error = None
        try:
            progress = ProgressPrinter()
            progress.write_line(f"Phase {iteration}/{rules.max_iterations}: asking LLM")
            try:
                last_decision = OpenRouterClient().propose_decision(
                    summary_payload,
                    dry_run=not live,
                    budget=budget,
                    current_search_space=extract_current_search_space(current_config_path),
                )
            except Exception:
                progress.close()
                raise
            else:
                progress.write_line(
                    format_decision_status(
                        phase=iteration,
                        max_phases=rules.max_iterations,
                        decision=last_decision,
                    )
                )
                progress.close()
        except InvalidDecisionResponse as exc:
            decision_error = str(exc)
            write_rejected_raw_decision(artifacts, iteration, exc.raw_content, decision_error)
            last_decision = fallback_decision(None, decision_error)
            progress = ProgressPrinter()
            progress.write_line(
                f"Phase {iteration}/{rules.max_iterations}: rejected malformed LLM response"
            )
            progress.close()
        try:
            apply_decision_to_config(current_config_path, last_decision, artifacts.work_config_path)
        except ValueError as exc:
            decision_error = str(exc)
            write_rejected_decision(artifacts, iteration, last_decision, decision_error)
            last_decision = fallback_decision(last_decision, decision_error)
            progress = ProgressPrinter()
            progress.write_line(
                f"Phase {iteration}/{rules.max_iterations}: rejected invalid LLM update"
            )
            progress.close()
        artifacts.decision_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts.decision_path.write_text(decision_to_json(last_decision) + "\n", encoding="utf-8")
        current_config_path = artifacts.work_config_path

        post_decision_stop = evaluate_stop_rules(
            summaries=summaries,
            best_history=best_history,
            iteration=iteration,
            trials=trials,
            elapsed_minutes=(time.monotonic() - start_time) / 60,
            rules=rules,
        )
        if last_decision.action == "stop":
            post_decision_stop = post_decision_stop or "llm_stop"

        record = _iteration_record(
            iteration=iteration,
            requested_trials=batch_trials,
            completed_trials=completed_trials,
            cumulative_trials=trials,
            iteration_summaries=iteration_summaries,
            best_history=best_history,
            decision=last_decision,
            decision_error=decision_error,
            training_error=None,
            elapsed_minutes=(time.monotonic() - start_time) / 60,
            stop_reason=post_decision_stop,
            iteration_summary_path=iteration_summary_path,
        )
        records.append(record)
        append_journal_record(artifacts.journal_path, record)

        if post_decision_stop:
            return _finish_loop(
                iterations=iteration,
                trials=trials,
                stop_reason=post_decision_stop,
                best_history=best_history,
                last_decision=last_decision,
                records=records,
                artifacts=artifacts,
                rules=rules,
                live=live,
                skip_training=skip_training,
            )

    return _finish_loop(
        iterations=rules.max_iterations,
        trials=trials,
        stop_reason="max_iterations",
        best_history=best_history,
        last_decision=last_decision,
        records=records,
        artifacts=artifacts,
        rules=rules,
        live=live,
        skip_training=skip_training,
    )


def loop_paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        run_dir / "logs",
        run_dir / "summary.json",
        run_dir / "decision.json",
        run_dir / "loop_config.yaml",
    )


def loop_artifacts(
    logs_dir: Path,
    summary_path: Path,
    decision_path: Path,
    work_config_path: Path,
) -> LoopArtifacts:
    base_dir = work_config_path.parent
    return LoopArtifacts(
        logs_dir=logs_dir,
        summary_path=summary_path,
        decision_path=decision_path,
        work_config_path=work_config_path,
        journal_path=base_dir / "journal.jsonl",
        report_path=base_dir / "loop_report.json",
        plot_path=base_dir / "performance.svg",
        iteration_summaries_dir=base_dir / "iterations",
        rejected_decisions_dir=base_dir / "rejected-decisions",
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

    _validate_decision_transition(sweep, decision)
    for name, range_config in decision.search_space_update.items():
        _set_nested_sweep_range(sweep, name, range_config.model_dump(mode="json"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")


def extract_current_search_space(config_path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    puffer = raw.get("puffer")
    if not isinstance(puffer, dict):
        return {}
    sweep = puffer.get("sweep")
    if not isinstance(sweep, dict):
        return {}

    active_names = _active_sweep_names(sweep)
    current: dict[str, dict[str, Any]] = {}
    for section_name in ("train", "vec", "policy"):
        section = sweep.get(section_name)
        if not isinstance(section, dict):
            continue
        for param_name, value in section.items():
            if active_names is not None and param_name not in active_names:
                continue
            if isinstance(value, dict) and {"distribution", "min", "max"}.issubset(value):
                current[f"{section_name}.{param_name}"] = {
                    "distribution": value.get("distribution"),
                    "min": value.get("min"),
                    "max": value.get("max"),
                    "scale": value.get("scale"),
                }
    return current


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


def append_journal_record(journal_path: Path, record: IterationRecord) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as journal_file:
        journal_file.write(json.dumps(asdict(record), sort_keys=True) + "\n")


class ProgressPrinter:
    def __init__(self) -> None:
        self._fd = os.dup(1)
        self._last_was_inline = False

    def update(self, line: str) -> None:
        self._write("\r" + line)
        self._last_was_inline = True

    def write_line(self, line: str) -> None:
        prefix = "\n" if self._last_was_inline else ""
        self._write(prefix + line + "\n")
        self._last_was_inline = False

    def close(self) -> None:
        if self._last_was_inline:
            self._write("\n")
            self._last_was_inline = False
        os.close(self._fd)

    def _write(self, text: str) -> None:
        os.write(self._fd, text.encode("utf-8", errors="replace"))


def run_sweep_with_progress(
    config_path: Path,
    max_runs: int,
    logs_dir: Path,
    before_logs: set[Path],
    phase: int,
    max_phases: int,
) -> None:
    progress = ProgressPrinter()
    command = [
        sys.executable,
        "-m",
        "puffer_llm_sweeper",
        "sweep",
        "--config",
        str(config_path),
        "--max-runs",
        str(max_runs),
        "--quiet",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        while process.poll() is None:
            completed = len(set(log_paths(logs_dir)) - before_logs)
            progress.update(format_sweep_progress(phase, max_phases, completed, max_runs, "running sweep"))
            time.sleep(0.5)
        stdout, stderr = process.communicate()
        completed = len(set(log_paths(logs_dir)) - before_logs)
        status = "sweep failed" if process.returncode else "sweep complete"
        progress.update(format_sweep_progress(phase, max_phases, completed, max_runs, status))
    finally:
        progress.close()

    if process.returncode:
        detail = (stderr or stdout or "").strip()
        if detail:
            detail = detail[-2000:]
            raise RuntimeError(f"sweep subprocess failed with exit code {process.returncode}: {detail}")
        raise RuntimeError(f"sweep subprocess failed with exit code {process.returncode}")


def format_sweep_progress(
    phase: int,
    max_phases: int,
    completed: int,
    total: int,
    status: str,
) -> str:
    total = max(total, 1)
    completed = min(max(completed, 0), total)
    width = 20
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"Phase {phase}/{max_phases}: runs {completed}/{total} [{bar}] {status}"


def format_decision_status(phase: int, max_phases: int, decision: LlmDecision) -> str:
    updates = ", ".join(sorted(decision.search_space_update)) or "no search-space updates"
    suggested = decision.suggested_trials if decision.suggested_trials is not None else "none"
    return (
        f"Phase {phase}/{max_phases}: LLM {decision.action} "
        f"(suggested_trials={suggested}; {updates})"
    )


def write_rejected_decision(
    artifacts: LoopArtifacts,
    iteration: int,
    decision: LlmDecision,
    error: str,
) -> Path:
    path = artifacts.rejected_decisions_dir / f"iteration-{iteration:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "decision": decision.model_dump(mode="json"),
        "error": error,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_rejected_raw_decision(
    artifacts: LoopArtifacts,
    iteration: int,
    raw_content: str,
    error: str,
) -> Path:
    path = artifacts.rejected_decisions_dir / f"iteration-{iteration:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_content": raw_content,
        "error": error,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def fallback_decision(rejected_decision: LlmDecision | None, error: str) -> LlmDecision:
    reason = f"Rejected invalid LLM search-space update: {_short_error(error)}"
    return parse_decision_json(
        {
            "action": "continue",
            "reason": reason,
            "suggested_trials": rejected_decision.suggested_trials if rejected_decision else 1,
            "search_space_update": {},
            "notes": ["rejected-live-decision"],
        }
    )


def _short_error(error: str, max_length: int = 440) -> str:
    normalized = " ".join(error.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."


def _finish_loop(
    iterations: int,
    trials: int,
    stop_reason: str,
    best_history: list[float | None],
    last_decision: LlmDecision | None,
    records: list[IterationRecord],
    artifacts: LoopArtifacts,
    rules: StopRules,
    live: bool,
    skip_training: bool,
) -> LoopResult:
    result = LoopResult(
        iterations=iterations,
        trials=trials,
        stop_reason=stop_reason,
        best_reward=_best_from_history(best_history),
        last_decision=last_decision,
        journal_path=artifacts.journal_path,
        report_path=artifacts.report_path,
        plot_path=artifacts.plot_path,
    )
    write_loop_report(result, records, artifacts.report_path, rules, live, skip_training)
    write_performance_svg(records, artifacts.plot_path)
    return result


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


def _get_nested_sweep_range(sweep: dict, dotted_name: str) -> dict[str, Any] | None:
    parts = dotted_name.split(".")
    if len(parts) != 2:
        raise ValueError(f"Expected sweep update path like section.name, got: {dotted_name}")
    active_names = _active_sweep_names(sweep)
    if active_names is not None and parts[1] not in active_names:
        return None
    section = sweep.get(parts[0])
    if not isinstance(section, dict):
        return None
    value = section.get(parts[1])
    return value if isinstance(value, dict) else None


def _validate_decision_transition(sweep: dict, decision: LlmDecision) -> None:
    if not decision.search_space_update:
        if decision.action in {DecisionAction.NARROW_SEARCH, DecisionAction.EXPAND_SEARCH}:
            raise ValueError(f"{decision.action} requires at least one search-space update.")
        return

    narrowed = False
    expanded = False
    for name, range_config in decision.search_space_update.items():
        current = _get_nested_sweep_range(sweep, name)
        if current is None:
            raise ValueError(f"Cannot update missing or inactive sweep key: {name}")
        current_min = current.get("min")
        current_max = current.get("max")
        if not isinstance(current_min, int | float) or not isinstance(current_max, int | float):
            continue

        if range_config.min >= current_min and range_config.max <= current_max:
            narrowed = narrowed or (range_config.min > current_min or range_config.max < current_max)
        elif decision.action == DecisionAction.NARROW_SEARCH:
            raise ValueError(f"narrow_search cannot expand {name}.")

        if range_config.min <= current_min and range_config.max >= current_max:
            expanded = expanded or (range_config.min < current_min or range_config.max > current_max)
        elif decision.action == DecisionAction.EXPAND_SEARCH:
            raise ValueError(f"expand_search cannot narrow {name}.")

    if decision.action == DecisionAction.NARROW_SEARCH and not narrowed:
        raise ValueError("narrow_search must narrow at least one existing range.")
    if decision.action == DecisionAction.EXPAND_SEARCH and not expanded:
        raise ValueError("expand_search must expand at least one existing range.")


def _active_sweep_names(sweep: dict) -> set[str] | None:
    raw = sweep.get("sweep_only")
    if raw is None:
        return None
    if isinstance(raw, str):
        return {name.strip() for name in raw.split(",") if name.strip()}
    if isinstance(raw, list):
        return {str(name).strip() for name in raw if str(name).strip()}
    return None


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


def _reset_loop_artifacts(artifacts: LoopArtifacts) -> None:
    artifacts.journal_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.iteration_summaries_dir.mkdir(parents=True, exist_ok=True)
    for path in (artifacts.journal_path, artifacts.report_path, artifacts.plot_path):
        if path.exists():
            path.unlink()
    for path in artifacts.iteration_summaries_dir.glob("iteration-*.json"):
        path.unlink()
    if artifacts.rejected_decisions_dir.exists():
        for path in artifacts.rejected_decisions_dir.glob("iteration-*.json"):
            path.unlink()


def _iteration_summary_path(artifacts: LoopArtifacts, iteration: int) -> Path:
    return artifacts.iteration_summaries_dir / f"iteration-{iteration:03d}.json"


def _iteration_record(
    iteration: int,
    requested_trials: int,
    completed_trials: int,
    cumulative_trials: int,
    iteration_summaries: list[RunSummary],
    best_history: list[float | None],
    decision: LlmDecision | None,
    decision_error: str | None,
    training_error: str | None,
    elapsed_minutes: float,
    stop_reason: str | None,
    iteration_summary_path: Path,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        requested_trials=requested_trials,
        completed_trials=completed_trials,
        cumulative_trials=cumulative_trials,
        iteration_best_reward=_best_reward(iteration_summaries),
        best_reward_so_far=_best_from_history(best_history),
        num_failures=sum(1 for summary in iteration_summaries if summary.failure),
        run_ids=[summary.run_id for summary in iteration_summaries],
        decision=decision.model_dump(mode="json") if decision else None,
        decision_error=decision_error,
        training_error=training_error,
        elapsed_minutes=elapsed_minutes,
        stop_reason=stop_reason,
        iteration_summary_path=str(iteration_summary_path),
    )


def write_loop_report(
    result: LoopResult,
    records: list[IterationRecord],
    report_path: Path,
    rules: StopRules,
    live: bool,
    skip_training: bool,
) -> None:
    payload = {
        "result": {
            "iterations": result.iterations,
            "completed_trials": result.trials,
            "stop_reason": result.stop_reason,
            "best_reward": result.best_reward,
        },
        "mode": {"live": live, "skip_training": skip_training},
        "stop_rules": asdict(rules),
        "llm": _llm_activity(records),
        "reward_trend": _reward_trend(records),
        "problem_flags": _problem_flags(result, records),
        "iterations": [asdict(record) for record in records],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _llm_activity(records: list[IterationRecord]) -> dict[str, Any]:
    decisions = [record.decision for record in records if record.decision]
    suggested_trial_counts = [
        decision.get("suggested_trials") for decision in decisions if decision.get("suggested_trials")
    ]
    changed_trial_count = False
    for idx, record in enumerate(records[:-1]):
        if not record.decision:
            continue
        suggested_trials = record.decision.get("suggested_trials")
        next_requested = records[idx + 1].requested_trials
        if suggested_trials == next_requested and suggested_trials != record.requested_trials:
            changed_trial_count = True
            break
    return {
        "num_decisions": len(decisions),
        "actions": [decision["action"] for decision in decisions],
        "changed_search_space": any(decision.get("search_space_update") for decision in decisions),
        "changed_trial_count": changed_trial_count,
        "suggested_trial_counts": suggested_trial_counts,
    }


def _reward_trend(records: list[IterationRecord]) -> dict[str, Any]:
    rewards = [record.iteration_best_reward for record in records if record.iteration_best_reward is not None]
    best_so_far = [record.best_reward_so_far for record in records if record.best_reward_so_far is not None]
    return {
        "iteration_best_rewards": rewards,
        "best_reward_so_far": best_so_far,
        "improved": len(best_so_far) >= 2 and best_so_far[-1] > best_so_far[0],
        "first_best_reward": best_so_far[0] if best_so_far else None,
        "final_best_reward": best_so_far[-1] if best_so_far else None,
    }


def _problem_flags(result: LoopResult, records: list[IterationRecord]) -> list[str]:
    flags: list[str] = []
    if result.trials == 0:
        flags.append("no_completed_trials")
    if any(record.training_error for record in records):
        flags.append("training_errors")
    if any(record.decision_error for record in records):
        flags.append("invalid_llm_decisions")
    if any(record.num_failures for record in records):
        flags.append("failed_runs")
    if any(record.completed_trials > 0 and record.iteration_best_reward is None for record in records):
        flags.append("missing_reward_metrics")
    llm = _llm_activity(records)
    if llm["num_decisions"] and not llm["changed_search_space"] and not llm["changed_trial_count"]:
        flags.append("llm_made_no_changes")
    trend = _reward_trend(records)
    if len(trend["best_reward_so_far"]) >= 2 and not trend["improved"]:
        flags.append("no_reward_improvement")
    if result.stop_reason == "max_trials":
        flags.append("budget_exhausted")
    if result.stop_reason == "llm_stop":
        flags.append("llm_requested_stop")
    return flags


def write_performance_svg(records: list[IterationRecord], output_path: Path) -> None:
    width = 720
    height = 360
    padding = 48
    points = [
        (record.iteration, record.iteration_best_reward, record.best_reward_so_far)
        for record in records
        if record.iteration_best_reward is not None or record.best_reward_so_far is not None
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        output_path.write_text(_empty_svg(width, height), encoding="utf-8")
        return

    values = [value for _, *series in points for value in series if value is not None]
    min_y = min(values)
    max_y = max(values)
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    min_x = min(iteration for iteration, _, _ in points)
    max_x = max(iteration for iteration, _, _ in points)
    if min_x == max_x:
        min_x -= 1
        max_x += 1

    def project(iteration: int, reward: float | None) -> tuple[float, float] | None:
        if reward is None:
            return None
        x = padding + (iteration - min_x) / (max_x - min_x) * (width - 2 * padding)
        y = height - padding - (reward - min_y) / (max_y - min_y) * (height - 2 * padding)
        return x, y

    iteration_line = _polyline(
        [project(iteration, iteration_best) for iteration, iteration_best, _ in points],
        "#2563eb",
    )
    best_line = _polyline(
        [project(iteration, best_so_far) for iteration, _, best_so_far in points],
        "#16a34a",
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#111827" stroke-width="1"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#111827" stroke-width="1"/>
  <text x="{padding}" y="24" font-family="sans-serif" font-size="16" fill="#111827">Loop performance</text>
  <text x="{padding}" y="{height - 12}" font-family="sans-serif" font-size="12" fill="#374151">iteration {min_x:g} to {max_x:g}</text>
  <text x="{width - padding - 120}" y="24" font-family="sans-serif" font-size="12" fill="#2563eb">iteration best</text>
  <text x="{width - padding - 120}" y="42" font-family="sans-serif" font-size="12" fill="#16a34a">best so far</text>
  <text x="8" y="{padding}" font-family="sans-serif" font-size="12" fill="#374151">{max_y:.3g}</text>
  <text x="8" y="{height - padding}" font-family="sans-serif" font-size="12" fill="#374151">{min_y:.3g}</text>
  {iteration_line}
  {best_line}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def _polyline(points: list[tuple[float, float] | None], color: str) -> str:
    valid = [point for point in points if point is not None]
    if len(valid) == 1:
        x, y = valid[0]
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
    if len(valid) < 2:
        return ""
    points_attr = " ".join(f"{x:.2f},{y:.2f}" for x, y in valid)
    return f'<polyline points="{points_attr}" fill="none" stroke="{color}" stroke-width="2"/>'


def _empty_svg(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="32" y="48" font-family="sans-serif" font-size="16" fill="#111827">No reward data</text>
</svg>
"""
