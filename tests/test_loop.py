from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from puffer_llm_sweeper.decisions import parse_decision_json
from puffer_llm_sweeper.loop import (
    MAX_TRIALS_PER_ITERATION,
    StopRules,
    _next_batch_trials,
    apply_decision_to_config,
    evaluate_stop_rules,
    extract_current_search_space,
    initialize_work_config,
    loop_paths,
    run_loop,
)
from puffer_llm_sweeper.openrouter import InvalidDecisionResponse


class LoopTests(unittest.TestCase):
    def test_target_reward_stop(self) -> None:
        reason = evaluate_stop_rules(
            summaries=[],
            best_history=[0.5, 1.0],
            iteration=1,
            trials=1,
            elapsed_minutes=0.0,
            rules=StopRules(
                max_iterations=5,
                max_trials=5,
                max_minutes=10,
                target_reward=1.0,
                no_improvement_iterations=3,
                improvement_window=3,
                improvement_epsilon=0.0,
                max_failures=2,
            ),
        )

        self.assertEqual(reason, "target_reward")

    def test_skip_training_loop_writes_summary_and_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "base.yaml"
            logs_dir = root / "logs"
            summary_path = root / "summary.json"
            decision_path = root / "decision.json"
            work_config_path = root / "loop_config.yaml"
            config_path.write_text("env_name: target\n", encoding="utf-8")

            result = run_loop(
                config_path=config_path,
                logs_dir=logs_dir,
                summary_path=summary_path,
                decision_path=decision_path,
                work_config_path=work_config_path,
                rules=StopRules(
                    max_iterations=1,
                    max_trials=1,
                    max_minutes=10,
                    target_reward=None,
                    no_improvement_iterations=3,
                    improvement_window=3,
                    improvement_epsilon=0.0,
                    max_failures=2,
                ),
                trials_per_iteration=3,
                skip_training=True,
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            work_config_exists = work_config_path.exists()

        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(summary["num_runs"], 0)
        self.assertEqual(decision["action"], "continue")
        self.assertTrue(work_config_exists)

    def test_run_dir_isolates_loop_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "loop-test"
            config_path = root / "base.yaml"
            config_path.write_text("env_name: target\n", encoding="utf-8")

            result = run_loop(
                config_path=config_path,
                logs_dir=root / "ignored-logs",
                summary_path=root / "ignored-summary.json",
                decision_path=root / "ignored-decision.json",
                work_config_path=root / "ignored-config.yaml",
                run_dir=run_dir,
                rules=StopRules(
                    max_iterations=1,
                    max_trials=1,
                    max_minutes=10,
                    target_reward=None,
                    no_improvement_iterations=3,
                    improvement_window=3,
                    improvement_epsilon=0.0,
                    max_failures=2,
                ),
                trials_per_iteration=1,
                skip_training=True,
            )
            logs_dir, summary_path, decision_path, work_config_path = loop_paths(run_dir)

            summary_exists = summary_path.exists()
            decision_exists = decision_path.exists()
            work_config = yaml.safe_load(work_config_path.read_text(encoding="utf-8"))

        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertTrue(summary_exists)
        self.assertTrue(decision_exists)
        self.assertEqual(str(logs_dir), str(run_dir / "logs"))
        self.assertEqual(work_config["output_dir"], str(run_dir))
        self.assertEqual(work_config["puffer"]["log_dir"], str(run_dir / "logs"))
        self.assertEqual(work_config["puffer"]["checkpoint_dir"], str(run_dir / "checkpoints"))
        self.assertEqual(work_config["puffer"]["train"]["data_dir"], str(run_dir / "pufferlib"))

    def test_initialize_work_config_preserves_existing_puffer_settings_with_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "base.yaml"
            output_path = root / "work.yaml"
            run_dir = root / "loop-run"
            source_path.write_text(
                "env_name: target\n"
                "puffer:\n"
                "  slowly: true\n"
                "  train:\n"
                "    total_timesteps: 1024\n",
                encoding="utf-8",
            )

            initialize_work_config(source_path, output_path, run_dir=run_dir)
            updated = yaml.safe_load(output_path.read_text(encoding="utf-8"))

        self.assertTrue(updated["puffer"]["slowly"])
        self.assertEqual(updated["puffer"]["train"]["total_timesteps"], 1024)
        self.assertEqual(updated["puffer"]["train"]["data_dir"], str(run_dir / "pufferlib"))

    def test_apply_decision_to_work_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "base.yaml"
            output_path = root / "next.yaml"
            source_path.write_text(
                "env_name: target\n"
                "puffer:\n"
                "  sweep:\n"
                "    metric: score\n"
                "    train:\n"
                "      learning_rate:\n"
                "        distribution: log_normal\n"
                "        min: 0.00001\n"
                "        max: 0.1\n"
                "        scale: 0.5\n",
                encoding="utf-8",
            )
            decision = parse_decision_json(
                {
                    "action": "narrow_search",
                    "reason": "Use lower LR.",
                    "search_space_update": {
                        "train.learning_rate": {
                            "distribution": "log_normal",
                            "min": 0.0001,
                            "max": 0.001,
                            "scale": 0.5,
                        }
                    },
                    "notes": [],
                }
            )

            apply_decision_to_config(source_path, decision, output_path)
            updated = yaml.safe_load(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            updated["puffer"]["sweep"]["train"]["learning_rate"],
            {"distribution": "log_normal", "min": 0.0001, "max": 0.001, "scale": 0.5},
        )
        self.assertEqual(updated["puffer"]["sweep"]["metric"], "score")

    def test_rejects_invalid_narrowing_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "base.yaml"
            output_path = root / "next.yaml"
            source_path.write_text(
                "env_name: target\n"
                "puffer:\n"
                "  sweep:\n"
                "    train:\n"
                "      learning_rate:\n"
                "        distribution: log_normal\n"
                "        min: 0.0001\n"
                "        max: 0.001\n"
                "        scale: 0.5\n",
                encoding="utf-8",
            )
            decision = parse_decision_json(
                {
                    "action": "narrow_search",
                    "reason": "This expands instead.",
                    "search_space_update": {
                        "train.learning_rate": {
                            "distribution": "log_normal",
                            "min": 0.00001,
                            "max": 0.01,
                            "scale": 0.5,
                        }
                    },
                    "notes": [],
                }
            )

            with self.assertRaises(ValueError):
                apply_decision_to_config(source_path, decision, output_path)

    def test_rejects_inactive_sweep_key_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "base.yaml"
            output_path = root / "next.yaml"
            source_path.write_text(
                "env_name: target\n"
                "puffer:\n"
                "  sweep:\n"
                "    sweep_only: learning_rate\n"
                "    train:\n"
                "      clip_coef:\n"
                "        distribution: uniform\n"
                "        min: 0.05\n"
                "        max: 0.5\n"
                "        scale: auto\n",
                encoding="utf-8",
            )
            decision = parse_decision_json(
                {
                    "action": "narrow_search",
                    "reason": "This key is configured but inactive.",
                    "search_space_update": {
                        "train.clip_coef": {
                            "distribution": "uniform",
                            "min": 0.1,
                            "max": 0.2,
                            "scale": "auto",
                        }
                    },
                    "notes": [],
                }
            )

            with self.assertRaises(ValueError):
                apply_decision_to_config(source_path, decision, output_path)

    def test_extract_current_search_space_respects_sweep_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "base.yaml"
            config_path.write_text(
                "env_name: target\n"
                "puffer:\n"
                "  sweep:\n"
                "    sweep_only: learning_rate\n"
                "    train:\n"
                "      learning_rate:\n"
                "        distribution: log_normal\n"
                "        min: 0.00001\n"
                "        max: 0.1\n"
                "        scale: 0.5\n"
                "      clip_coef:\n"
                "        distribution: uniform\n"
                "        min: 0.05\n"
                "        max: 0.5\n"
                "        scale: auto\n",
                encoding="utf-8",
            )

            current = extract_current_search_space(config_path)

        self.assertIn("train.learning_rate", current)
        self.assertNotIn("train.clip_coef", current)

    def test_next_batch_trials_clamps_llm_suggestion(self) -> None:
        decision = parse_decision_json(
            {
                "action": "continue",
                "reason": "Use the max batch.",
                "suggested_trials": MAX_TRIALS_PER_ITERATION,
                "search_space_update": {},
                "notes": [],
            }
        )

        self.assertEqual(
            _next_batch_trials(default_trials=3, remaining_trials=4, last_decision=decision),
            4,
        )

    def test_loop_counts_completed_trials_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "loop"
            config_path = root / "base.yaml"
            config_path.write_text("env_name: target\n", encoding="utf-8")

            def fake_run_sweep(_config_path: Path, max_runs: int | None = None, **_kwargs) -> int:
                logs_dir = run_dir / "logs" / "target"
                logs_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(max_runs or 0):
                    run_id = len(list(logs_dir.glob("*.json"))) + 1
                    (logs_dir / f"run-{run_id}.json").write_text(
                        json.dumps({"metrics": {"env/score": [float(run_id + idx)]}}),
                        encoding="utf-8",
                    )
                return 0

            with patch("puffer_llm_sweeper.loop.run_sweep", side_effect=fake_run_sweep):
                result = run_loop(
                    config_path=config_path,
                    logs_dir=root / "ignored-logs",
                    summary_path=root / "ignored-summary.json",
                    decision_path=root / "ignored-decision.json",
                    work_config_path=root / "ignored-config.yaml",
                    run_dir=run_dir,
                    rules=StopRules(
                        max_iterations=1,
                        max_trials=5,
                        max_minutes=10,
                        target_reward=None,
                        no_improvement_iterations=3,
                        improvement_window=3,
                        improvement_epsilon=0.0,
                        max_failures=2,
                    ),
                    trials_per_iteration=2,
                )

            journal = [
                json.loads(line)
                for line in result.journal_path.read_text(encoding="utf-8").splitlines()
            ]
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            plot_exists = result.plot_path.exists()
            iteration_summary = json.loads(
                (run_dir / "iterations" / "iteration-001.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.trials, 2)
        self.assertEqual(journal[0]["requested_trials"], 2)
        self.assertEqual(journal[0]["completed_trials"], 2)
        self.assertEqual(report["result"]["completed_trials"], 2)
        self.assertFalse(report["llm"]["changed_trial_count"])
        self.assertEqual(iteration_summary["num_runs"], 2)
        self.assertTrue(plot_exists)

    def test_report_detects_llm_trial_count_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "loop"
            config_path = root / "base.yaml"
            config_path.write_text("env_name: target\n", encoding="utf-8")

            def fake_run_sweep(_config_path: Path, max_runs: int | None = None, **_kwargs) -> int:
                logs_dir = run_dir / "logs" / "target"
                logs_dir.mkdir(parents=True, exist_ok=True)
                for _ in range(max_runs or 0):
                    run_id = len(list(logs_dir.glob("*.json"))) + 1
                    (logs_dir / f"run-{run_id}.json").write_text(
                        json.dumps({"metrics": {"env/score": [float(run_id)]}}),
                        encoding="utf-8",
                    )
                return 0

            with patch("puffer_llm_sweeper.loop.run_sweep", side_effect=fake_run_sweep):
                result = run_loop(
                    config_path=config_path,
                    logs_dir=root / "ignored-logs",
                    summary_path=root / "ignored-summary.json",
                    decision_path=root / "ignored-decision.json",
                    work_config_path=root / "ignored-config.yaml",
                    run_dir=run_dir,
                    rules=StopRules(
                        max_iterations=2,
                        max_trials=5,
                        max_minutes=10,
                        target_reward=None,
                        no_improvement_iterations=3,
                        improvement_window=3,
                        improvement_epsilon=0.0,
                        max_failures=2,
                    ),
                    trials_per_iteration=2,
                )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["llm"]["changed_trial_count"])
        self.assertNotIn("llm_made_no_changes", report["problem_flags"])

    def test_loop_records_invalid_live_decision_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "loop"
            config_path = root / "base.yaml"
            config_path.write_text(
                "env_name: target\n"
                "puffer:\n"
                "  sweep:\n"
                "    sweep_only: learning_rate\n"
                "    train:\n"
                "      learning_rate:\n"
                "        distribution: log_normal\n"
                "        min: 0.0001\n"
                "        max: 0.001\n"
                "        scale: 0.5\n",
                encoding="utf-8",
            )

            def fake_run_sweep(_config_path: Path, max_runs: int | None = None, **_kwargs) -> int:
                logs_dir = run_dir / "logs" / "target"
                logs_dir.mkdir(parents=True, exist_ok=True)
                (logs_dir / "run-1.json").write_text(
                    json.dumps({"metrics": {"env/score": [1.0]}}),
                    encoding="utf-8",
                )
                return 0

            class FakeClient:
                def propose_decision(self, *_args, **_kwargs):
                    return parse_decision_json(
                        {
                            "action": "narrow_search",
                            "reason": "This expands while claiming to narrow.",
                            "suggested_trials": 4,
                            "search_space_update": {
                                "train.learning_rate": {
                                    "distribution": "log_normal",
                                    "min": 0.00001,
                                    "max": 0.01,
                                    "scale": 0.5,
                                }
                            },
                            "notes": [],
                        }
                    )

            with (
                patch("puffer_llm_sweeper.loop.run_sweep", side_effect=fake_run_sweep),
                patch("puffer_llm_sweeper.loop.OpenRouterClient", return_value=FakeClient()),
            ):
                result = run_loop(
                    config_path=config_path,
                    logs_dir=root / "ignored-logs",
                    summary_path=root / "ignored-summary.json",
                    decision_path=root / "ignored-decision.json",
                    work_config_path=root / "ignored-config.yaml",
                    run_dir=run_dir,
                    rules=StopRules(
                        max_iterations=1,
                        max_trials=3,
                        max_minutes=10,
                        target_reward=None,
                        no_improvement_iterations=3,
                        improvement_window=3,
                        improvement_epsilon=0.0,
                        max_failures=2,
                    ),
                    trials_per_iteration=1,
                    live=True,
                )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
            rejected = json.loads(
                (run_dir / "rejected-decisions" / "iteration-001.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(decision["action"], "continue")
        self.assertIn("invalid_llm_decisions", report["problem_flags"])
        self.assertIn("narrow_search cannot expand", report["iterations"][0]["decision_error"])
        self.assertEqual(rejected["decision"]["action"], "narrow_search")

    def test_loop_stops_before_llm_when_trial_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "loop"
            config_path = root / "base.yaml"
            config_path.write_text("env_name: target\n", encoding="utf-8")

            def fake_run_sweep(_config_path: Path, max_runs: int | None = None, **_kwargs) -> int:
                logs_dir = run_dir / "logs" / "target"
                logs_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(max_runs or 0):
                    (logs_dir / f"run-{idx}.json").write_text(
                        json.dumps({"metrics": {"env/score": [float(idx)]}}),
                        encoding="utf-8",
                    )
                return 0

            with (
                patch("puffer_llm_sweeper.loop.run_sweep", side_effect=fake_run_sweep),
                patch("puffer_llm_sweeper.loop.OpenRouterClient") as client_cls,
            ):
                result = run_loop(
                    config_path=config_path,
                    logs_dir=root / "ignored-logs",
                    summary_path=root / "ignored-summary.json",
                    decision_path=root / "ignored-decision.json",
                    work_config_path=root / "ignored-config.yaml",
                    run_dir=run_dir,
                    rules=StopRules(
                        max_iterations=3,
                        max_trials=3,
                        max_minutes=10,
                        target_reward=None,
                        no_improvement_iterations=3,
                        improvement_window=3,
                        improvement_epsilon=0.0,
                        max_failures=2,
                    ),
                    trials_per_iteration=3,
                    live=True,
                )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.stop_reason, "max_trials")
        self.assertEqual(report["llm"]["num_decisions"], 0)
        client_cls.assert_not_called()

    def test_loop_records_malformed_live_decision_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "loop"
            config_path = root / "base.yaml"
            config_path.write_text("env_name: target\n", encoding="utf-8")

            def fake_run_sweep(_config_path: Path, max_runs: int | None = None, **_kwargs) -> int:
                logs_dir = run_dir / "logs" / "target"
                logs_dir.mkdir(parents=True, exist_ok=True)
                (logs_dir / "run-1.json").write_text(
                    json.dumps({"metrics": {"env/score": [1.0]}}),
                    encoding="utf-8",
                )
                return 0

            class FakeClient:
                def propose_decision(self, *_args, **_kwargs):
                    raise InvalidDecisionResponse(
                        '{"action":"continue","suggested_trials":0}',
                        "suggested_trials must be >= 1",
                    )

            with (
                patch("puffer_llm_sweeper.loop.run_sweep", side_effect=fake_run_sweep),
                patch("puffer_llm_sweeper.loop.OpenRouterClient", return_value=FakeClient()),
            ):
                result = run_loop(
                    config_path=config_path,
                    logs_dir=root / "ignored-logs",
                    summary_path=root / "ignored-summary.json",
                    decision_path=root / "ignored-decision.json",
                    work_config_path=root / "ignored-config.yaml",
                    run_dir=run_dir,
                    rules=StopRules(
                        max_iterations=1,
                        max_trials=3,
                        max_minutes=10,
                        target_reward=None,
                        no_improvement_iterations=3,
                        improvement_window=3,
                        improvement_epsilon=0.0,
                        max_failures=2,
                    ),
                    trials_per_iteration=1,
                    live=True,
                )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            rejected = json.loads(
                (run_dir / "rejected-decisions" / "iteration-001.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertIn("invalid_llm_decisions", report["problem_flags"])
        self.assertIn("suggested_trials must be >= 1", rejected["error"])


if __name__ == "__main__":
    unittest.main()
