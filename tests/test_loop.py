from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from puffer_llm_sweeper.decisions import parse_decision_json
from puffer_llm_sweeper.loop import (
    MAX_TRIALS_PER_ITERATION,
    StopRules,
    _next_batch_trials,
    apply_decision_to_config,
    evaluate_stop_rules,
    run_loop,
)


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

    def test_apply_decision_to_work_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "base.yaml"
            output_path = root / "next.yaml"
            source_path.write_text("env_name: target\npuffer:\n  sweep:\n    metric: score\n")
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


if __name__ == "__main__":
    unittest.main()
