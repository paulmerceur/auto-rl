from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from puffer_llm_sweeper.loop import StopRules, evaluate_stop_rules, run_loop


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
            config_path.write_text("env_name: target\n", encoding="utf-8")

            result = run_loop(
                config_path=config_path,
                logs_dir=logs_dir,
                summary_path=summary_path,
                decision_path=decision_path,
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
                skip_training=True,
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            decision = json.loads(decision_path.read_text(encoding="utf-8"))

        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(summary["num_runs"], 0)
        self.assertEqual(decision["action"], "continue")


if __name__ == "__main__":
    unittest.main()
