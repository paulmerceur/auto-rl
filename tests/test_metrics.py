from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from puffer_llm_sweeper.metrics import parse_log, summarize_logs, write_summary


class MetricsTests(unittest.TestCase):
    def test_parse_puffer_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "target" / "run-1.json"
            log_path.parent.mkdir()
            log_path.write_text(
                json.dumps(
                    {
                        "base": {"env_name": "target"},
                        "sweep": {"metric": "score"},
                        "train": {"learning_rate": 0.001},
                        "metrics": {
                            "env/score": [1.0, 2.5, 2.0],
                            "agent_steps": [128, 256],
                            "uptime": [1.25],
                            "SPS": [2048],
                            "loss/policy": [0.4, 0.2],
                            "perf/inference": [0.01],
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = parse_log(log_path)

        self.assertEqual(summary.run_id, "run-1")
        self.assertEqual(summary.env_name, "target")
        self.assertEqual(summary.final_reward, 2.0)
        self.assertEqual(summary.best_reward, 2.5)
        self.assertEqual(summary.training_steps, 256)
        self.assertFalse(summary.failure)
        self.assertEqual(summary.ppo_metrics["loss/policy"], 0.2)

    def test_write_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs"
            output_path = Path(tmpdir) / "summary.json"
            run_dir = logs_dir / "target"
            run_dir.mkdir(parents=True)
            (run_dir / "run-1.json").write_text(
                json.dumps({"metrics": {"env/score": [1.0]}}),
                encoding="utf-8",
            )

            summaries = summarize_logs(logs_dir)
            write_summary(summaries, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["num_runs"], 1)
        self.assertEqual(payload["num_failures"], 0)
        self.assertEqual(payload["best_reward"], 1.0)

    def test_parse_pufferlib_3_metric_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "cartpole" / "run-1.json"
            log_path.parent.mkdir()
            log_path.write_text(
                json.dumps(
                    {
                        "env_name": "cartpole",
                        "metrics": {
                            "environment/score": [10.0, 12.0],
                            "losses/policy_loss": [0.5],
                            "performance/update": [0.1],
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = parse_log(log_path)

        self.assertEqual(summary.final_reward, 12.0)
        self.assertEqual(summary.best_reward, 12.0)
        self.assertEqual(summary.ppo_metrics["losses/policy_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
