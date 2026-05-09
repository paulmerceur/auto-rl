from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from puffer_llm_sweeper.runner import load_run_config, merge_config


class RunnerConfigTests(unittest.TestCase):
    def test_load_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "base.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "env_name: target",
                        "output_dir: runs",
                        "puffer:",
                        "  train:",
                        "    total_timesteps: 100000",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_run_config(config_path)

        self.assertEqual(config.env_name, "target")
        self.assertEqual(config.output_dir, Path("runs"))
        self.assertEqual(config.puffer_overrides["train"]["total_timesteps"], 100000)

    def test_merge_config_preserves_base(self) -> None:
        base = {"train": {"learning_rate": 0.01, "gamma": 0.99}}
        overrides = {"train": {"learning_rate": 0.001}}

        merged = merge_config(base, overrides)

        self.assertEqual(merged["train"], {"learning_rate": 0.001, "gamma": 0.99})
        self.assertEqual(base["train"]["learning_rate"], 0.01)


if __name__ == "__main__":
    unittest.main()
