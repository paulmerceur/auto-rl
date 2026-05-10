from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from puffer_llm_sweeper.runner import _json_safe, build_puffer_args, load_run_config, merge_config, write_returned_logs


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

    def test_write_returned_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "base.yaml"
            config_path.write_text("env_name: cartpole\noutput_dir: out\n", encoding="utf-8")
            config = load_run_config(config_path)
            config = type(config)(
                env_name=config.env_name,
                output_dir=Path(tmpdir) / config.output_dir,
                puffer_overrides=config.puffer_overrides,
            )

            output_path = write_returned_logs(
                config,
                {"train": {"total_timesteps": 10}},
                [{"environment/score": 1.0}, {"environment/score": 2.0}],
            )

            payload = output_path.read_text(encoding="utf-8")

        self.assertIn('"env_name": "cartpole"', payload)
        self.assertIn('"environment/score"', payload)

    def test_json_safe_encodes_bytes(self) -> None:
        self.assertEqual(_json_safe({"nccl_id": b"\x00\xff"}), {"nccl_id": "00ff"})

    def test_base_sweep_limits_active_parameters(self) -> None:
        class FakePufferl:
            @staticmethod
            def load_config(env_name: str) -> dict:
                self = FakePufferl
                self.env_name = env_name
                return {
                    "sweep": {
                        "vec": {
                            "num_buffers": {
                                "distribution": "uniform",
                                "min": 1,
                                "max": 8,
                                "scale": "auto",
                            }
                        },
                    },
                    "train": {},
                }

        config = load_run_config(Path("configs/base.yaml"))
        args = build_puffer_args(config, FakePufferl)

        self.assertIn("hidden_size", args["sweep"]["sweep_only"])
        self.assertNotIn("num_layers", args["sweep"]["sweep_only"])
        self.assertEqual(args["sweep"]["vec"]["num_buffers"]["distribution"], "int_uniform")


if __name__ == "__main__":
    unittest.main()
