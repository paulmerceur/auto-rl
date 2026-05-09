from __future__ import annotations

import unittest

from puffer_llm_sweeper.decisions import DecisionAction, parse_decision_json


class DecisionTests(unittest.TestCase):
    def test_valid_decision(self) -> None:
        decision = parse_decision_json(
            {
                "action": "narrow_search",
                "reason": "Best runs cluster at lower learning rates.",
                "suggested_trials": 3,
                "search_space_update": {
                    "train.learning_rate": {
                        "distribution": "log_normal",
                        "min": 0.0001,
                        "max": 0.001,
                        "scale": 0.5,
                    }
                },
                "notes": ["keep entropy range unchanged"],
            }
        )

        self.assertEqual(decision.action, DecisionAction.NARROW_SEARCH)
        self.assertEqual(decision.suggested_trials, 3)
        self.assertEqual(decision.search_space_update["train.learning_rate"].distribution, "log_normal")

    def test_accepts_puffer_train_entropy_name(self) -> None:
        decision = parse_decision_json(
            {
                "action": "expand_search",
                "reason": "Try more entropy.",
                "search_space_update": {
                    "train.ent_coef": {
                        "distribution": "log_normal",
                        "min": 0.001,
                        "max": 0.05,
                        "scale": "auto",
                    }
                },
                "notes": [],
            }
        )

        self.assertEqual(decision.search_space_update["train.ent_coef"].distribution, "log_normal")

    def test_accepts_zero_entropy_for_uniform_distribution(self) -> None:
        decision = parse_decision_json(
            {
                "action": "expand_search",
                "reason": "Allow disabling entropy regularization.",
                "search_space_update": {
                    "train.ent_coef": {
                        "distribution": "uniform",
                        "min": 0.0,
                        "max": 0.1,
                        "scale": "auto",
                    }
                },
                "notes": [],
            }
        )

        self.assertEqual(decision.search_space_update["train.ent_coef"].min, 0.0)

    def test_rejects_zero_entropy_for_log_distribution(self) -> None:
        with self.assertRaises(ValueError):
            parse_decision_json(
                {
                    "action": "expand_search",
                    "reason": "Invalid log range.",
                    "search_space_update": {
                        "train.ent_coef": {
                            "distribution": "log_normal",
                            "min": 0.0,
                            "max": 0.1,
                            "scale": "auto",
                        }
                    },
                    "notes": [],
                }
            )

    def test_accepts_minibatch_sweep_update(self) -> None:
        decision = parse_decision_json(
            {
                "action": "expand_search",
                "reason": "Try larger minibatches.",
                "search_space_update": {
                    "train.minibatch_size": {
                        "distribution": "uniform_pow2",
                        "min": 4096,
                        "max": 65536,
                        "scale": "auto",
                    }
                },
                "notes": [],
            }
        )

        self.assertEqual(decision.search_space_update["train.minibatch_size"].max, 65536)

    def test_rejects_float_distribution_for_integer_key(self) -> None:
        with self.assertRaises(ValueError):
            parse_decision_json(
                {
                    "action": "expand_search",
                    "reason": "This would produce float layer counts.",
                    "search_space_update": {
                        "policy.num_layers": {
                            "distribution": "uniform",
                            "min": 1,
                            "max": 8,
                            "scale": "auto",
                        }
                    },
                    "notes": [],
                }
            )

    def test_accepts_int_distribution_for_integer_key(self) -> None:
        decision = parse_decision_json(
            {
                "action": "expand_search",
                "reason": "Try deeper policies.",
                "search_space_update": {
                    "policy.num_layers": {
                        "distribution": "int_uniform",
                        "min": 1,
                        "max": 8,
                        "scale": "auto",
                    }
                },
                "notes": [],
            }
        )

        self.assertEqual(decision.search_space_update["policy.num_layers"].distribution, "int_uniform")

    def test_rejects_unknown_parameter(self) -> None:
        with self.assertRaises(ValueError):
            parse_decision_json(
                {
                    "action": "expand_search",
                    "reason": "Try a new thing.",
                    "search_space_update": {
                        "shell_command": {
                            "distribution": "uniform",
                            "min": 1,
                            "max": 2,
                            "scale": "auto",
                        }
                    },
                    "notes": [],
                }
            )

    def test_rejects_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            parse_decision_json(
                {
                    "action": "narrow_search",
                    "reason": "Too broad.",
                    "search_space_update": {
                        "train.learning_rate": {
                            "distribution": "log_normal",
                            "min": 0.0001,
                            "max": 2.0,
                            "scale": 0.5,
                        }
                    },
                    "notes": [],
                }
            )

    def test_rejects_too_many_suggested_trials(self) -> None:
        with self.assertRaises(ValueError):
            parse_decision_json(
                {
                    "action": "continue",
                    "reason": "Spend too much.",
                    "suggested_trials": 11,
                    "search_space_update": {},
                    "notes": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
