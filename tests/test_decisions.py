from __future__ import annotations

import unittest

from puffer_llm_sweeper.decisions import DecisionAction, parse_decision_json


class DecisionTests(unittest.TestCase):
    def test_valid_decision(self) -> None:
        decision = parse_decision_json(
            {
                "action": "narrow_search",
                "reason": "Best runs cluster at lower learning rates.",
                "search_space_update": {
                    "learning_rate": {"min": 0.0001, "max": 0.001, "scale": "log"}
                },
                "notes": ["keep entropy range unchanged"],
            }
        )

        self.assertEqual(decision.action, DecisionAction.NARROW_SEARCH)
        self.assertEqual(decision.search_space_update["learning_rate"].scale, "log")

    def test_rejects_unknown_parameter(self) -> None:
        with self.assertRaises(ValueError):
            parse_decision_json(
                {
                    "action": "expand_search",
                    "reason": "Try a new thing.",
                    "search_space_update": {
                        "shell_command": {"min": 1, "max": 2, "scale": "linear"}
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
                        "learning_rate": {"min": 0.0001, "max": 2.0, "scale": "log"}
                    },
                    "notes": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
