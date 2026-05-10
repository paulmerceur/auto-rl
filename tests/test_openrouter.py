from __future__ import annotations

import unittest

from puffer_llm_sweeper.openrouter import OpenRouterClient, OpenRouterConfig, build_decision_messages


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"action":"stop","reason":"Target reached.","suggested_trials":1,'
                            '"search_space_update":{},"notes":[]}'
                        )
                    }
                }
            ]
        }


class FakeSession:
    def __init__(self) -> None:
        self.payload = None

    def post(self, *args, **kwargs) -> FakeResponse:
        self.payload = kwargs["json"]
        return FakeResponse()


class OpenRouterTests(unittest.TestCase):
    def test_mock_decision(self) -> None:
        client = OpenRouterClient(
            OpenRouterConfig(api_key=None, model="test/model", base_url="https://example.com")
        )

        decision = client.propose_decision({"runs": []}, dry_run=True)

        self.assertEqual(decision.action, "continue")

    def test_live_request_shape_with_fake_session(self) -> None:
        session = FakeSession()
        client = OpenRouterClient(
            OpenRouterConfig(api_key="test-key", model="test/model", base_url="https://example.com"),
            session=session,
        )

        decision = client.propose_decision({"runs": []}, dry_run=False)

        self.assertEqual(decision.action, "stop")
        self.assertEqual(session.payload["model"], "test/model")
        self.assertEqual(session.payload["response_format"], {"type": "json_object"})

    def test_prompt_includes_summary(self) -> None:
        messages = build_decision_messages(
            {"best_reward": 1.0},
            budget={"trials_remaining": 10},
            current_search_space={
                "train.learning_rate": {
                    "distribution": "log_normal",
                    "min": 0.00001,
                    "max": 0.1,
                    "scale": 0.5,
                }
            },
        )

        self.assertIn("valid JSON only", messages[0]["content"])
        self.assertIn('"best_reward": 1.0', messages[1]["content"])
        self.assertIn('"trials_remaining": 10', messages[1]["content"])
        self.assertIn('"train.ent_coef"', messages[1]["content"])
        self.assertIn("train.minibatch_size", messages[1]["content"])
        self.assertIn("Integer-only keys", messages[1]["content"])
        self.assertNotIn("policy.num_layers", messages[1]["content"])
        self.assertIn("Specific distribution requirements", messages[1]["content"])
        self.assertIn("Absolute search-space bounds", messages[1]["content"])
        self.assertIn("Current active search-space bounds", messages[1]["content"])
        self.assertIn("narrow_search may only update keys already present", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
