"""Tiny OpenRouter chat-completions client."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from puffer_llm_sweeper.decisions import (
    ALLOWED_SEARCH_BOUNDS,
    INTEGER_SEARCH_KEYS,
    LlmDecision,
    parse_decision_json,
)


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4.1-mini"


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str | None
    model: str
    base_url: str


def load_openrouter_config() -> OpenRouterConfig:
    load_dotenv()
    return OpenRouterConfig(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        model=os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL,
        base_url=(os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
    )


class OpenRouterClient:
    def __init__(
        self,
        config: OpenRouterConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or load_openrouter_config()
        self.session = session or requests.Session()

    def propose_decision(
        self,
        summary: dict[str, Any],
        dry_run: bool = True,
        budget: dict[str, Any] | None = None,
    ) -> LlmDecision:
        if dry_run:
            return parse_decision_json(
                {
                    "action": "continue",
                    "reason": "Mock decision: continue with the current search space.",
                    "suggested_trials": 1,
                    "search_space_update": {},
                    "notes": ["dry-run"],
                }
            )

        if not self.config.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for live OpenRouter calls.")

        response = self.session.post(
            f"{self.config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/paulmerceur/auto-rl",
                "X-Title": "puffer-llm-sweeper",
            },
            json={
                "model": self.config.model,
                "messages": build_decision_messages(summary, budget=budget),
                "temperature": 0.2,
                "max_tokens": 700,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return parse_decision_json(content)


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as summary_file:
        payload = json.load(summary_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Summary must be a JSON object: {path}")
    return payload


def build_decision_messages(
    summary: dict[str, Any],
    budget: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are assisting with RL experiment management. Given completed PufferLib "
        "runs, propose the next search-space adjustment. Do not invent metrics. Do "
        "not request arbitrary code execution. Output valid JSON only. The JSON must "
        "match the requested schema exactly."
    )
    user_prompt = (
        "Return one JSON object with exactly these top-level keys: action, reason, "
        "suggested_trials, search_space_update, notes.\n"
        "action must be one of: continue, narrow_search, expand_search, stop.\n"
        "suggested_trials must be an integer from 1 to 10. It is only a soft "
        "recommendation and may be clamped by hard budgets.\n"
        "search_space_update must be an object whose keys are only: "
        f"{', '.join(sorted(ALLOWED_SEARCH_BOUNDS))}.\n"
        "Each search_space_update value must be an object with numeric min, numeric "
        "max, distribution equal to uniform, int_uniform, uniform_pow2, log_normal, "
        "or logit_normal, and scale as a number, auto, or time. Do not output lists "
        "of candidate values.\n"
        f"Absolute search-space bounds:\n{json.dumps(_bounds_payload(), indent=2, sort_keys=True)}\n"
        "For log_normal and uniform_pow2, min must be greater than 0. For "
        "logit_normal, min and max must be within [0, 1), and min must be less than max.\n"
        "Integer-only keys must use int_uniform or uniform_pow2. Integer-only keys are: "
        f"{', '.join(sorted(INTEGER_SEARCH_KEYS))}.\n"
        "Example: "
        '{"action":"narrow_search","reason":"short explanation","suggested_trials":3,'
        '"search_space_update":{"train.learning_rate":{"distribution":"log_normal",'
        '"min":0.0001,"max":0.001,"scale":0.5}},"notes":["short note"]}\n\n'
        f"Hard budget context:\n{json.dumps(budget or {}, indent=2, sort_keys=True)}\n\n"
        f"Completed run summary:\n{json.dumps(summary, indent=2, sort_keys=True)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _bounds_payload() -> dict[str, dict[str, float]]:
    return {
        name: {"min": lower, "max": upper}
        for name, (lower, upper) in ALLOWED_SEARCH_BOUNDS.items()
    }
