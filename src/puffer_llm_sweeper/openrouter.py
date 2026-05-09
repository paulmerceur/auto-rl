"""Tiny OpenRouter chat-completions client."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from puffer_llm_sweeper.decisions import LlmDecision, parse_decision_json


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

    def propose_decision(self, summary: dict[str, Any], dry_run: bool = True) -> LlmDecision:
        if dry_run:
            return parse_decision_json(
                {
                    "action": "continue",
                    "reason": "Mock decision: continue with the current search space.",
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
                "messages": build_decision_messages(summary),
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


def build_decision_messages(summary: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "You are assisting with RL experiment management. Given completed PufferLib "
        "runs, propose the next search-space adjustment. Do not invent metrics. Do "
        "not request arbitrary code execution. Output valid JSON only. The JSON must "
        "match the requested schema exactly."
    )
    user_prompt = (
        "Return one JSON object with exactly these top-level keys: action, reason, "
        "search_space_update, notes.\n"
        "action must be one of: continue, narrow_search, expand_search, stop.\n"
        "search_space_update must be an object whose keys are only: learning_rate, "
        "entropy_coef, gamma, clip_coef, vf_coef, max_grad_norm.\n"
        "Each search_space_update value must be an object with numeric min, numeric "
        "max, and scale equal to linear or log. Do not nest keys under train/env. "
        "Do not output lists of candidate values.\n"
        "Example: "
        '{"action":"narrow_search","reason":"short explanation",'
        '"search_space_update":{"learning_rate":{"min":0.0001,"max":0.001,'
        '"scale":"log"}},"notes":["short note"]}\n\n'
        f"Completed run summary:\n{json.dumps(summary, indent=2, sort_keys=True)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
