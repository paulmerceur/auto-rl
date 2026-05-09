"""Validated LLM decision schema for constrained experiment management."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class DecisionAction(StrEnum):
    CONTINUE = "continue"
    NARROW_SEARCH = "narrow_search"
    EXPAND_SEARCH = "expand_search"
    STOP = "stop"


ALLOWED_SEARCH_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (1e-6, 1.0),
    "entropy_coef": (0.0, 1.0),
    "gamma": (0.0, 1.0),
    "clip_coef": (0.0, 1.0),
    "vf_coef": (0.0, 10.0),
    "max_grad_norm": (0.0, 10.0),
}


class SearchSpaceRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float
    max: float
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def validate_range(self) -> SearchSpaceRange:
        if self.min >= self.max:
            raise ValueError("min must be less than max")
        if self.scale == "log" and self.min <= 0:
            raise ValueError("log scale requires min > 0")
        return self


class LlmDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: DecisionAction
    reason: str = Field(min_length=1, max_length=500)
    search_space_update: dict[str, SearchSpaceRange] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("search_space_update")
    @classmethod
    def validate_search_space(
        cls, value: dict[str, SearchSpaceRange]
    ) -> dict[str, SearchSpaceRange]:
        for name, range_config in value.items():
            bounds = ALLOWED_SEARCH_BOUNDS.get(name)
            if bounds is None:
                raise ValueError(f"Unsupported search-space key: {name}")
            lower, upper = bounds
            if range_config.min < lower or range_config.max > upper:
                raise ValueError(
                    f"{name} range must stay within [{lower}, {upper}], "
                    f"got [{range_config.min}, {range_config.max}]"
                )
        return value


def parse_decision_json(raw: str | dict[str, Any]) -> LlmDecision:
    try:
        if isinstance(raw, str):
            return LlmDecision.model_validate_json(raw)
        return LlmDecision.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid LLM decision: {exc}") from exc


def decision_to_json(decision: LlmDecision) -> str:
    return json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True)
