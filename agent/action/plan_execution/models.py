from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_numpy_scalars(value: Any) -> Any:
    value_type = type(value)
    is_numpy_scalar = any(
        base.__name__ == "generic"
        and base.__module__.split(".", 1)[0] == "numpy"
        for base in value_type.__mro__
    )
    if is_numpy_scalar:
        item = getattr(value, "item", None)
        if callable(item):
            return item()
    if isinstance(value, dict):
        return {
            key: _normalize_numpy_scalars(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_numpy_scalars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_numpy_scalars(item) for item in value)
    return value


class StrictExecutionModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


class VerificationResult(StrictExecutionModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    success: bool
    code: str = "OK"
    message: str = ""
    measurements: dict[str, Any] = Field(default_factory=dict)
    verification_level: str = "state_observed"

    @field_validator("success", mode="before")
    @classmethod
    def normalize_numpy_bool(cls, value: Any) -> Any:
        return _normalize_numpy_scalars(value)

    @field_validator("measurements", mode="before")
    @classmethod
    def normalize_numpy_measurements(cls, value: Any) -> Any:
        return _normalize_numpy_scalars(value)


class VerificationRequest(StrictExecutionModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    step: Any
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    state_history: list[dict[str, Any]] = Field(default_factory=list)
    episode_initial_state: dict[str, Any]


class StepExecutionRecord(StrictExecutionModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    step_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    object_id: str | None = None
    target_id: str | None = None
    coverage_level: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    verifier: str = Field(min_length=1)
    attempt_count: int = Field(ge=1)
    success: bool
    start_frame: int = Field(ge=1)
    end_frame: int = Field(ge=1)
    controller_completed: bool
    semantic_requirements: list[str] = Field(default_factory=list)
    verification: VerificationResult

    @model_validator(mode="after")
    def frame_span_is_ordered(self) -> "StepExecutionRecord":
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must not precede start_frame")
        if self.success != self.verification.success:
            raise ValueError("success must match verification.success")
        return self


class ExecutionReport(StrictExecutionModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    execution_success: bool = False
    failed_step: str | None = None
    steps: list[StepExecutionRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def success_state_is_consistent(self) -> "ExecutionReport":
        if self.execution_success and self.failed_step is not None:
            raise ValueError("successful reports cannot have failed_step")
        if self.execution_success and any(not step.success for step in self.steps):
            raise ValueError("successful reports cannot contain failure records")
        return self
