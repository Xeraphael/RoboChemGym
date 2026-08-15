from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from agent.action.plan_execution.parameter_resolver import ParameterResolver
from agent.planning.models import AgentPlan
from agent.planning.registry import CapabilityRegistry


class ParameterPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    parameters: dict[str, object]

    @field_validator("parameters")
    @classmethod
    def require_parameters(cls, value: dict[str, object]) -> dict[str, object]:
        if not value:
            raise ValueError("parameter patch must not be empty")
        return value


def apply_parameter_patch(
    plan: AgentPlan,
    patch: ParameterPatch,
    registry: CapabilityRegistry,
) -> AgentPlan:
    if not isinstance(patch, ParameterPatch):
        raise ValueError("patch must be a ParameterPatch")
    patch = ParameterPatch.model_validate(patch.model_dump(mode="python"))

    before = plan.model_dump(mode="python")
    updated = plan.model_copy(deep=True)
    step_index = next(
        (index for index, step in enumerate(updated.actions) if step.id == patch.step_id),
        None,
    )
    if step_index is None:
        raise ValueError(f"unknown step id: {patch.step_id}")

    step = updated.actions[step_index]
    definition = registry.actions.get(step.type.value)
    definition.validate_parameters(patch.parameters, tunable_only=True)
    step.parameters.update(deepcopy(patch.parameters))
    ParameterResolver(registry, updated).resolve(step)

    expected = deepcopy(before)
    expected["actions"][step_index]["parameters"].update(
        deepcopy(patch.parameters)
    )
    if updated.model_dump(mode="python") != expected:
        raise ValueError("parameter patch changed immutable plan structure")
    return updated


class PlanParameterOptimizer:
    def __init__(
        self,
        client: Any,
        registry: CapabilityRegistry,
        *,
        model: str,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model

    def propose(self, plan: AgentPlan, report: dict[str, Any]) -> ParameterPatch | None:
        if hasattr(report, "model_dump"):
            report = report.model_dump(mode="python")
        if not isinstance(report, dict):
            raise ValueError("execution report must be an object")

        failed_step_id = report.get("failed_step")
        if not isinstance(failed_step_id, str) or not failed_step_id:
            raise ValueError("execution report requires a failed step")
        step = next(
            (item for item in plan.actions if item.id == failed_step_id),
            None,
        )
        if step is None:
            raise ValueError(f"unknown failed step: {failed_step_id}")

        failed_record = next(
            (
                item
                for item in reversed(report.get("steps", []))
                if isinstance(item, dict) and item.get("step_id") == failed_step_id
            ),
            None,
        )
        if failed_record is None:
            raise ValueError("execution report is missing the failed-step verification record")
        verification = failed_record.get("verification")
        if not isinstance(verification, dict):
            raise ValueError("failed-step verification record must be an object")
        if (
            failed_record.get("success") is not False
            or verification.get("success") is not False
        ):
            raise ValueError(
                "failed record and verification must report success false"
            )
        code = verification.get("code")
        measurements = verification.get("measurements")
        if not isinstance(code, str) or not isinstance(measurements, dict):
            raise ValueError("failed-step verification requires code and measurements")

        definition = self.registry.actions.get(step.type.value)
        request = {
            "failed_step": {"id": step.id, "type": step.type.value},
            "verification": {
                "code": code,
                "measurements": deepcopy(measurements),
            },
            "current_parameters": ParameterResolver(
                self.registry,
                plan,
            ).resolve(step),
            "allowed_tunable_parameters": {
                name: definition.parameter_constraints[name].model_dump(mode="json")
                for name in definition.tunable_parameters
            },
        }
        raw = self.client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one ParameterPatch JSON object or null. "
                        "Only tune the allowlisted parameters for the failed step. "
                        "Do not return markdown, Python, source code, or edits to plan structure."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        request,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                },
            ],
            model=self.model,
            temperature=0,
        )
        if not isinstance(raw, str):
            raise ValueError("optimizer response must be JSON text")
        raw = raw.strip()
        if raw == "null":
            return None

        def reject_nonfinite_constant(value: str):
            raise ValueError(f"invalid JSON numeric constant: {value}")

        try:
            payload = json.loads(raw, parse_constant=reject_nonfinite_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("optimizer response must be a strict JSON object or null") from exc
        if not isinstance(payload, dict):
            raise ValueError("optimizer response must be a strict JSON object or null")

        patch = ParameterPatch.model_validate(payload)
        if patch.step_id != failed_step_id:
            raise ValueError("optimizer patch must target the failed step")
        definition.validate_parameters(patch.parameters, tunable_only=True)
        return patch
