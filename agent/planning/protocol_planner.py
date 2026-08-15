from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.planning.llm_client import ChatClient
from agent.planning.models import AgentPlan
from agent.planning.validator import PlanValidator, ValidationReport, plan_fingerprint


PlanningStatus = Literal["valid", "planning_failed", "client_failed"]


class PlanningAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    raw_response: str
    parse_error: str | None = None
    validation_report: ValidationReport | None = None
    client_error: str | None = None

    @model_validator(mode="after")
    def require_one_outcome(self) -> "PlanningAttempt":
        outcomes = (
            self.parse_error,
            self.validation_report,
            self.client_error,
        )
        if sum(outcome is not None for outcome in outcomes) != 1:
            raise ValueError(
                "planning attempt requires exactly one of parse_error, "
                "validation_report, or client_error"
            )
        return self


class PlanningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PlanningStatus
    plan: AgentPlan | None = None
    final_report: ValidationReport | None = None
    attempts: list[PlanningAttempt] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state(self) -> "PlanningResult":
        if self.status == "valid":
            if self.plan is None:
                raise ValueError("valid planning result requires a plan")
            if self.final_report is None or not self.final_report.valid:
                raise ValueError("valid planning result requires a valid final_report")
        elif self.plan is not None:
            raise ValueError("non-valid planning result cannot carry a plan")

        if (
            self.plan is not None
            and self.final_report is not None
            and self.final_report.plan_fingerprint != plan_fingerprint(self.plan)
        ):
            raise ValueError("final_report fingerprint does not match plan")

        if self.status == "client_failed" and (
            not self.attempts or self.attempts[-1].client_error is None
        ):
            raise ValueError(
                "client_failed planning result requires a final client_error attempt"
            )
        if (
            self.status == "planning_failed"
            and self.final_report is not None
            and self.final_report.valid
        ):
            raise ValueError("planning_failed result cannot carry a valid final_report")
        return self

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


class ProtocolPlanningService:
    def __init__(
        self,
        client: ChatClient,
        validator: PlanValidator,
        root: Path,
        *,
        model: str,
    ):
        self.client = client
        self.validator = validator
        self.root = root
        self.model = model
        self.prompt = (
            root / "agent" / "planning" / "prompts" / "agent_plan_prompt.txt"
        ).read_text(encoding="utf-8")

    def create_plan(self, protocol_text: str, max_attempts: int = 3) -> PlanningResult:
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3 total attempts")

        attempts: list[PlanningAttempt] = []
        last_invalid_plan: AgentPlan | None = None
        last_validation_report: ValidationReport | None = None
        previous_error: dict[str, str] | None = None

        for index in range(1, max_attempts + 1):
            messages = self._messages(
                protocol_text,
                last_invalid_plan,
                last_validation_report,
                previous_error,
            )
            try:
                raw = self.client.complete(
                    messages,
                    model=self.model,
                    temperature=0.1,
                )
            except Exception as exc:
                attempts.append(
                    PlanningAttempt(
                        index=index,
                        raw_response="",
                        client_error=f"{type(exc).__name__}: {exc}",
                    )
                )
                return PlanningResult(
                    status="client_failed",
                    final_report=last_validation_report,
                    attempts=attempts,
                )

            raw_response = raw if isinstance(raw, str) else ""
            try:
                plan = AgentPlan.model_validate(self._extract_json(raw))
                self._normalize_instance_names(plan)
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                attempts.append(
                    PlanningAttempt(
                        index=index,
                        raw_response=raw_response,
                        parse_error=str(exc),
                    )
                )
                previous_error = {
                    "code": "PLAN_JSON_INVALID",
                    "message": str(exc),
                }
                continue

            report = self.validator.validate(plan)
            attempts.append(
                PlanningAttempt(
                    index=index,
                    raw_response=raw_response,
                    validation_report=report,
                )
            )
            if report.valid:
                return PlanningResult(
                    status="valid",
                    plan=plan,
                    final_report=report,
                    attempts=attempts,
                )

            last_invalid_plan = plan
            last_validation_report = report
            previous_error = None

        return PlanningResult(
            status="planning_failed",
            final_report=last_validation_report,
            attempts=attempts,
        )

    @staticmethod
    def _normalize_instance_names(plan: AgentPlan) -> None:
        variant_counts: dict[str, int] = {}
        used_names = {
            obj.instance_name
            for obj in plan.scene.objects
            if obj.asset_id != "ErlenmeyerFlask"
        }
        for obj in plan.scene.objects:
            if obj.asset_id != "ErlenmeyerFlask":
                continue
            phase = str(obj.properties.get("content_phase", "default"))
            prefix = {
                "solid": "ErlenmeyerFlask_Solid",
                "liquid": "ErlenmeyerFlask_Liquid",
            }.get(phase, "ErlenmeyerFlask")
            index = variant_counts.get(prefix, 0) + 1
            candidate = f"{prefix}{index}"
            while candidate in used_names:
                index += 1
                candidate = f"{prefix}{index}"
            variant_counts[prefix] = index
            used_names.add(candidate)
            obj.instance_name = candidate

    def _messages(
        self,
        protocol_text: str,
        previous_plan: AgentPlan | None,
        previous_report: ValidationReport | None,
        previous_error: dict[str, str] | None,
    ) -> list[dict[str, str]]:
        user = {"protocol": protocol_text}
        if previous_plan is not None and previous_report is not None:
            user["previous_plan"] = previous_plan.model_dump(mode="json")
            user["validator_errors"] = [
                issue.model_dump(mode="json") for issue in previous_report.issues
            ]
            user["repair_instruction"] = (
                "Return a complete replacement AgentPlan and resolve every blocked "
                "validator issue. Do not preserve an unresolved capability when the "
                "closest registered physical action exists and only its outcome or "
                "modifier is unobservable."
            )
        if previous_error is not None:
            user["previous_error"] = previous_error
        return [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": json.dumps(user, ensure_ascii=False),
            },
        ]

    def _system_prompt(self) -> str:
        registry_summary = {
            "assets": {
                asset_id: {
                    "aliases": definition.aliases,
                    "category": definition.category,
                    "variants": sorted(definition.variants),
                    "variant_property": definition.variant_property,
                    "supported_actions": definition.supported_actions,
                }
                for asset_id, definition in sorted(
                    self.validator.registry.assets.definitions.items()
                )
            },
            "actions": {
                name: {
                    "object_categories": definition.object_categories,
                    "target_categories": definition.target_categories,
                    "supported_parameters": definition.supported_parameters,
                    "parameter_constraints": {
                        parameter: constraint.model_dump(mode="json")
                        for parameter, constraint in sorted(
                            definition.parameter_constraints.items()
                        )
                    },
                    "default_parameters": definition.default_parameters,
                    "degradable_modifiers": definition.degradable_modifiers,
                }
                for name, definition in sorted(
                    self.validator.registry.actions.definitions.items()
                )
            },
        }
        return "\n\n".join(
            [
                self.prompt,
                "AgentPlan JSON schema:\n"
                + json.dumps(
                    AgentPlan.model_json_schema(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "Registry summary:\n"
                + json.dumps(
                    registry_summary,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ]
        )

    @staticmethod
    def _extract_json(raw: object):
        if not isinstance(raw, str):
            raise ValueError(
                f"response must be a string, got {type(raw).__name__}"
            )
        text = raw.strip()
        if not text:
            raise ValueError("response must contain nonempty JSON content")
        if text.startswith("```"):
            _, separator, fenced_content = text.partition("\n")
            if not separator:
                raise ValueError(
                    "fenced response requires an opening fence line and closing fence"
                )
            content_lines = fenced_content.splitlines()
            closing_index = next(
                (
                    index
                    for index, line in enumerate(content_lines)
                    if line.startswith("```")
                ),
                None,
            )
            if closing_index is None:
                raise ValueError("fenced response is missing a closing fence")
            closing_line = content_lines[closing_index]
            trailing_lines = content_lines[closing_index + 1 :]
            if closing_line.strip() != "```" or any(
                line.strip() for line in trailing_lines
            ):
                raise ValueError("fenced response has trailing text after closing fence")
            text = "\n".join(content_lines[:closing_index]).strip()
            if not text:
                raise ValueError("response must contain nonempty JSON content")
        return json.loads(text)
