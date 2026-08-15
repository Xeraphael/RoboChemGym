from __future__ import annotations

from enum import Enum
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from agent.planning.models import ActionType, AgentPlan, AnnotationStatus, CoverageLevel
from agent.planning.registry import CapabilityRegistry


REGISTRY_FINGERPRINT_SCHEMA_VERSION = "capability-registry-execution-v1"


class IssueSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: IssueSeverity
    level: CoverageLevel
    message: str
    step_id: str | None = None
    field: str | None = None
    repair_hint: str | None = None


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    supported_count: int = Field(default=0, ge=0)
    degraded_count: int = Field(default=0, ge=0)
    blocked_count: int = Field(default=0, ge=0)
    step_coverage: dict[str, CoverageLevel] = Field(default_factory=dict)


class SymbolicState:
    def __init__(self, object_ids: list[str]):
        self.held_object: str | None = None
        self.object_locations = {object_id: "scene" for object_id in object_ids}
        self.device_states: dict[str, str] = {}


def _canonical_fingerprint(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def plan_fingerprint(plan: AgentPlan) -> str:
    return _canonical_fingerprint(plan.model_dump(mode="json"))


def registry_fingerprint(registry: CapabilityRegistry) -> str:
    payload = {
        "schema_version": REGISTRY_FINGERPRINT_SCHEMA_VERSION,
        "assets": {
            name: definition.model_dump(mode="json")
            for name, definition in registry.assets.definitions.items()
        },
        "actions": {
            name: definition.model_dump(mode="json")
            for name, definition in registry.actions.definitions.items()
        },
    }
    return _canonical_fingerprint(payload)


class PlanValidator:
    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    def validate(self, plan: AgentPlan) -> ValidationReport:
        issues: list[ValidationIssue] = []
        objects = {obj.id: obj for obj in plan.scene.objects}
        resolved = {}
        invalid_object_ids: set[str] = set()

        for obj in plan.scene.objects:
            if obj.asset_id not in self.registry.assets.definitions:
                invalid_object_ids.add(obj.id)
                issues.append(self._blocked(
                    "UNKNOWN_ASSET",
                    None,
                    "asset_id",
                    f"asset {obj.asset_id} is not registered",
                    "select an asset_id from AssetRegistry",
                ))
                continue
            definition = self.registry.assets.get(obj.asset_id)
            if definition.variants:
                variant = str(obj.properties.get(definition.variant_property or "", "default"))
                if variant not in definition.variants:
                    invalid_object_ids.add(obj.id)
                    issues.append(self._blocked(
                        "UNKNOWN_ASSET_VARIANT",
                        None,
                        f"scene.objects.{obj.id}.properties",
                        f"asset {obj.asset_id} has no variant {variant}",
                        f"use one of {sorted(definition.variants)}",
                    ))
            asset = self.registry.assets.resolve(obj.asset_id, obj.properties)
            resolved[obj.id] = asset
            if not (self.registry.root / asset.usd_path).is_file():
                invalid_object_ids.add(obj.id)
                issues.append(self._blocked(
                    "ASSET_FILE_MISSING",
                    None,
                    "asset_id",
                    f"registered asset path does not exist: {asset.usd_path}",
                    "repair the version-controlled asset registry before executing the plan",
                ))

        state = SymbolicState(list(objects))
        step_coverage: dict[str, CoverageLevel] = {}

        for step in plan.actions:
            step_issues_before = len(issues)
            if step.type.value not in self.registry.actions.definitions:
                issues.append(self._blocked(
                    "ACTION_NOT_REGISTERED",
                    step.id,
                    "type",
                    f"action {step.type.value} is not registered",
                    "register the action before executing the plan",
                ))
                step_coverage[step.id] = CoverageLevel.BLOCKED
                continue

            definition = self.registry.actions.get(step.type.value)
            references_invalid_asset = (
                step.object in invalid_object_ids or step.target in invalid_object_ids
            )

            if definition.required_object and not step.object:
                issues.append(self._blocked(
                    "MISSING_OBJECT",
                    step.id,
                    "object",
                    "action requires object",
                    "set object to a scene object id",
                ))
            if definition.required_target and not step.target:
                issues.append(self._blocked(
                    "MISSING_TARGET",
                    step.id,
                    "target",
                    "action requires target",
                    "set target to a scene object id",
                ))
            if step.object and step.object not in objects:
                issues.append(self._blocked(
                    "UNKNOWN_OBJECT_REFERENCE",
                    step.id,
                    "object",
                    f"unknown object {step.object}",
                    "reference an object declared in scene.objects",
                ))
            if step.target and step.target not in objects:
                issues.append(self._blocked(
                    "UNKNOWN_TARGET_REFERENCE",
                    step.id,
                    "target",
                    f"unknown target {step.target}",
                    "reference an object declared in scene.objects",
                ))

            if step.object in resolved and resolved[step.object].category not in definition.object_categories:
                issues.append(self._blocked(
                    "OBJECT_CATEGORY_UNSUPPORTED",
                    step.id,
                    "object",
                    "object category is not supported",
                    "select a compatible object",
                ))
            if step.target in resolved and resolved[step.target].category not in definition.target_categories:
                issues.append(self._blocked(
                    "TARGET_CATEGORY_UNSUPPORTED",
                    step.id,
                    "target",
                    "target category is not supported",
                    "select a compatible target",
                ))
            if step.object in resolved and step.type.value not in resolved[step.object].supported_actions:
                issues.append(self._blocked(
                    "ASSET_ACTION_UNSUPPORTED",
                    step.id,
                    "object",
                    "asset does not support this action",
                    "select an action supported by the asset",
                ))
            if step.target in resolved:
                target_capability = "place_target" if step.type == ActionType.PLACE else step.type.value
                if target_capability not in resolved[step.target].supported_actions:
                    issues.append(self._blocked(
                        "TARGET_ACTION_UNSUPPORTED",
                        step.id,
                        "target",
                        f"target does not support {target_capability}",
                        "select a target that supports the requested action",
                    ))

            try:
                definition.validate_parameters(step.parameters)
            except ValueError as exc:
                issues.append(self._blocked(
                    "INVALID_PARAMETER_VALUE",
                    step.id,
                    "parameters",
                    str(exc),
                    "use only registered parameters with values inside their declared ranges",
                ))
            unsupported_modifiers = set(step.modifiers) - set(definition.degradable_modifiers)
            for modifier in sorted(unsupported_modifiers):
                issues.append(self._blocked(
                    "UNSUPPORTED_MODIFIER",
                    step.id,
                    f"modifiers.{modifier}",
                    f"modifier {modifier} has no degradation rule",
                    "remove the modifier or use a registered one",
                ))

            structural_issues = issues[step_issues_before:]
            structurally_blocked = references_invalid_asset or any(
                issue.level == CoverageLevel.BLOCKED for issue in structural_issues
            )
            if not structurally_blocked:
                self._apply_sequence_rules(step, definition, state, issues)
            new_issues = issues[step_issues_before:]
            if references_invalid_asset or any(
                issue.level == CoverageLevel.BLOCKED for issue in new_issues
            ):
                step_coverage[step.id] = CoverageLevel.BLOCKED
            elif step.modifiers:
                step_coverage[step.id] = CoverageLevel.DEGRADED
            else:
                step_coverage[step.id] = CoverageLevel.SUPPORTED

        for annotation in plan.semantic_annotations:
            if annotation.status == AnnotationStatus.NOT_EXECUTABLE:
                issues.append(self._blocked(
                    "SEMANTIC_NOT_EXECUTABLE",
                    None,
                    "semantic_annotations",
                    f"{annotation.source_text}: {annotation.reason}",
                    "move the missing core action or asset to unresolved_capabilities and stop execution",
                ))
            else:
                issues.append(ValidationIssue(
                    code="SEMANTIC_DEGRADATION",
                    severity=IssueSeverity.WARNING,
                    level=CoverageLevel.DEGRADED,
                    message=f"{annotation.source_text}: {annotation.reason}",
                ))
            for step_id in annotation.step_ids:
                if annotation.status == AnnotationStatus.NOT_EXECUTABLE:
                    step_coverage[step_id] = CoverageLevel.BLOCKED
                elif step_coverage.get(step_id) == CoverageLevel.SUPPORTED:
                    step_coverage[step_id] = CoverageLevel.DEGRADED

        for capability in plan.unresolved_capabilities:
            issues.append(self._blocked(
                "UNRESOLVED_CAPABILITY",
                None,
                None,
                f"{capability.source_text}: {capability.reason}",
                "map to an existing asset and action or stop planning",
            ))

        blocked_count = sum(issue.level == CoverageLevel.BLOCKED for issue in issues)
        supported_count = sum(level == CoverageLevel.SUPPORTED for level in step_coverage.values())
        degraded_count = sum(
            bool(step.modifiers) and step_coverage.get(step.id) != CoverageLevel.BLOCKED
            for step in plan.actions
        ) + sum(
            annotation.status != AnnotationStatus.NOT_EXECUTABLE
            for annotation in plan.semantic_annotations
        )
        return ValidationReport(
            plan_fingerprint=plan_fingerprint(plan),
            registry_fingerprint=registry_fingerprint(self.registry),
            valid=blocked_count == 0,
            issues=issues,
            supported_count=supported_count,
            degraded_count=degraded_count,
            blocked_count=blocked_count,
            step_coverage=step_coverage,
        )

    def _apply_sequence_rules(self, step, definition, state: SymbolicState, issues: list[ValidationIssue]) -> None:
        failed = False
        for precondition in definition.preconditions:
            if precondition == "gripper_empty" and state.held_object is not None:
                issues.append(self._blocked(
                    "GRIPPER_OCCUPIED",
                    step.id,
                    "object",
                    f"gripper already holds {state.held_object}",
                    "place the held object before this action",
                ))
                failed = True
            elif precondition == "holding_object" and state.held_object != step.object:
                issues.append(self._blocked(
                    "OBJECT_NOT_HELD",
                    step.id,
                    "object",
                    f"{step.type.value} requires {step.object} to be held",
                    f"insert pick({step.object}) before this step",
                ))
                failed = True
            elif precondition == "target_closed" and state.device_states.get(step.target, "closed") != "closed":
                issues.append(self._blocked(
                    "TARGET_NOT_CLOSED",
                    step.id,
                    "target",
                    f"{step.target} is not closed",
                    "close the target before opening it again",
                ))
                failed = True
            elif precondition == "target_open" and state.device_states.get(step.target, "closed") != "open":
                issues.append(self._blocked(
                    "TARGET_NOT_OPEN",
                    step.id,
                    "target",
                    f"{step.target} is not open",
                    "insert open(target) before close(target)",
                ))
                failed = True
        if failed:
            return
        for effect in definition.effects:
            if effect == "hold_object":
                state.held_object = step.object
            elif effect == "release_object":
                state.held_object = None
            elif effect == "object_at_target":
                state.object_locations[step.object] = step.target
            elif effect == "target_activated":
                state.device_states[step.target] = "activated"
            elif effect == "target_open":
                state.device_states[step.target] = "open"
            elif effect == "target_closed":
                state.device_states[step.target] = "closed"

    @staticmethod
    def _blocked(code, step_id, field, message, repair_hint) -> ValidationIssue:
        return ValidationIssue(
            code=code,
            severity=IssueSeverity.ERROR,
            level=CoverageLevel.BLOCKED,
            step_id=step_id,
            field=field,
            message=message,
            repair_hint=repair_hint,
        )
