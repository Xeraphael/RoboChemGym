from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class ActionType(str, Enum):
    PICK = "pick"
    PLACE = "place"
    POUR = "pour"
    PRESS = "press"
    PRESS_Z = "press_z"
    SHAKE = "shake"
    OPEN = "open"
    CLOSE = "close"


class CoverageLevel(str, Enum):
    SUPPORTED = "supported"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class AnnotationStatus(str, Enum):
    NOT_OBSERVABLE = "not_observable"
    NOT_EXECUTABLE = "not_executable"
    APPROXIMATED = "approximated"


class SceneObject(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    asset_id: str = Field(min_length=1)
    instance_name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    role: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class ScenePlan(StrictModel):
    objects: list[SceneObject]

    @model_validator(mode="after")
    def unique_objects(self) -> "ScenePlan":
        ids = [obj.id for obj in self.objects]
        names = [obj.instance_name for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("scene object ids must be unique")
        if len(names) != len(set(names)):
            raise ValueError("scene instance names must be unique")
        return self


class ActionStep(StrictModel):
    id: str = Field(min_length=1, pattern=r"^step_[0-9]{3,}$")
    type: ActionType
    object: str | None = None
    target: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    modifiers: dict[str, Any] = Field(default_factory=dict)
    source_text: str = ""


class SemanticAnnotation(StrictModel):
    source_text: str
    status: AnnotationStatus
    reason: str
    step_ids: list[str] = Field(default_factory=list)


class UnresolvedCapability(StrictModel):
    source_text: str
    missing_asset: str | None = None
    missing_action: str | None = None
    reason: str


class AgentPlan(StrictModel):
    schema_version: str = "1.0"
    plan_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    metadata: dict[str, Any] = Field(default_factory=dict)
    scene: ScenePlan
    actions: list[ActionStep]
    semantic_annotations: list[SemanticAnnotation] = Field(default_factory=list)
    unresolved_capabilities: list[UnresolvedCapability] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_action_ids(self) -> "AgentPlan":
        ids = [step.id for step in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError("action step ids must be unique")
        unknown_annotation_steps = {
            step_id
            for annotation in self.semantic_annotations
            for step_id in annotation.step_ids
            if step_id not in ids
        }
        if unknown_annotation_steps:
            raise ValueError(
                f"semantic annotations reference unknown steps: {sorted(unknown_annotation_steps)}"
            )
        return self
