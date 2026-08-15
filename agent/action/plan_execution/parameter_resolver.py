from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.planning.models import ActionStep, AgentPlan
from agent.planning.registry import CapabilityRegistry


class ParameterResolver:
    def __init__(self, registry: CapabilityRegistry, plan: AgentPlan) -> None:
        self.registry = registry
        self.objects = {
            scene_object.id: deepcopy(scene_object)
            for scene_object in plan.scene.objects
        }

    def resolve(self, step: ActionStep) -> dict[str, Any]:
        action_name = str(getattr(step.type, "value", step.type))
        try:
            action = self.registry.actions.get(action_name)
        except KeyError:
            raise KeyError(f"unknown action definition: {action_name}") from None

        object_id = getattr(step, "object", None)
        target_id = getattr(step, "target", None)
        if action.required_object and not object_id:
            raise ValueError(f"{action_name} requires object reference")
        if action.required_target and not target_id:
            raise ValueError(f"{action_name} requires target reference")

        scene_object = self._referenced_object(object_id) if object_id else None
        target_object = self._referenced_object(target_id) if target_id else None

        resolved_object = None
        if scene_object is not None:
            resolved_object = self.registry.assets.resolve(
                scene_object.asset_id,
                deepcopy(scene_object.properties),
            )
            if action_name not in resolved_object.supported_actions:
                raise ValueError(
                    f"object {object_id!r} asset {scene_object.asset_id!r} "
                    f"does not support {action_name}"
                )
            if resolved_object.category not in action.object_categories:
                raise ValueError(
                    f"object category {resolved_object.category!r} is not "
                    f"valid for action {action_name!r}"
                )

        if target_object is not None:
            resolved_target = self.registry.assets.resolve(
                target_object.asset_id,
                deepcopy(target_object.properties),
            )
            target_capability = (
                "place_target" if action_name == "place" else action_name
            )
            if target_capability not in resolved_target.supported_actions:
                raise ValueError(
                    f"target {target_id!r} asset {target_object.asset_id!r} "
                    f"does not support {target_capability}"
                )
            if resolved_target.category not in action.target_categories:
                raise ValueError(
                    f"target category {resolved_target.category!r} is not "
                    f"valid for action {action_name!r}"
                )

        result = deepcopy(action.default_parameters)
        if resolved_object is not None:
            result.update(
                deepcopy(resolved_object.action_defaults.get(action_name, {}))
            )
        result.update(deepcopy(getattr(step, "parameters", {})))
        action.validate_parameters(result)
        return deepcopy(result)

    def _referenced_object(self, object_id: str):
        try:
            return self.objects[object_id]
        except KeyError:
            raise KeyError(f"unknown scene object reference: {object_id}") from None
