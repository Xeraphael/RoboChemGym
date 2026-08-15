from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from agent.action.plan_execution.parameter_resolver import ParameterResolver
from agent.planning.models import ActionStep, AgentPlan


_DEFAULT_OBJECT_SIZE = np.array([0.04, 0.04, 0.08])
_ORIENTATION_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "agent"
    / "action"
    / "plan_execution"
    / "orientation_profiles.yaml"
)


class StateResolver:
    def __init__(self, plan: AgentPlan) -> None:
        self.instances = {
            scene_object.id: scene_object.instance_name
            for scene_object in plan.scene.objects
        }

    def instance(self, object_id: str) -> str:
        try:
            return self.instances[object_id]
        except KeyError:
            raise KeyError(f"unknown plan object id: {object_id}") from None

    def position(
        self,
        state: Mapping[str, Any],
        object_id: str,
        *anchors: str,
    ) -> np.ndarray:
        position, _ = self.position_with_anchor(
            state, object_id, *anchors
        )
        return position

    def position_with_anchor(
        self,
        state: Mapping[str, Any],
        object_id: str,
        *anchors: str,
    ) -> tuple[np.ndarray, str]:
        instance = self.instance(object_id)
        keys = []
        for anchor in anchors:
            key = f"{instance}_{anchor}"
            keys.append(key)
            if key in state:
                return np.asarray(state[key]).copy(), anchor
        expected = ", ".join(keys)
        raise KeyError(
            f"missing state position for {object_id} ({instance}); "
            f"expected one of: {expected}"
        )


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@lru_cache(maxsize=1)
def _load_orientation_profiles_cached() -> tuple[
    tuple[str, tuple[float, float, float]], ...
]:
    path = _ORIENTATION_PROFILE_PATH
    try:
        data = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_DuplicateKeySafeLoader,
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"invalid orientation profiles at {path}: {exc}"
        ) from exc

    if not isinstance(data, dict) or set(data) != {"profiles"}:
        raise ValueError(
            f"invalid orientation profiles at {path}: "
            "top-level mapping must contain only 'profiles'"
        )
    profiles = data["profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(
            f"invalid orientation profiles at {path}: "
            "'profiles' must be a non-empty mapping"
        )

    normalized: list[tuple[str, tuple[float, float, float]]] = []
    for name, values in profiles.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"invalid orientation profiles at {path}: "
                "profile names must be non-empty strings"
            )
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(
                f"invalid orientation profiles at {path}: "
                f"profile {name!r} must contain exactly three values"
            )
        components: list[float] = []
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"invalid orientation profiles at {path}: "
                    f"profile {name!r} values must be finite numbers"
                )
            components.append(float(value))
        normalized.append((name, tuple(components)))
    return tuple(normalized)


def load_orientation_profiles() -> dict[str, list[float]]:
    return {
        name: list(values)
        for name, values in _load_orientation_profiles_cached()
    }


load_orientation_profiles.cache_clear = _load_orientation_profiles_cached.cache_clear


def orientation(parameters: Mapping[str, Any]) -> np.ndarray:
    profile = parameters["orientation_profile"]
    profiles = dict(_load_orientation_profiles_cached())
    try:
        euler_degrees = profiles[profile]
    except KeyError:
        raise KeyError(f"unknown orientation profile: {profile}") from None
    quaternion_xyzw = Rotation.from_euler(
        "xyz", euler_degrees, degrees=True
    ).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


class BaseActionAdapter:
    def __init__(
        self,
        controller,
        resolver: StateResolver,
        parameters: ParameterResolver,
    ) -> None:
        self.controller = controller
        self.resolver = resolver
        self.parameters = parameters
        self.step_def: ActionStep | None = None
        self.resolved_parameters: dict[str, Any] | None = None

    def prepare(self, step: ActionStep, context) -> None:
        del context
        self.step_def = deepcopy(step)
        self.resolved_parameters = self.parameters.resolve(self.step_def)

    def is_done(self):
        return self.controller.is_done()

    def reset(self) -> None:
        self.controller.reset()
        self.step_def = None
        self.resolved_parameters = None

    def _prepared(self) -> tuple[ActionStep, dict[str, Any]]:
        if self.step_def is None or self.resolved_parameters is None:
            raise RuntimeError("adapter must be prepared before step")
        return self.step_def, self.resolved_parameters


class PickActionAdapter(BaseActionAdapter):
    def __init__(
        self,
        controller,
        gripper_control,
        resolver: StateResolver,
        parameters: ParameterResolver,
    ) -> None:
        super().__init__(controller, resolver, parameters)
        self.gripper_control = gripper_control

    def prepare(self, step: ActionStep, context) -> None:
        self.gripper_control.release_object()
        super().prepare(step, context)

    def step(self, state: Mapping[str, Any]):
        step, parameters = self._prepared()
        object_id = _required_reference(step.object, "object", step.id)
        instance = self.resolver.instance(object_id)
        picking_position, picking_anchor = self.resolver.position_with_anchor(
            state, object_id, "grisp_position", "position"
        )
        action = self.controller.forward(
            picking_position=picking_position,
            current_joint_positions=_state_array(state, "joint_positions"),
            object_name=instance,
            object_size=_optional_state_array(
                state, f"{instance}_size", _DEFAULT_OBJECT_SIZE
            ),
            gripper_control=self.gripper_control,
            gripper_position=_state_array(state, "gripper_position"),
            end_effector_orientation=orientation(parameters),
            pre_offset_z=parameters["pre_offset_z"],
            after_offset_z=parameters["after_offset_z"],
            pre_offset_x=parameters["pre_offset_x"],
            gripper_distances=parameters.get("gripper_distance"),
            object_prim_path=f"/World/{instance}",
            pick_z_offset=(
                0.0 if picking_anchor == "grisp_position" else None
            ),
        )
        self.gripper_control.update_grasped_object_position()
        return action


class PlaceActionAdapter(BaseActionAdapter):
    def __init__(
        self,
        controller,
        gripper_control,
        resolver: StateResolver,
        parameters: ParameterResolver,
    ) -> None:
        super().__init__(controller, resolver, parameters)
        self.gripper_control = gripper_control

    def prepare(self, step: ActionStep, context) -> None:
        self.gripper_control.begin_position_tracking()
        super().prepare(step, context)

    def step(self, state: Mapping[str, Any]):
        step, parameters = self._prepared()
        object_id = _required_reference(step.object, "object", step.id)
        target_id = _required_reference(step.target, "target", step.id)
        self.gripper_control.update_grasped_object_position()
        return self.controller.forward(
            place_position=self.resolver.position(
                state, target_id, "place_position", "position"
            ),
            current_joint_positions=_state_array(state, "joint_positions"),
            gripper_control=self.gripper_control,
            gripper_position=_state_array(state, "gripper_position"),
            object_position=self.resolver.position(
                state, object_id, "position"
            ),
            end_effector_orientation=orientation(parameters),
            pre_place_z=parameters["pre_place_z"],
            place_offset_z=parameters["place_offset_z"],
        )


class PourActionAdapter(BaseActionAdapter):
    def __init__(
        self,
        controller,
        robot,
        gripper_control,
        resolver: StateResolver,
        parameters: ParameterResolver,
    ) -> None:
        super().__init__(controller, resolver, parameters)
        self.robot = robot
        self.gripper_control = gripper_control

    def prepare(self, step: ActionStep, context) -> None:
        super().prepare(step, context)
        gripper_frame_path = self.gripper_control.gripper_frame_path
        if not gripper_frame_path:
            raise ValueError("gripper frame path is required for pour pose tracking")
        if not self.gripper_control.grasped_object_path:
            raise ValueError("grasped object path is required for pour pose tracking")

    def step(self, state: Mapping[str, Any]):
        step, parameters = self._prepared()
        object_id = _required_reference(step.object, "object", step.id)
        target_id = _required_reference(step.target, "target", step.id)
        source = self.resolver.instance(object_id)
        if self.controller._event == 2 and self.gripper_control._relative_mat is None:
            self.gripper_control.init_pose_tracking(
                self.gripper_control.grasped_object_path,
                self.gripper_control.gripper_frame_path,
            )
        action = self.controller.forward(
            articulation_controller=self.robot.get_articulation_controller(),
            source_size=_optional_state_array(
                state, f"{source}_size", _DEFAULT_OBJECT_SIZE
            ),
            target_position=self.resolver.position(
                state, target_id, "position"
            ),
            current_joint_velocities=np.asarray(
                self.robot.get_joint_velocities()
            ).copy(),
            current_joint_positions=_state_array(state, "joint_positions"),
            gripper_position=_state_array(state, "gripper_position"),
            source_name=source,
            pour_speed=parameters.get("pour_speed"),
            target_end_effector_orientation=orientation(parameters),
        )
        if self.controller._event >= 2:
            self.gripper_control.update_grasped_object_pose()
        else:
            self.gripper_control.update_grasped_object_position()
        return action


class PressActionAdapter(BaseActionAdapter):
    anchor = "press_position"

    def __init__(
        self,
        controller,
        gripper_control,
        resolver: StateResolver,
        parameters: ParameterResolver,
    ) -> None:
        super().__init__(controller, resolver, parameters)
        self.gripper_control = gripper_control

    def prepare(self, step: ActionStep, context) -> None:
        super().prepare(step, context)
        _, parameters = self._prepared()
        self.controller.reset(initial_offset=parameters["press_distance"])

    def step(self, state: Mapping[str, Any]):
        step, parameters = self._prepared()
        target_id = _required_reference(step.target, "target", step.id)
        return self.controller.forward(
            target_position=self.resolver.position(
                state, target_id, self.anchor
            ),
            current_joint_positions=_state_array(state, "joint_positions"),
            gripper_control=self.gripper_control,
            gripper_position=_state_array(state, "gripper_position"),
            end_effector_orientation=orientation(parameters),
        )


class PressZActionAdapter(PressActionAdapter):
    anchor = "pressz_position"

    def prepare(self, step: ActionStep, context) -> None:
        BaseActionAdapter.prepare(self, step, context)
        _, parameters = self._prepared()
        self.controller.reset(press_distance=parameters["press_distance"])


class ShakeActionAdapter(BaseActionAdapter):
    def __init__(
        self,
        controller,
        gripper_control,
        resolver: StateResolver,
        parameters: ParameterResolver,
    ) -> None:
        super().__init__(controller, resolver, parameters)
        self.gripper_control = gripper_control

    def prepare(self, step: ActionStep, context) -> None:
        super().prepare(step, context)
        _, parameters = self._prepared()
        self.controller.reset(shake_distance=parameters["shake_distance"])

    def step(self, state: Mapping[str, Any]):
        _, parameters = self._prepared()
        self.gripper_control.update_grasped_object_position()
        return self.controller.forward(
            current_joint_positions=_state_array(state, "joint_positions"),
            gripper_position=_state_array(state, "gripper_position"),
            end_effector_orientation=orientation(parameters),
        )


class OpenActionAdapter(BaseActionAdapter):
    def prepare(self, step: ActionStep, context) -> None:
        super().prepare(step, context)
        _, parameters = self._prepared()
        self.controller.reset(furniture_type=parameters["furniture_type"])

    def step(self, state: Mapping[str, Any]):
        step, parameters = self._prepared()
        target_id = _required_reference(step.target, "target", step.id)
        return self.controller.forward(
            handle_position=self.resolver.position(
                state, target_id, "handle_position"
            ),
            current_joint_positions=_state_array(state, "joint_positions"),
            gripper_position=_state_array(state, "gripper_position"),
            revolute_joint_position=self.resolver.position(
                state, target_id, "revolute_joint_position"
            ),
            end_effector_orientation=orientation(parameters),
            angle=parameters["angle"],
        )


class CloseActionAdapter(BaseActionAdapter):
    def prepare(self, step: ActionStep, context) -> None:
        super().prepare(step, context)
        _, parameters = self._prepared()
        self.controller.reset(furniture_type=parameters["furniture_type"])

    def step(self, state: Mapping[str, Any]):
        step, parameters = self._prepared()
        target_id = _required_reference(step.target, "target", step.id)
        return self.controller.forward(
            handle_position=self.resolver.position(
                state, target_id, "handle_position"
            ),
            current_joint_positions=_state_array(state, "joint_positions"),
            gripper_position=_state_array(state, "gripper_position"),
            end_effector_orientation=orientation(parameters),
            angle=parameters["angle"],
            revolute_joint_position=self.resolver.position(
                state, target_id, "revolute_joint_position"
            ),
            push_distance=parameters.get("push_distance"),
        )


def _required_reference(value: str | None, field: str, step_id: str) -> str:
    if value is None:
        raise ValueError(f"{step_id} requires {field} reference")
    return value


def _state_array(state: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(state[key]).copy()


def _optional_state_array(
    state: Mapping[str, Any], key: str, default: np.ndarray
) -> np.ndarray:
    if key not in state:
        return default.copy()
    return np.asarray(state[key]).copy()
