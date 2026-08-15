from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from agent.action.plan_execution.models import VerificationRequest, VerificationResult


_POSITION_SIZE = 3
_NUMERIC_ATOL = 1e-12
# Covers the three-ULP loss in the 0.88 - 0.80 lift calculation.
_THRESHOLD_ULP_BUDGET = 3
_SHAKE_REVERSAL_EXCURSION = 0.06


def _instance(request: VerificationRequest, ref: Any) -> str | None:
    if not isinstance(ref, str) or not ref:
        return None
    mapping_present = False
    for state in (
        request.post_state,
        request.pre_state,
        request.episode_initial_state,
    ):
        mapping = state.get("plan_instance_names")
        if not isinstance(mapping, Mapping):
            continue
        mapping_present = True
        instance = mapping.get(ref)
        if isinstance(instance, str) and instance:
            return instance
    if mapping_present:
        return None
    return ref.replace("_", "").title()


def _finite_array(value: Any, *, size: int | None = None, minimum: int = 0):
    try:
        array = np.asarray(value, dtype=float)
    except (OverflowError, TypeError, ValueError):
        return None
    if array.ndim != 1:
        return None
    if size is not None and array.size != size:
        return None
    if array.size < minimum or not bool(np.all(np.isfinite(array))):
        return None
    return array


def _state_vector(state: Mapping[str, Any], key: str):
    return _finite_array(state.get(key), size=_POSITION_SIZE)


def _position(state: Mapping[str, Any], instance: str | None, *suffixes: str):
    if instance is None:
        return None
    for suffix in suffixes:
        position = _state_vector(state, f"{instance}_{suffix}")
        if position is not None:
            return position
    return None


def _joints(state: Mapping[str, Any], minimum: int):
    return _finite_array(state.get("joint_positions"), minimum=minimum)


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        scalar = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _at_least(value: float, threshold: float) -> bool:
    slack = math.ulp(threshold) * _THRESHOLD_ULP_BUDGET
    return bool(value >= threshold - slack)


def _at_most(value: float, threshold: float) -> bool:
    slack = math.ulp(threshold) * _THRESHOLD_ULP_BUDGET
    return bool(value <= threshold + slack)


def _safe_subtract(left: Any, right: Any):
    with np.errstate(invalid="ignore", over="ignore"):
        result = np.subtract(left, right)
    return result if bool(np.all(np.isfinite(result))) else None


def _safe_norm(vector: Any) -> float | None:
    absolute = np.abs(vector)
    scale = float(np.max(absolute))
    if scale == 0.0:
        return 0.0
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        normalized = vector / scale
        norm = scale * np.sqrt(np.sum(normalized * normalized))
    return _finite_scalar(norm)


def _safe_distance(left: Any, right: Any) -> float | None:
    difference = _safe_subtract(left, right)
    return None if difference is None else _safe_norm(difference)


def _safe_mean_absolute(values: Any) -> float | None:
    absolute = np.abs(values)
    scale = float(np.max(absolute))
    if scale == 0.0:
        return 0.0
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        mean = scale * np.mean(absolute / scale)
    return _finite_scalar(mean)


def _safe_amplitude(values: Any) -> float | None:
    difference = _safe_subtract(np.max(values), np.min(values))
    return None if difference is None else _finite_scalar(difference)


def _safe_unit_vector(vector: Any):
    scale = float(np.max(np.abs(vector)))
    if scale <= _NUMERIC_ATOL:
        return None
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        scaled = vector / scale
    norm = _safe_norm(scaled)
    if norm is None or norm == 0.0:
        return None
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        unit = scaled / norm
    return unit if bool(np.all(np.isfinite(unit))) else None


def _meaningful_reversals(values: Any, minimum_excursion: float) -> int | None:
    anchor = float(values[0])
    extreme = anchor
    direction = 0
    reversals = 0
    for raw_value in values[1:]:
        value = float(raw_value)
        if direction == 0:
            upward = _safe_subtract(value, anchor)
            downward = _safe_subtract(anchor, value)
            if upward is None or downward is None:
                return None
            if _at_least(float(upward), minimum_excursion):
                direction = 1
                extreme = value
            elif _at_least(float(downward), minimum_excursion):
                direction = -1
                extreme = value
            continue

        if direction > 0:
            if value > extreme:
                extreme = value
                continue
            excursion = _safe_subtract(extreme, value)
            if excursion is None:
                return None
            if _at_least(float(excursion), minimum_excursion):
                reversals += 1
                direction = -1
                extreme = value
            continue

        if value < extreme:
            extreme = value
            continue
        excursion = _safe_subtract(value, extreme)
        if excursion is None:
            return None
        if _at_least(float(excursion), minimum_excursion):
            reversals += 1
            direction = 1
            extreme = value
    return reversals


def _planar_angle(center: Any, start: Any, end: Any) -> float | None:
    start_vector = _safe_subtract(start[:2], center[:2])
    end_vector = _safe_subtract(end[:2], center[:2])
    if start_vector is None or end_vector is None:
        return None
    start_unit = _safe_unit_vector(start_vector)
    end_unit = _safe_unit_vector(end_vector)
    if start_unit is None or end_unit is None:
        return None
    with np.errstate(invalid="ignore", over="ignore"):
        cross = float(
            start_unit[0] * end_unit[1] - start_unit[1] * end_unit[0]
        )
        dot = float(np.dot(start_unit, end_unit))
        angle = float(np.arctan2(cross, dot))
    return angle if np.isfinite(angle) else None


def _required_angle(step: Any) -> float | None:
    parameters = getattr(step, "parameters", {})
    if not isinstance(parameters, Mapping):
        return None
    requested = _finite_scalar(parameters.get("angle", 50.0))
    if requested is None or requested <= 0.0:
        return None
    return float(np.deg2rad(requested * 0.7))


class PickVerifier:
    stable_frames = 5

    def verify(self, request: VerificationRequest) -> VerificationResult:
        instance = _instance(request, getattr(request.step, "object", None))
        before = _position(request.pre_state, instance, "position")
        after = _position(request.post_state, instance, "position")
        if (
            before is None
            or after is None
            or len(request.state_history) < self.stable_frames
        ):
            return VerificationResult(
                success=False,
                code="PICK_STATE_MISSING",
                message="pick verification state is incomplete",
            )

        distances = []
        last_object = None
        last_gripper = None
        for state in request.state_history[-self.stable_frames :]:
            obj = _position(state, instance, "position")
            gripper = _state_vector(state, "gripper_position")
            if obj is None or gripper is None:
                return VerificationResult(
                    success=False,
                    code="PICK_STATE_MISSING",
                    message="pick verification state is incomplete",
                )
            distance = _safe_distance(obj, gripper)
            if distance is None:
                return VerificationResult(
                    success=False,
                    code="PICK_STATE_MISSING",
                    message="pick verification state is incomplete",
                )
            distances.append(distance)
            last_object = obj
            last_gripper = gripper

        post_gripper = None
        if "gripper_position" in request.post_state:
            post_gripper = _state_vector(request.post_state, "gripper_position")
        if (
            last_object is None
            or not np.array_equal(last_object, after)
            or (
                "gripper_position" in request.post_state
                and (
                    post_gripper is None
                    or not np.array_equal(last_gripper, post_gripper)
                )
            )
        ):
            return VerificationResult(
                success=False,
                code="PICK_STATE_MISSING",
                message="pick verification state is incomplete",
            )

        lift_difference = _safe_subtract(after[2], before[2])
        if lift_difference is None:
            return VerificationResult(
                success=False,
                code="PICK_STATE_MISSING",
                message="pick verification state is incomplete",
            )
        lift = float(lift_difference)
        stable_distance = float(max(distances))
        success = _at_least(lift, 0.08) and _at_most(stable_distance, 0.10)
        return VerificationResult(
            success=bool(success),
            code="OK" if success else "GRASP_NOT_ESTABLISHED",
            measurements={
                "lift": lift,
                "stable_object_gripper_distance": stable_distance,
                "stable_frames": int(self.stable_frames),
            },
            verification_level="state_observed",
        )


class PlaceVerifier:
    def verify(self, request: VerificationRequest) -> VerificationResult:
        obj = _instance(request, getattr(request.step, "object", None))
        target = _instance(request, getattr(request.step, "target", None))
        obj_pos = _position(request.post_state, obj, "position")
        target_pos = _position(
            request.post_state,
            target,
            "place_position",
            "position",
        )
        joints = _joints(request.post_state, 2)
        if obj_pos is None or target_pos is None or joints is None:
            return VerificationResult(success=False, code="PLACE_STATE_MISSING")

        distance = _safe_distance(obj_pos, target_pos)
        horizontal_distance = _safe_distance(obj_pos[:2], target_pos[:2])
        vertical_distance = _safe_distance(obj_pos[2:], target_pos[2:])
        gripper_opening = _safe_mean_absolute(joints[-2:])
        if (
            distance is None
            or horizontal_distance is None
            or vertical_distance is None
            or gripper_opening is None
        ):
            return VerificationResult(success=False, code="PLACE_STATE_MISSING")
        success = (
            _at_most(horizontal_distance, 0.08)
            and _at_most(vertical_distance, 0.20)
            and _at_least(gripper_opening, 0.025)
        )
        return VerificationResult(
            success=bool(success),
            code="OK" if success else "PLACE_OR_RELEASE_FAILED",
            measurements={
                "object_target_distance": distance,
                "horizontal_object_target_distance": horizontal_distance,
                "vertical_object_target_distance": vertical_distance,
                "gripper_opening": gripper_opening,
            },
            verification_level="state_observed",
        )


class PourVerifier:
    hold_frames = 3
    target_distance_threshold = 0.35

    def verify(self, request: VerificationRequest) -> VerificationResult:
        source = _instance(request, getattr(request.step, "object", None))
        target = _instance(request, getattr(request.step, "target", None))
        initial_joints = _joints(request.pre_state, 7)
        if source is None or target is None or initial_joints is None:
            return VerificationResult(
                success=False,
                code="POUR_STATE_MISSING",
                verification_level="motion_only",
            )

        initial_wrist = float(initial_joints[6])
        held_distances: list[float] = []
        target_distances: list[float] = []
        wrist_deltas: list[float] = []
        valid_observations = 0
        current_hold_frames = 0
        longest_hold_frames = 0
        required_wrist = float(np.deg2rad(45.0))
        for state in request.state_history:
            source_pos = _position(state, source, "position")
            target_pos = _position(state, target, "position")
            gripper = _state_vector(state, "gripper_position")
            joints = _joints(state, 7)
            if (
                source_pos is None
                or target_pos is None
                or gripper is None
                or joints is None
            ):
                current_hold_frames = 0
                continue
            held_distance = _safe_distance(source_pos, gripper)
            target_distance = _safe_distance(gripper, target_pos)
            wrist_difference = _safe_subtract(joints[6], initial_wrist)
            if (
                held_distance is None
                or target_distance is None
                or wrist_difference is None
            ):
                return VerificationResult(
                    success=False,
                    code="POUR_STATE_MISSING",
                    verification_level="motion_only",
                )
            wrist_delta = float(abs(wrist_difference))
            held_distances.append(held_distance)
            target_distances.append(target_distance)
            wrist_deltas.append(wrist_delta)
            valid_observations += 1
            if (
                _at_most(held_distance, 0.10)
                and _at_most(target_distance, self.target_distance_threshold)
                and _at_least(wrist_delta, required_wrist)
            ):
                current_hold_frames += 1
                longest_hold_frames = max(
                    longest_hold_frames,
                    current_hold_frames,
                )
            else:
                current_hold_frames = 0

        if valid_observations < self.hold_frames:
            return VerificationResult(
                success=False,
                code="POUR_STATE_MISSING",
                verification_level="motion_only",
            )

        max_held_distance = float(max(held_distances))
        success = (
            longest_hold_frames >= self.hold_frames
            and _at_most(max_held_distance, 0.10)
        )
        return VerificationResult(
            success=bool(success),
            code="OK" if success else "POUR_POSE_NOT_REACHED",
            measurements={
                "max_source_gripper_distance": max_held_distance,
                "minimum_gripper_target_distance": float(min(target_distances)),
                "maximum_wrist_delta": float(max(wrist_deltas)),
                "rotated_hold_frames": int(longest_hold_frames),
            },
            verification_level="motion_only",
        )


class PressVerifier:
    anchor = "press_position"
    target_pose_offset = (0.0, 0.0, 0.05)

    def verify(self, request: VerificationRequest) -> VerificationResult:
        target = _instance(request, getattr(request.step, "target", None))
        button_distances: list[float] = []
        target_pose_distances: list[float] = []
        target_pose_offset = np.asarray(self.target_pose_offset, dtype=float)
        for state in request.state_history:
            button = _position(state, target, self.anchor)
            gripper = _state_vector(state, "gripper_position")
            if button is not None and gripper is not None:
                target_pose = _safe_subtract(button, -target_pose_offset)
                button_distance = _safe_distance(gripper, button)
                target_pose_distance = _safe_distance(gripper, target_pose)
                if button_distance is None or target_pose_distance is None:
                    return VerificationResult(
                        success=False,
                        code="PRESS_STATE_MISSING",
                        verification_level="motion_only",
                    )
                button_distances.append(button_distance)
                target_pose_distances.append(target_pose_distance)

        before = _position(request.pre_state, target, self.anchor)
        after = _position(request.post_state, target, self.anchor)
        if not target_pose_distances or before is None or after is None:
            return VerificationResult(
                success=False,
                code="PRESS_STATE_MISSING",
                verification_level="motion_only",
            )

        minimum_button_distance = float(min(button_distances))
        minimum_target_pose_distance = float(min(target_pose_distances))
        button_displacement = _safe_distance(after, before)
        if button_displacement is None:
            return VerificationResult(
                success=False,
                code="PRESS_STATE_MISSING",
                verification_level="motion_only",
            )
        observed = _at_least(button_displacement, 0.003)
        success = _at_most(minimum_target_pose_distance, 0.04)
        return VerificationResult(
            success=bool(success),
            code="OK" if success else "BUTTON_NOT_REACHED",
            measurements={
                "minimum_gripper_button_distance": minimum_button_distance,
                "minimum_gripper_target_pose_distance": minimum_target_pose_distance,
                "button_displacement": button_displacement,
            },
            verification_level="state_observed" if observed else "motion_only",
        )


class PressZVerifier(PressVerifier):
    anchor = "pressz_position"
    target_pose_offset = (0.0, 0.0, 0.0)


class ShakeVerifier:
    def verify(self, request: VerificationRequest) -> VerificationResult:
        instance = _instance(request, getattr(request.step, "object", None))
        points = []
        held_distances = []
        tracking_missing = False
        for state in request.state_history:
            gripper = _state_vector(state, "gripper_position")
            if gripper is None:
                continue
            points.append(gripper)
            obj = _position(state, instance, "position")
            if obj is None:
                tracking_missing = True
                continue
            held_distance = _safe_distance(obj, gripper)
            if held_distance is None:
                return VerificationResult(
                    success=False,
                    code="SHAKE_OBJECT_TRACKING_MISSING",
                )
            held_distances.append(held_distance)

        if len(points) < 4:
            return VerificationResult(success=False, code="SHAKE_HISTORY_TOO_SHORT")
        if tracking_missing or len(held_distances) != len(points):
            return VerificationResult(
                success=False,
                code="SHAKE_OBJECT_TRACKING_MISSING",
            )

        lateral_positions = np.asarray([point[1] for point in points], dtype=float)
        amplitude = _safe_amplitude(lateral_positions)
        if amplitude is None:
            return VerificationResult(
                success=False,
                code="SHAKE_OBJECT_TRACKING_MISSING",
            )
        reversals = _meaningful_reversals(
            lateral_positions,
            _SHAKE_REVERSAL_EXCURSION,
        )
        if reversals is None:
            return VerificationResult(
                success=False,
                code="SHAKE_OBJECT_TRACKING_MISSING",
            )
        cycles = int(reversals // 2)
        max_held_distance = float(max(held_distances))
        success = (
            _at_least(amplitude, 0.12)
            and cycles >= 2
            and _at_most(max_held_distance, 0.10)
        )
        return VerificationResult(
            success=bool(success),
            code="OK" if success else "SHAKE_PATTERN_INCOMPLETE",
            measurements={
                "amplitude": amplitude,
                "cycles": cycles,
                "max_object_gripper_distance": max_held_distance,
            },
            verification_level="state_observed",
        )


class OpenVerifier:
    def verify(self, request: VerificationRequest) -> VerificationResult:
        target = _instance(request, getattr(request.step, "target", None))
        closed_handle = _position(
            request.episode_initial_state,
            target,
            "handle_position",
        )
        before_handle = _position(request.pre_state, target, "handle_position")
        after_handle = _position(request.post_state, target, "handle_position")
        center = _position(request.post_state, target, "revolute_joint_position")
        required = _required_angle(request.step)
        if (
            closed_handle is None
            or before_handle is None
            or after_handle is None
            or center is None
            or required is None
        ):
            return VerificationResult(success=False, code="OPEN_STATE_MISSING")

        pre_residual_angle = _planar_angle(
            center,
            closed_handle,
            before_handle,
        )
        final_residual_angle = _planar_angle(
            center,
            closed_handle,
            after_handle,
        )
        if pre_residual_angle is None or final_residual_angle is None:
            return VerificationResult(success=False, code="OPEN_STATE_MISSING")
        pre_residual = float(abs(pre_residual_angle))
        observed_angle = float(abs(final_residual_angle))
        progress_value = _safe_subtract(observed_angle, pre_residual)
        if progress_value is None:
            return VerificationResult(success=False, code="OPEN_STATE_MISSING")
        progress = float(progress_value)
        success = _at_least(observed_angle, required) and progress > 0.0
        return VerificationResult(
            success=bool(success),
            code="OK" if success else "OPEN_ANGLE_NOT_REACHED",
            measurements={
                "observed_angle": observed_angle,
                "required_angle": float(required),
                "pre_closed_pose_residual": pre_residual,
                "open_progress": progress,
            },
            verification_level="state_observed",
        )


class CloseVerifier:
    def verify(self, request: VerificationRequest) -> VerificationResult:
        target = _instance(request, getattr(request.step, "target", None))
        initial_handle = _position(
            request.episode_initial_state,
            target,
            "handle_position",
        )
        before_handle = _position(request.pre_state, target, "handle_position")
        after_handle = _position(request.post_state, target, "handle_position")
        center = _position(request.post_state, target, "revolute_joint_position")
        required_motion = _required_angle(request.step)
        if (
            initial_handle is None
            or before_handle is None
            or after_handle is None
            or center is None
            or required_motion is None
        ):
            return VerificationResult(success=False, code="CLOSE_STATE_MISSING")

        residual_angle = _planar_angle(center, initial_handle, after_handle)
        motion_angle = _planar_angle(center, before_handle, after_handle)
        if residual_angle is None or motion_angle is None:
            return VerificationResult(success=False, code="CLOSE_STATE_MISSING")
        residual = float(abs(residual_angle))
        motion = float(abs(motion_angle))
        residual_limit = float(np.deg2rad(12.0))
        success = _at_most(residual, residual_limit) and _at_least(
            motion,
            required_motion,
        )
        return VerificationResult(
            success=bool(success),
            code="OK" if success else "CLOSE_ANGLE_NOT_REACHED",
            measurements={
                "closed_pose_residual": residual,
                "observed_motion": motion,
                "required_motion": float(required_motion),
            },
            verification_level="state_observed",
        )
