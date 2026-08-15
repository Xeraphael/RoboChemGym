from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from agent.action.plan_execution.models import (
    VerificationRequest,
    VerificationResult,
)


def expand_policy_action(action) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (8,) or not np.all(np.isfinite(values)):
        raise ValueError("policy action must be a finite [8] vector")
    return np.concatenate((values[:7], values[7:8], values[7:8]))


def bound_policy_action(action, current_joints, *, max_step: float) -> np.ndarray:
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive and finite")
    current = np.asarray(current_joints, dtype=np.float32)
    if current.shape != (9,) or not np.all(np.isfinite(current)):
        raise ValueError("current joints must be a finite [9] vector")
    target = expand_policy_action(action)
    return np.clip(target, current - max_step, current + max_step)


def validate_checkpoint_compatibility(
    compatibility: Mapping[str, Any],
    runtime_camera_order: list[str],
    runtime_camera_shapes: Mapping[str, list[int]],
) -> dict[str, Any]:
    required = {
        "model_type",
        "model_arguments",
        "dataset_schema_version",
        "split_version",
        "camera_order",
        "camera_shapes",
        "action_convention",
        "gripper_convention",
        "dataset_identity",
    }
    missing = sorted(required - set(compatibility))
    if missing:
        raise ValueError(
            "checkpoint compatibility fields are missing: " + ", ".join(missing)
        )
    if compatibility["model_type"] != "act":
        raise ValueError("evaluation supports ACT checkpoints only")
    if compatibility["dataset_schema_version"] != "1.0":
        raise ValueError("checkpoint dataset schema is incompatible")
    if compatibility["split_version"] != "1.0":
        raise ValueError("checkpoint split version is incompatible")
    identity = compatibility["dataset_identity"]
    if not isinstance(identity, Mapping) or not identity.get("episodes"):
        raise ValueError("checkpoint dataset identity is invalid")
    if compatibility["action_convention"] != "absolute_joint_position":
        raise ValueError("checkpoint action convention is incompatible")
    if (
        compatibility["gripper_convention"]
        != "first_finger_position_second_mirrored"
    ):
        raise ValueError("checkpoint gripper convention is incompatible")
    camera_order = list(compatibility["camera_order"])
    if camera_order != list(runtime_camera_order):
        raise ValueError("runtime camera order differs from checkpoint")
    camera_shapes = {
        name: list(shape)
        for name, shape in compatibility["camera_shapes"].items()
    }
    if camera_shapes != {
        name: list(shape) for name, shape in runtime_camera_shapes.items()
    }:
        raise ValueError("runtime camera shapes differ from checkpoint")
    arguments = dict(compatibility["model_arguments"])
    if list(arguments.get("camera_names", [])) != camera_order:
        raise ValueError("checkpoint model camera order is inconsistent")
    if arguments.get("robot_state_dim") != 8 or arguments.get("action_dim") != 8:
        raise ValueError("checkpoint state/action dimensions are incompatible")
    if not isinstance(arguments.get("num_queries"), int) or arguments["num_queries"] <= 0:
        raise ValueError("checkpoint action horizon is invalid")
    return arguments


class PassivePlanEvaluator:
    """Observe policy state without producing or modifying robot actions."""

    def __init__(self, plan, coverage, verifier_factories) -> None:
        self.plan = deepcopy(plan)
        self.coverage = dict(coverage)
        self.verifier_factories = dict(verifier_factories)
        missing = sorted(
            {
                step.type.value
                for step in self.plan.actions
                if step.type.value not in self.verifier_factories
            }
        )
        if missing:
            raise ValueError("passive verifiers are missing: " + ", ".join(missing))
        self.reset()

    def reset(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.index = 0
        self.step_start = 0
        self.last_result: VerificationResult | None = None

    @property
    def done(self) -> bool:
        return self.index >= len(self.plan.actions)

    @property
    def length(self) -> int:
        return len(self.history)

    def update(self, state: Mapping[str, Any]) -> bool:
        if self.done:
            return True
        self.history.append(deepcopy(dict(state)))
        while not self.done:
            step = self.plan.actions[self.index]
            verifier = self.verifier_factories[step.type.value]()
            result = verifier.verify(
                VerificationRequest(
                    step=step,
                    pre_state=self.history[self.step_start],
                    post_state=self.history[-1],
                    state_history=self.history[self.step_start :],
                    episode_initial_state=self.history[0],
                )
            )
            self.last_result = result
            if not result.success:
                break
            self.steps.append(
                self._step_report(
                    step,
                    result,
                    success=True,
                    achieved_frame=self.length,
                )
            )
            self.index += 1
            self.step_start = self.length - 1
            self.last_result = None
        return self.done

    def report(self) -> dict[str, Any]:
        if self.done:
            return {
                "status": "completed",
                "success": True,
                "failure_code": "OK",
                "failed_step": None,
                "length": self.length,
                "steps": deepcopy(self.steps),
            }

        step = self.plan.actions[self.index]
        result = self.last_result or VerificationResult(
            success=False,
            code="EVALUATION_STATE_MISSING",
            verification_level="state_observed",
        )
        steps = [
            *self.steps,
            self._step_report(step, result, success=False, achieved_frame=None),
        ]
        for pending in self.plan.actions[self.index + 1 :]:
            steps.append(
                {
                    "step_id": pending.id,
                    "action": pending.type.value,
                    "success": False,
                    "code": "NOT_REACHED",
                    "measurements": {},
                    "verification_level": "not_observed",
                    "coverage_level": self._coverage(pending.id),
                    "achieved_frame": None,
                }
            )
        return {
            "status": "completed",
            "success": False,
            "failure_code": result.code,
            "failed_step": step.id,
            "length": self.length,
            "steps": steps,
        }

    def _step_report(self, step, result, *, success, achieved_frame):
        return {
            "step_id": step.id,
            "action": step.type.value,
            "success": bool(success),
            "code": result.code,
            "measurements": deepcopy(result.measurements),
            "verification_level": result.verification_level,
            "coverage_level": self._coverage(step.id),
            "achieved_frame": achieved_frame,
        }

    def _coverage(self, step_id: str) -> str:
        value = self.coverage[step_id]
        return str(getattr(value, "value", value))
