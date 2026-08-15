from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omni.isaac.core.utils.types import ArticulationAction

from agent.action.plan_execution.verifiers import (
    CloseVerifier,
    OpenVerifier,
    PickVerifier,
    PlaceVerifier,
    PourVerifier,
    PressVerifier,
    PressZVerifier,
    ShakeVerifier,
)
from agent.planning.models import AgentPlan
from agent.planning.validator import ValidationReport, plan_fingerprint
from pipeline.dataset import atomic_json
from pipeline.evaluation import (
    PassivePlanEvaluator,
    bound_policy_action,
    validate_checkpoint_compatibility,
)
from pipeline.metrics import summarize_evaluation
from policy.model.common.normalizer import LinearNormalizer
from policy.policy.act_image_policy import ACTImagePolicy


VERIFIERS = {
    "pick": PickVerifier,
    "place": PlaceVerifier,
    "pour": PourVerifier,
    "press": PressVerifier,
    "press_z": PressZVerifier,
    "shake": ShakeVerifier,
    "open": OpenVerifier,
    "close": CloseVerifier,
}


def policy_action_to_articulation(action, current_joints, *, max_step) -> ArticulationAction:
    return ArticulationAction(
        joint_positions=bound_policy_action(
            action,
            current_joints,
            max_step=max_step,
        )
    )


def _state_snapshot(state, instance_names):
    snapshot = {
        key: value
        for key, value in state.items()
        if key not in {"camera_data", "camera_display"}
    }
    snapshot["plan_instance_names"] = instance_names
    return snapshot


class PolicyController:
    def __init__(self, cfg, robot):
        self.cfg = cfg
        self.robot = robot
        self.mode = "evaluate"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_needed = False
        self._last_success = False
        root = Path(__file__).resolve().parents[1]
        checkpoint_path = root / str(cfg.evaluation.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        required_checkpoint = {"compatibility", "model", "normalizer"}
        missing_checkpoint = sorted(required_checkpoint - set(checkpoint))
        if missing_checkpoint:
            raise ValueError(
                "checkpoint fields are missing: " + ", ".join(missing_checkpoint)
            )
        compatibility = checkpoint["compatibility"]
        runtime_camera_order = [
            f"{camera.name}_rgb"
            for camera in cfg.cameras
            if "rgb" in str(camera.image_type).split("+")
        ]
        runtime_camera_shapes = {
            f"{camera.name}_rgb": [
                3,
                int(camera.resolution[1]),
                int(camera.resolution[0]),
            ]
            for camera in cfg.cameras
            if "rgb" in str(camera.image_type).split("+")
        }
        model_arguments = validate_checkpoint_compatibility(
            compatibility,
            runtime_camera_order,
            runtime_camera_shapes,
        )
        self.policy = ACTImagePolicy(**model_arguments).to(self.device)
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(checkpoint["normalizer"])
        normalizer.to(self.device)
        self.policy.set_normalizer(normalizer)
        self.policy.load_state_dict(checkpoint["model"])
        self.policy.eval()
        self.camera_names = runtime_camera_order
        self.action_chunk = []
        self.max_steps = int(cfg.evaluation.max_steps)
        self.max_joint_step = float(cfg.evaluation.max_joint_step)
        self.plan = AgentPlan.model_validate_json(
            (root / str(cfg.agent.plan_path)).read_text(encoding="utf-8")
        )
        validation = ValidationReport.model_validate_json(
            (root / str(cfg.agent.validation_report_path)).read_text(encoding="utf-8")
        )
        step_ids = {step.id for step in self.plan.actions}
        if (
            not validation.valid
            or validation.plan_fingerprint != plan_fingerprint(self.plan)
            or set(validation.step_coverage) != step_ids
            or self.plan.unresolved_capabilities
        ):
            raise ValueError("evaluation plan validation contract is incompatible")
        self.coverage = validation.step_coverage
        self.instance_names = {
            item.id: item.instance_name for item in self.plan.scene.objects
        }
        self.output_dir = root / str(cfg.evaluation.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_id = f"{checkpoint_path.parent.parent.name}/{checkpoint_path.name}"
        self._episode_reports = []
        self._episode_metadata = None
        self.evaluator = PassivePlanEvaluator(
            self.plan,
            self.coverage,
            VERIFIERS,
        )

    def step(self, state):
        snapshot = _state_snapshot(state, self.instance_names)
        try:
            completed = self.evaluator.update(snapshot)
        except Exception as exc:
            return self._abort_step("EVALUATION_STATE_ERROR", exc)
        if completed or self.evaluator.length >= self.max_steps:
            report = self._enrich_report(self.evaluator.report())
            return self._complete(report)
        if not self.action_chunk:
            try:
                obs, joints = self._observation(state)
                with torch.no_grad():
                    predicted = (
                        self.policy.predict_action(obs)["action"][0].detach().cpu().numpy()
                    )
                if (
                    predicted.ndim != 2
                    or predicted.shape[1] != 8
                    or len(predicted) == 0
                    or not np.all(np.isfinite(predicted))
                ):
                    raise ValueError("policy prediction must be finite [horizon,8]")
                self.action_chunk = list(predicted)
            except Exception as exc:
                return self._abort_step("POLICY_INFERENCE_ERROR", exc)
        try:
            action = policy_action_to_articulation(
                self.action_chunk.pop(0),
                state["joint_positions"],
                max_step=self.max_joint_step,
            )
        except Exception as exc:
            return self._abort_step("POLICY_ACTION_INVALID", exc)
        return action, False, False

    def _observation(self, state):
        camera_data = state.get("camera_data", {})
        images = {}
        for name in self.camera_names:
            image = np.asarray(camera_data.get(name))
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"camera observation is incompatible: {name}")
            images[name] = (
                torch.from_numpy(np.ascontiguousarray(image))
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(self.device)
            )
        joints = np.asarray(state.get("joint_positions"), dtype=np.float32)
        if joints.shape != (9,) or not np.all(np.isfinite(joints)):
            raise ValueError("joint observation must be a finite [9] vector")
        images["agent_pose"] = torch.from_numpy(
            np.concatenate((joints[:7], joints[7:8]))
        ).unsqueeze(0).to(self.device)
        return images, joints

    def _enrich_report(self, report):
        metadata = self._episode_metadata or {}
        randomization = metadata.get("randomization")
        return {
            **report,
            "episode_index": len(self._episode_reports),
            "checkpoint_id": self.checkpoint_id,
            "config_id": str(self.cfg.name),
            "dataset_split": str(self.cfg.evaluation.dataset_split),
            "evaluation_seed_set": str(self.cfg.evaluation.seed_set),
            "randomization": randomization,
            "camera_calibration": metadata.get("camera_calibration", []),
        }

    def _complete(self, report):
        self._persist_report(report)
        self._last_success = bool(report["success"])
        self.reset_needed = True
        return None, True, self._last_success

    def _abort_step(self, code, exc):
        self.abort(code, f"{type(exc).__name__}: {exc}")
        return None, True, False

    def _persist_report(self, report):
        index = len(self._episode_reports)
        self._episode_reports.append(report)
        atomic_json(self.output_dir / f"episode_{index:04d}.json", report)

    def episode_num(self):
        return len(self._episode_reports)

    def need_reset(self):
        return self.reset_needed

    def abort(self, code, message=""):
        if self.reset_needed:
            return
        report = self.evaluator.report()
        report.update(
            status="failed",
            success=False,
            failure_code=str(code),
            failure_message=str(message),
        )
        self._complete(self._enrich_report(report))

    def start_evaluation_episode(self, metadata=None):
        self._episode_metadata = metadata or {}

    def reset(self):
        self.reset_needed = False
        self._last_success = False
        self.action_chunk = []
        self._episode_metadata = None
        self.evaluator.reset()

    def close(self):
        summary = {
            **summarize_evaluation(self._episode_reports),
            "checkpoint_id": self.checkpoint_id,
            "dataset_split": str(self.cfg.evaluation.dataset_split),
            "evaluation_seed_set": str(self.cfg.evaluation.seed_set),
            "config_id": str(self.cfg.name),
        }
        atomic_json(self.output_dir / "summary.json", summary)
