from pathlib import Path

from agent.action.plan_execution.executor import SequentialPlanExecutor
from agent.action.plan_execution.parameter_resolver import ParameterResolver
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
from agent.planning.models import AgentPlan, AnnotationStatus, CoverageLevel
from agent.planning.registry import CapabilityRegistry
from agent.planning.validator import (
    PlanValidator,
    ValidationReport,
    plan_fingerprint,
    registry_fingerprint,
)
from controllers.atomic_actions.close_controller import CloseController
from controllers.atomic_actions.open_controller import OpenController
from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from controllers.atomic_actions.pour_controller import PourController
from controllers.atomic_actions.press_controller import PressController
from controllers.atomic_actions.pressZ_controller import PressZController
from controllers.atomic_actions.shake_controller import ShakeController
from controllers.base_controller import BaseController
from controllers.plan_action_adapters import (
    CloseActionAdapter,
    OpenActionAdapter,
    PickActionAdapter,
    PlaceActionAdapter,
    PourActionAdapter,
    PressActionAdapter,
    PressZActionAdapter,
    ShakeActionAdapter,
    StateResolver,
)


def _resolve_artifact_path(root: Path, configured_path) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _load_trajectory_recorder():
    from agent.action.rating.trajectory_recorder import TrajectoryRecorder

    return TrajectoryRecorder


def _validate_execution_contract(plan: AgentPlan, validation: ValidationReport) -> None:
    if validation.plan_fingerprint != plan_fingerprint(plan):
        raise ValueError("validation report fingerprint does not match plan")
    if plan.unresolved_capabilities:
        raise ValueError("plan contains unresolved capabilities")
    if not validation.valid:
        raise ValueError("validation report is not valid")
    if any(issue.level == CoverageLevel.BLOCKED for issue in validation.issues):
        raise ValueError("validation report contains a blocked validation issue")

    step_ids = [step.id for step in plan.actions]
    coverage_step_ids = set(validation.step_coverage)
    expected_step_ids = set(step_ids)
    missing = sorted(expected_step_ids - coverage_step_ids)
    if missing:
        raise ValueError(
            "missing validation coverage for steps: " + ", ".join(missing)
        )
    extra = sorted(coverage_step_ids - expected_step_ids)
    if extra:
        raise ValueError(
            "validation coverage steps are not in the plan: " + ", ".join(extra)
        )

    blocked = [
        step_id
        for step_id in step_ids
        if validation.step_coverage[step_id] == CoverageLevel.BLOCKED
    ]
    if blocked:
        raise ValueError(
            "blocked validation coverage for steps: " + ", ".join(blocked)
        )
    if validation.blocked_count > 0:
        raise ValueError("validation report blocked_count must be zero")

    supported_count = sum(
        level == CoverageLevel.SUPPORTED
        for level in validation.step_coverage.values()
    )
    degraded_count = sum(
        bool(step.modifiers)
        and validation.step_coverage[step.id] != CoverageLevel.BLOCKED
        for step in plan.actions
    ) + sum(
        annotation.status != AnnotationStatus.NOT_EXECUTABLE
        for annotation in plan.semantic_annotations
    )
    if (
        validation.supported_count != supported_count
        or validation.degraded_count != degraded_count
    ):
        raise ValueError("validation coverage counts are inconsistent")


def _validate_current_report(
    plan: AgentPlan,
    stored_validation: ValidationReport,
    registry: CapabilityRegistry,
) -> ValidationReport:
    if stored_validation.registry_fingerprint != registry_fingerprint(registry):
        raise ValueError(
            "stored validation registry fingerprint does not match current "
            "registry; recompile artifacts"
        )
    current_validation = PlanValidator(registry).validate(plan)
    try:
        _validate_execution_contract(plan, current_validation)
    except ValueError as exc:
        raise ValueError(
            f"current plan validation failed; recompile artifacts: {exc}"
        ) from exc
    if stored_validation.model_dump(mode="json") != current_validation.model_dump(
        mode="json"
    ):
        raise ValueError(
            "stored validation report does not match current validation; "
            "recompile artifacts"
        )
    return current_validation


class PlanExecutorController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.mode = getattr(self, "mode", "execute")

        root = Path(__file__).resolve().parents[1]
        registry = CapabilityRegistry.load_default(root)
        self.plan_path = _resolve_artifact_path(root, cfg.agent.plan_path)
        self.validation_report_path = _resolve_artifact_path(
            root, cfg.agent.validation_report_path
        )
        self.report_path = _resolve_artifact_path(
            root, cfg.agent.execution_report_path
        )
        self.trajectory_path = _resolve_artifact_path(
            root, cfg.agent.trajectory_path
        )

        self.plan = AgentPlan.model_validate_json(
            self.plan_path.read_text(encoding="utf-8")
        )
        validation = ValidationReport.model_validate_json(
            self.validation_report_path.read_text(encoding="utf-8")
        )
        _validate_execution_contract(self.plan, validation)
        validation = _validate_current_report(self.plan, validation, registry)
        self.plan_identity = plan_fingerprint(self.plan)
        self.validation_identity = validation.plan_fingerprint

        resolver = StateResolver(self.plan)
        parameters = ParameterResolver(registry, self.plan)

        atomic_controllers = {
            "pick": PickController(
                name="plan_pick",
                cspace_controller=self.rmp_controller,
            ),
            "place": PlaceController(
                name="plan_place",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                _position_threshold=0.06,
            ),
            "pour": PourController(
                name="plan_pour",
                cspace_controller=self.rmp_controller,
            ),
            "press": PressController(
                name="plan_press",
                cspace_controller=self.rmp_controller,
            ),
            "press_z": PressZController(
                name="plan_press_z",
                cspace_controller=self.rmp_controller,
            ),
            "shake": ShakeController(
                name="plan_shake",
                cspace_controller=self.rmp_controller,
            ),
            "open": OpenController(
                name="plan_open",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
            ),
            "close": CloseController(
                name="plan_close",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
            ),
        }
        adapters = {
            "pick": PickActionAdapter(
                atomic_controllers["pick"],
                self.gripper_control,
                resolver,
                parameters,
            ),
            "place": PlaceActionAdapter(
                atomic_controllers["place"],
                self.gripper_control,
                resolver,
                parameters,
            ),
            "pour": PourActionAdapter(
                atomic_controllers["pour"],
                robot,
                self.gripper_control,
                resolver,
                parameters,
            ),
            "press": PressActionAdapter(
                atomic_controllers["press"],
                self.gripper_control,
                resolver,
                parameters,
            ),
            "press_z": PressZActionAdapter(
                atomic_controllers["press_z"],
                self.gripper_control,
                resolver,
                parameters,
            ),
            "shake": ShakeActionAdapter(
                atomic_controllers["shake"],
                self.gripper_control,
                resolver,
                parameters,
            ),
            "open": OpenActionAdapter(
                atomic_controllers["open"], resolver, parameters
            ),
            "close": CloseActionAdapter(
                atomic_controllers["close"], resolver, parameters
            ),
        }
        verifiers = {
            "pick": PickVerifier(),
            "place": PlaceVerifier(),
            "pour": PourVerifier(),
            "press": PressVerifier(),
            "press_z": PressZVerifier(),
            "shake": ShakeVerifier(),
            "open": OpenVerifier(),
            "close": CloseVerifier(),
        }
        self.runner = SequentialPlanExecutor(
            self.plan,
            adapters,
            verifiers,
            coverage_by_step=validation.step_coverage,
            action_timeouts={
                action_name: definition.max_frames
                for action_name, definition in registry.actions.definitions.items()
            },
            max_retries=1,
        )
        self.trajectory_recorder = _load_trajectory_recorder()(frame_interval=1)
        self._terminal_persisted = False
        self._completed_plan_runs = (
            self.data_collector.episode_count if self.mode == "collect" else 0
        )
        self.state = None
        self._collection_frame = 0
        self._pending_collection_report = None
        self._transactional_collection = bool(
            self.mode == "collect" and hasattr(self.data_collector, "record_step")
        )

    def step(self, state):
        self.state = state
        if self.runner.done and self._terminal_persisted:
            self._last_success = self.runner.success
            self.reset_needed = True
            return None, True, self.runner.success
        if "gripper_position" in state:
            self.trajectory_recorder.record(state["gripper_position"])
        if (
            self.mode == "collect"
            and not self._transactional_collection
            and "camera_data" in state
        ):
            self.data_collector.cache_step(
                camera_images=state["camera_data"],
                joint_angles=state["joint_positions"][:-1],
                language_instruction=self.get_language_instruction(),
            )
        action = self.runner.step(state)
        if not self.runner.done:
            return action, False, False

        if self.mode == "collect" and self._transactional_collection:
            self._prepare_collection_terminal()
        else:
            self._persist_terminal(state)
        return action, True, self.runner.success

    def abort(self, code, message=""):
        self.runner.fail(code, message)
        self._persist_terminal()

    def _prepare_collection_terminal(self):
        self._last_success = self.runner.success
        self.reset_needed = True
        if self._terminal_persisted:
            return
        self._completed_plan_runs += 1
        self._terminal_persisted = True
        self.trajectory_recorder.save(str(self.trajectory_path))
        if hasattr(self.runner.report, "model_dump"):
            self._pending_collection_report = self.runner.report.model_dump(mode="json")
        else:
            self._pending_collection_report = dict(
                getattr(self.runner.report, "payload", {})
            )

    def finalize_collection_episode(self):
        if self.mode != "collect" or self._pending_collection_report is None:
            return
        report = self._pending_collection_report
        self._pending_collection_report = None
        if self.runner.success:
            self.data_collector.commit_episode(report)
        else:
            self.data_collector.fail_episode(report)

    def _persist_terminal(self, state=None):
        self._last_success = self.runner.success
        self.reset_needed = True
        if self._terminal_persisted:
            return
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.runner.report, "model_dump"):
            report = self.runner.report.model_dump(mode="json")
        else:
            report = dict(getattr(self.runner.report, "payload", {}))
        if self.mode != "collect" or not self._transactional_collection:
            self.report_path.write_text(
                self.runner.report.model_dump_json(indent=2),
                encoding="utf-8",
            )
        self._completed_plan_runs += 1
        self._terminal_persisted = True
        self.trajectory_recorder.save(str(self.trajectory_path))
        if self.mode == "collect" and self._transactional_collection:
            self._pending_collection_report = None
            if self.runner.success:
                self.data_collector.commit_episode(report)
            else:
                self.data_collector.fail_episode(report)
        elif self.mode == "collect":
            if self.runner.success and state is not None:
                self.data_collector.write_cached_data(state["joint_positions"][:-1])
            else:
                self.data_collector.clear_cache()

    def start_collection_episode(self, randomization=None):
        if self.mode != "collect" or not self._transactional_collection:
            return None
        self._collection_frame = 0
        self._pending_collection_report = None
        return self.data_collector.start_episode(
            {
                "plan_id": self.plan.plan_id,
                "plan_identity": self.plan_identity,
                "validation_identity": self.validation_identity,
                "randomization": randomization,
            }
        )

    def record_applied_action(self, state, action):
        if self.mode != "collect" or not self._transactional_collection:
            return
        source_fps = float(getattr(self.cfg.collector.video, "source_fps", 60.0))
        self.data_collector.record_step(
            camera_images=state["camera_data"],
            joint_positions=state["joint_positions"],
            action=action,
            timestamp=self._collection_frame / source_fps,
            language_instruction=self.get_language_instruction(),
        )
        self._collection_frame += 1

    def episode_num(self):
        return self._completed_plan_runs

    def reset(self):
        self.runner.reset()
        self.trajectory_recorder.reset()
        super().reset()
        self._terminal_persisted = False
        self._pending_collection_report = None

    def get_language_instruction(self):
        return str(self.plan.metadata.get("language_instruction", ""))
