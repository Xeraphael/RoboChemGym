import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from agent.action.plan_execution.executor import SequentialPlanExecutor
from agent.action.plan_execution.models import VerificationResult
from agent.planning.models import AgentPlan, CoverageLevel
from agent.planning.registry import CapabilityRegistry
from agent.planning.validator import PlanValidator


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/action_agent/fixtures/protocol1_plan.json"
EXPECTED_ACTIONS = [
    ("step_001", "pick", "solid_flask", None),
    ("step_002", "place", "solid_flask", "heating_plate"),
    ("step_003", "press", None, "heating_plate"),
    ("step_004", "pick", "liquid_flask", None),
    ("step_005", "pour", "liquid_flask", "solid_flask"),
    ("step_006", "place", "liquid_flask", "return_platform"),
]
EXPECTED_COVERAGE = {
    "step_001": CoverageLevel.SUPPORTED,
    "step_002": CoverageLevel.SUPPORTED,
    "step_003": CoverageLevel.DEGRADED,
    "step_004": CoverageLevel.SUPPORTED,
    "step_005": CoverageLevel.DEGRADED,
    "step_006": CoverageLevel.SUPPORTED,
}
CONTROLLER_IMPORTS = {
    "controllers.base_controller": "BaseController",
    "controllers.open_controller": "OpenTaskController",
    "controllers.pickpour_controller": "PickPourTaskController",
    "controllers.placepress_controller": "PlacePressTaskController",
    "controllers.pick_controller": "PickTaskController",
    "controllers.pour_controller": "PourTaskController",
    "controllers.place_controller": "PlaceTaskController",
    "controllers.press_controller": "PressTaskController",
    "controllers.shake_controller": "ShakeTaskController",
    "controllers.stir_controller": "StirTaskController",
    "controllers.stirglassrod_controller": "StirGlassrodTaskController",
    "controllers.pickplace_controller": "PickPlaceTaskController",
    "controllers.shakebeaker_controller": "ShakeBeakerTaskController",
    "controllers.cleanbeaker_controller": "CleanBeakerTaskController",
    "controllers.cleanbeaker7policy_controller": "CleanBeaker7PolicyTaskController",
    "controllers.device_operate_controller": "DeviceOperateController",
    "controllers.opentransportpour_controller": "OpenTransportPourController",
    "controllers.LiquidMixing_controller": "LiquidMixingController",
    "controllers.close_controller": "CloseTaskController",
    "controllers.openclose_controller": "OpenCloseTaskController",
    "controllers.grasp_controller": "GraspObjectTaskController",
    "controllers.door_pick_pour_controller": "DoorPickPourTaskController",
    "controllers.benzoic_acid_synthesis_controller": (
        "BenzoicAcidSynthesisController"
    ),
    "controllers.synthesize_controller": "SynthesizeController",
    "controllers.benzoic_acid_dissolution_controller": (
        "BenzoicAcidDissolutionController"
    ),
    "controllers.beaker_pick_controller": "BeakerPickTaskController",
    "controllers.group_beaker_scale_controller": "GroupBeakerScaleController",
    "controllers.beaker_flask_experiment_controller": (
        "BeakerFlaskExperimentController"
    ),
    "controllers.policy_controller": "PolicyController",
}


def load_fixture():
    return AgentPlan.model_validate_json(FIXTURE_PATH.read_text(encoding="utf-8"))


class RecordingAdapter:
    def __init__(self, action_type, prepared_steps):
        self.action_type = action_type
        self.prepared_steps = prepared_steps
        self.current_action = None

    def prepare(self, step, context):
        del context
        self.current_action = (
            step.id,
            self.action_type,
            step.object,
            step.target,
        )
        self.prepared_steps.append((step.id, self.action_type))

    def step(self, state):
        del state
        return self.current_action

    def is_done(self):
        return self.current_action is not None

    def reset(self):
        self.current_action = None


class RecordingVerifier:
    def __init__(self, action_type, verified_steps):
        self.action_type = action_type
        self.verified_steps = verified_steps

    def verify(self, request):
        self.verified_steps.append((request.step.id, self.action_type))
        return VerificationResult(
            success=True,
            code=f"MOCK_{self.action_type.upper()}_OK",
            verification_level="mock_state",
        )


def run_fixture_through_executor(plan, coverage_by_step):
    action_types = list(dict.fromkeys(step.type.value for step in plan.actions))
    prepared_steps = []
    verified_steps = []
    adapters = {
        action_type: RecordingAdapter(action_type, prepared_steps)
        for action_type in action_types
    }
    verifiers = {
        action_type: RecordingVerifier(action_type, verified_steps)
        for action_type in action_types
    }
    runner = SequentialPlanExecutor(
        plan,
        adapters,
        verifiers,
        coverage_by_step=coverage_by_step,
        max_retries=0,
    )

    actions = [
        runner.step({"frame": frame})
        for frame in range(1, len(plan.actions) + 1)
    ]
    if not runner.done:
        raise AssertionError(
            "canonical fixture did not finish within one frame per action"
        )

    records = runner.report.steps
    return SimpleNamespace(
        actions=actions,
        records=[
            (record.step_id, record.action, record.object_id, record.target_id)
            for record in records
        ],
        prepared_steps=prepared_steps,
        verified_steps=verified_steps,
        execution_success=runner.report.execution_success,
        controller_completed=[record.controller_completed for record in records],
        step_successes=[record.success for record in records],
        adapters=[record.adapter for record in records],
        verifiers=[record.verifier for record in records],
    )


def _stub_controller_module(module_name, class_name, controller_class=None):
    module = ModuleType(module_name)
    setattr(
        module,
        class_name,
        controller_class or type(class_name, (), {}),
    )
    return module


def load_isolated_controller_factory():
    class StubPlanExecutorController:
        def __init__(self, cfg, robot):
            self.cfg = cfg
            self.robot = robot

    stubs = {
        module_name: _stub_controller_module(module_name, class_name)
        for module_name, class_name in CONTROLLER_IMPORTS.items()
    }
    stubs["controllers.plan_executor"] = _stub_controller_module(
        "controllers.plan_executor",
        "PlanExecutorController",
        StubPlanExecutorController,
    )
    module_name = "factories._protocol1_controller_factory"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "factories/controller_factory.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("controller factory module could not be loaded")
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, stubs, clear=False):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)

    return module, StubPlanExecutorController


class Protocol1PipelineTests(unittest.TestCase):
    def test_fixture_has_exact_executable_degradation_contract(self):
        plan = load_fixture()
        report = PlanValidator(CapabilityRegistry.load_default(ROOT)).validate(plan)

        self.assertTrue(report.valid)
        self.assertEqual(report.blocked_count, 0)
        self.assertEqual(report.degraded_count, 5)
        self.assertEqual(report.supported_count, 4)
        self.assertEqual(report.step_coverage, EXPECTED_COVERAGE)
        self.assertEqual(
            [
                (
                    issue.code,
                    issue.severity.value,
                    issue.level.value,
                    issue.step_id,
                )
                for issue in report.issues
            ],
            [("SEMANTIC_DEGRADATION", "warning", "degraded", None)] * 3,
        )
        self.assertEqual(len(plan.scene.objects), 4)
        self.assertEqual(
            [
                (step.id, step.type.value, step.object, step.target)
                for step in plan.actions
            ],
            EXPECTED_ACTIONS,
        )

    def test_fixture_executes_all_six_actions_through_real_plan_executor(self):
        plan = load_fixture()
        validation = PlanValidator(
            CapabilityRegistry.load_default(ROOT)
        ).validate(plan)

        result = run_fixture_through_executor(plan, validation.step_coverage)

        self.assertEqual(result.actions, EXPECTED_ACTIONS)
        self.assertEqual(result.records, EXPECTED_ACTIONS)
        expected_routing = [(item[0], item[1]) for item in EXPECTED_ACTIONS]
        self.assertEqual(result.prepared_steps, expected_routing)
        self.assertEqual(result.verified_steps, expected_routing)
        self.assertTrue(result.execution_success)
        self.assertTrue(all(result.controller_completed))
        self.assertTrue(all(result.step_successes))
        self.assertEqual(result.adapters, ["RecordingAdapter"] * 6)
        self.assertEqual(result.verifiers, ["RecordingVerifier"] * 6)

    def test_default_factory_constructs_generic_plan_executor_backend(self):
        factory, controller_class = load_isolated_controller_factory()
        cfg = object()
        robot = object()

        with patch.dict(os.environ, {"AGENT_MONITOR_MODE": "false"}):
            controller = factory.create_controller(
                "plan_executor",
                cfg=cfg,
                robot=robot,
            )

        self.assertIs(factory._controller_registry["plan_executor"], controller_class)
        self.assertIsInstance(controller, controller_class)
        self.assertIs(controller.cfg, cfg)
        self.assertIs(controller.robot, robot)


if __name__ == "__main__":
    unittest.main()
