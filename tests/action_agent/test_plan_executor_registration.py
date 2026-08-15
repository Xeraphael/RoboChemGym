import ast
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from agent.planning.models import (
    ActionStep,
    ActionType,
    AgentPlan,
    CoverageLevel,
    SceneObject,
    ScenePlan,
    UnresolvedCapability,
)
from agent.planning.registry import CapabilityRegistry
from agent.planning.validator import (
    IssueSeverity,
    PlanValidator,
    ValidationIssue,
    ValidationReport,
    plan_fingerprint,
    registry_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = CapabilityRegistry.load_default(ROOT)
FACTORY_PATH = ROOT / "factories" / "controller_factory.py"
CONTROLLER_PATH = ROOT / "controllers" / "plan_executor.py"
ACTION_NAMES = [
    "pick",
    "place",
    "pour",
    "press",
    "press_z",
    "shake",
    "open",
    "close",
]

EXISTING_REGISTRATIONS = [
    ("pickpour", "PickPourTaskController"),
    ("open", "OpenTaskController"),
    ("close", "CloseTaskController"),
    ("openclose", "OpenCloseTaskController"),
    ("pick", "PickTaskController"),
    ("pour", "PourTaskController"),
    ("place", "PlaceTaskController"),
    ("pickplace", "PickPlaceTaskController"),
    ("placepress", "PlacePressTaskController"),
    ("press", "PressTaskController"),
    ("shake", "ShakeTaskController"),
    ("stir", "StirTaskController"),
    ("stirglassrod", "StirGlassrodTaskController"),
    ("shakebeaker", "ShakeBeakerTaskController"),
    ("cleanbeaker", "CleanBeakerTaskController"),
    ("cleanbeaker7policy", "CleanBeaker7PolicyTaskController"),
    ("device_operate", "DeviceOperateController"),
    ("OpenTransportPour", "OpenTransportPourController"),
    ("LiquidMixing", "LiquidMixingController"),
    ("grasp", "GraspObjectTaskController"),
    ("door_pick_pour", "DoorPickPourTaskController"),
    ("benzoic_acid_synthesis_experiment", "BenzoicAcidSynthesisController"),
    ("synthesize_experiment", "SynthesizeController"),
    ("benzoic_acid_dissolution_experiment", "BenzoicAcidDissolutionController"),
    ("beaker_pick", "BeakerPickTaskController"),
    ("group_beaker_scale_experiment", "GroupBeakerScaleController"),
    ("beaker_flask_experiment", "BeakerFlaskExperimentController"),
]


def factory_tree():
    return ast.parse(FACTORY_PATH.read_text(encoding="utf-8"))


def registrations(tree):
    result = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "register_controller"
            and len(node.value.args) >= 2
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[1], ast.Name)
        ):
            continue
        result.append((node.value.args[0].value, node.value.args[1].id))
    return result


def make_plan():
    return AgentPlan(
        plan_id="task11_runtime",
        metadata={"language_instruction": "execute the validated protocol"},
        scene=ScenePlan(
            objects=[
                SceneObject(
                    id="source",
                    asset_id="SourceAsset",
                    instance_name="Source1",
                    role="source",
                ),
                SceneObject(
                    id="target",
                    asset_id="TargetAsset",
                    instance_name="Target1",
                    role="target",
                ),
            ]
        ),
        actions=[
            ActionStep(
                id=f"step_{index:03d}",
                type=ActionType(action_name),
                object="source",
                target="target",
            )
            for index, action_name in enumerate(ACTION_NAMES, start=1)
        ],
    )


def copy_registry_manifests(destination_root):
    destination = destination_root / "agent" / "planning" / "registry"
    destination.parent.mkdir(parents=True)
    shutil.copytree(ROOT / "agent" / "planning" / "registry", destination)


def load_registry_copy(destination_root, mutate_pick=None):
    copy_registry_manifests(destination_root)
    if mutate_pick is not None:
        actions_path = (
            destination_root
            / "agent"
            / "planning"
            / "registry"
            / "actions.yaml"
        )
        manifest = yaml.safe_load(actions_path.read_text(encoding="utf-8"))
        mutate_pick(manifest["actions"]["pick"])
        actions_path.write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return CapabilityRegistry.load_default(destination_root)


def make_validation(plan, *, registry=DEFAULT_REGISTRY, valid=True, coverage=None):
    if coverage is None:
        coverage = {
            step.id: CoverageLevel.SUPPORTED
            for step in plan.actions
        }
    return ValidationReport(
        plan_fingerprint=plan_fingerprint(plan),
        registry_fingerprint=registry_fingerprint(registry),
        valid=valid,
        supported_count=sum(
            level == CoverageLevel.SUPPORTED for level in coverage.values()
        ),
        degraded_count=sum(
            level == CoverageLevel.DEGRADED for level in coverage.values()
        ),
        blocked_count=sum(
            level == CoverageLevel.BLOCKED for level in coverage.values()
        ),
        step_coverage=coverage,
    )


def write_bundle(directory, validation=None, plan=None):
    plan = plan or make_plan()
    validation = validation or make_validation(plan)
    paths = SimpleNamespace(
        plan=directory / "plan.json",
        validation=directory / "validation.json",
        report=directory / "execution-report.json",
        trajectory=directory / "trajectory.json",
    )
    paths.plan.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    paths.validation.write_text(
        validation.model_dump_json(indent=2), encoding="utf-8"
    )
    return plan, paths


def relative_cfg(paths, *, mode="infer", collector=None):
    return SimpleNamespace(
        mode=mode,
        data_collector=collector,
        agent=SimpleNamespace(
            plan_path=str(paths.plan.relative_to(ROOT)),
            validation_report_path=str(paths.validation.relative_to(ROOT)),
            execution_report_path=str(paths.report.relative_to(ROOT)),
            trajectory_path=str(paths.trajectory.relative_to(ROOT)),
        ),
    )


class FakeCollector:
    def __init__(self):
        self.episode_count = 0
        self.cache_calls = []
        self.write_calls = []
        self.clear_calls = 0

    def cache_step(self, **kwargs):
        self.cache_calls.append(kwargs)

    def write_cached_data(self, joint_angles):
        self.write_calls.append(np.asarray(joint_angles).copy())

    def clear_cache(self):
        self.clear_calls += 1


def recording_class(name, records):
    class RecordingClass:
        def __init__(self, *args, **kwargs):
            self.kind = name
            self.args = args
            self.kwargs = kwargs
            records.append(self)

    RecordingClass.__name__ = name
    return RecordingClass


def module_with(**attributes):
    module = ModuleType(attributes.pop("_name"))
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def load_controller_module():
    records = SimpleNamespace(
        base_inits=[],
        registry_roots=[],
        atomic={name: [] for name in ACTION_NAMES},
        adapters={name: [] for name in ACTION_NAMES},
        verifiers={name: [] for name in ACTION_NAMES},
        resolvers=[],
        parameter_resolvers=[],
        runners=[],
        recorders=[],
        validator_registries=[],
        validated_plans=[],
        validation_factory=None,
    )

    class StubBaseController:
        def __init__(self, cfg, robot):
            records.base_inits.append((cfg, robot))
            self.cfg = cfg
            self.robot = robot
            self.mode = cfg.mode
            self.data_collector = cfg.data_collector
            self.rmp_controller = object()
            self.gripper_control = object()
            self.reset_needed = False
            self._last_success = False
            self.base_reset_count = 0

        def reset(self):
            self.base_reset_count += 1
            self.reset_needed = False
            self._last_success = False

    class FakeStateResolver:
        def __init__(self, plan):
            self.plan = plan
            records.resolvers.append(self)

    class FakeParameterResolver:
        def __init__(self, registry, plan):
            self.registry = registry
            self.plan = plan
            records.parameter_resolvers.append(self)

    registry = CapabilityRegistry.load_default(ROOT)
    definitions = registry.actions.definitions

    class FakeCapabilityRegistry:
        @classmethod
        def load_default(cls, root):
            records.registry_roots.append(root)
            return records.registry

    class FakePlanValidator:
        def __init__(self, current_registry):
            records.validator_registries.append(current_registry)

        def validate(self, plan):
            records.validated_plans.append(plan)
            return records.validation_factory(plan)

    class FakeReport:
        def __init__(self, payload):
            self.payload = payload

        def model_dump_json(self, *, indent):
            return json.dumps(self.payload, indent=indent)

    class FakeSequentialPlanExecutor:
        def __init__(self, plan, adapters, verifiers, **kwargs):
            self.plan = plan
            self.adapters = adapters
            self.verifiers = verifiers
            self.kwargs = kwargs
            self.done = False
            self.success = False
            self.report = FakeReport({"execution_success": False})
            self.results = []
            self.step_calls = []
            self.reset_count = 0
            records.runners.append(self)

        def step(self, state):
            self.step_calls.append(state)
            if not self.results:
                return None
            action, done, success, payload = self.results.pop(0)
            self.done = done
            self.success = success
            self.report = FakeReport(payload)
            return action

        def reset(self):
            self.reset_count += 1
            self.done = False
            self.success = False

    class FakeTrajectoryRecorder:
        def __init__(self, frame_interval):
            self.frame_interval = frame_interval
            self.record_calls = []
            self.save_calls = []
            self.reset_count = 0
            records.recorders.append(self)

        def record(self, position):
            self.record_calls.append(np.asarray(position).copy())

        def save(self, path):
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('{"trajectory": true}', encoding="utf-8")
            self.save_calls.append(destination)

        def reset(self):
            self.reset_count += 1
            self.record_calls.clear()

    atomic_class_names = {
        "pick": "PickController",
        "place": "PlaceController",
        "pour": "PourController",
        "press": "PressController",
        "press_z": "PressZController",
        "shake": "ShakeController",
        "open": "OpenController",
        "close": "CloseController",
    }
    atomic_modules = {}
    for action_name, class_name in atomic_class_names.items():
        controller_class = recording_class(class_name, records.atomic[action_name])
        filename = "pressZ_controller" if action_name == "press_z" else f"{action_name}_controller"
        module_name = f"controllers.atomic_actions.{filename}"
        atomic_modules[module_name] = module_with(
            _name=module_name,
            **{class_name: controller_class},
        )

    adapter_class_names = {
        "pick": "PickActionAdapter",
        "place": "PlaceActionAdapter",
        "pour": "PourActionAdapter",
        "press": "PressActionAdapter",
        "press_z": "PressZActionAdapter",
        "shake": "ShakeActionAdapter",
        "open": "OpenActionAdapter",
        "close": "CloseActionAdapter",
    }
    adapter_attributes = {"StateResolver": FakeStateResolver}
    for action_name, class_name in adapter_class_names.items():
        adapter_attributes[class_name] = recording_class(
            class_name, records.adapters[action_name]
        )

    verifier_class_names = {
        "pick": "PickVerifier",
        "place": "PlaceVerifier",
        "pour": "PourVerifier",
        "press": "PressVerifier",
        "press_z": "PressZVerifier",
        "shake": "ShakeVerifier",
        "open": "OpenVerifier",
        "close": "CloseVerifier",
    }
    verifier_attributes = {}
    for action_name, class_name in verifier_class_names.items():
        verifier_attributes[class_name] = recording_class(
            class_name, records.verifiers[action_name]
        )

    atomic_package = ModuleType("controllers.atomic_actions")
    atomic_package.__path__ = []
    plan_execution_package = ModuleType("agent.action.plan_execution")
    plan_execution_package.__path__ = []
    rating_package = ModuleType("agent.action.rating")
    rating_package.__path__ = []
    stubs = {
        "controllers.base_controller": module_with(
            _name="controllers.base_controller", BaseController=StubBaseController
        ),
        "controllers.atomic_actions": atomic_package,
        **atomic_modules,
        "controllers.plan_action_adapters": module_with(
            _name="controllers.plan_action_adapters", **adapter_attributes
        ),
        "agent.action.plan_execution": plan_execution_package,
        "agent.action.plan_execution.executor": module_with(
            _name="agent.action.plan_execution.executor",
            SequentialPlanExecutor=FakeSequentialPlanExecutor,
        ),
        "agent.action.plan_execution.parameter_resolver": module_with(
            _name="agent.action.plan_execution.parameter_resolver",
            ParameterResolver=FakeParameterResolver,
        ),
        "agent.action.plan_execution.verifiers": module_with(
            _name="agent.action.plan_execution.verifiers", **verifier_attributes
        ),
        "agent.action.rating": rating_package,
        "agent.action.rating.trajectory_recorder": module_with(
            _name="agent.action.rating.trajectory_recorder",
            TrajectoryRecorder=FakeTrajectoryRecorder,
        ),
        "agent.planning.registry": module_with(
            _name="agent.planning.registry", CapabilityRegistry=FakeCapabilityRegistry
        ),
    }
    module_name = "controllers._task11_plan_executor"
    with patch.dict(sys.modules, stubs, clear=False):
        spec = importlib.util.spec_from_file_location(module_name, CONTROLLER_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    module._load_trajectory_recorder = lambda: FakeTrajectoryRecorder
    module.PlanValidator = FakePlanValidator
    records.registry = registry
    records.validation_factory = lambda plan: make_validation(
        plan,
        registry=records.registry,
    )
    records.definition_timeouts = {
        name: definition.max_frames
        for name, definition in definitions.items()
    }
    return module, records


class PlanExecutorRegistrationTests(unittest.TestCase):
    def test_factory_imports_plan_executor_exactly_once(self):
        imports = [
            node
            for node in factory_tree().body
            if isinstance(node, ast.ImportFrom)
            and node.module == "controllers.plan_executor"
            and [(alias.name, alias.asname) for alias in node.names]
            == [("PlanExecutorController", None)]
        ]

        self.assertEqual(len(imports), 1)

    def test_factory_registers_plan_executor_exactly_once(self):
        actual = registrations(factory_tree())

        self.assertEqual(actual.count(("plan_executor", "PlanExecutorController")), 1)

    def test_existing_factory_registrations_are_preserved_once_and_in_order(self):
        actual = registrations(factory_tree())

        self.assertEqual(
            actual,
            [
                *EXISTING_REGISTRATIONS,
                ("plan_executor", "PlanExecutorController"),
                ("policy", "PolicyController"),
            ],
        )


class PlanExecutorControllerRuntimeTests(unittest.TestCase):
    def require_controller_module(self):
        self.assertTrue(
            CONTROLLER_PATH.is_file(),
            "controllers/plan_executor.py must implement the Isaac-facing wrapper",
        )
        return load_controller_module()

    def test_valid_plan_wires_all_actions_and_resolves_relative_paths(self):
        module, records = self.require_controller_module()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            plan, paths = write_bundle(directory)
            cfg = relative_cfg(paths)
            robot = SimpleNamespace(gripper=object())

            controller = module.PlanExecutorController(cfg, robot)

            self.assertEqual(records.base_inits, [(cfg, robot)])
            self.assertEqual(records.registry_roots, [ROOT])
            self.assertEqual(records.validator_registries, [records.registry])
            self.assertEqual(records.validated_plans, [plan])
            self.assertEqual(controller.plan, plan)
            self.assertEqual(controller.get_language_instruction(), "execute the validated protocol")
            self.assertEqual(controller.report_path, paths.report)
            self.assertEqual(controller.trajectory_path, paths.trajectory)
            self.assertEqual(controller.trajectory_recorder.frame_interval, 1)

            self.assertEqual(len(records.runners), 1)
            runner = records.runners[0]
            self.assertEqual(list(runner.adapters), ACTION_NAMES)
            self.assertEqual(list(runner.verifiers), ACTION_NAMES)
            self.assertEqual(runner.kwargs["action_timeouts"], records.definition_timeouts)
            self.assertEqual(runner.kwargs["max_retries"], 1)
            self.assertEqual(
                runner.kwargs["coverage_by_step"],
                make_validation(plan).step_coverage,
            )

            self.assertEqual(len(records.resolvers), 1)
            self.assertEqual(len(records.parameter_resolvers), 1)
            resolver = records.resolvers[0]
            parameters = records.parameter_resolvers[0]
            self.assertEqual(resolver.plan, plan)
            self.assertIs(parameters.registry, records.registry)
            self.assertEqual(parameters.plan, plan)

            expected_atomic_kwargs = {
                "pick": {
                    "name": "plan_pick",
                    "cspace_controller": controller.rmp_controller,
                },
                "place": {
                    "name": "plan_place",
                    "cspace_controller": controller.rmp_controller,
                    "gripper": robot.gripper,
                    "_position_threshold": 0.06,
                },
                "pour": {
                    "name": "plan_pour",
                    "cspace_controller": controller.rmp_controller,
                },
                "press": {
                    "name": "plan_press",
                    "cspace_controller": controller.rmp_controller,
                },
                "press_z": {
                    "name": "plan_press_z",
                    "cspace_controller": controller.rmp_controller,
                },
                "shake": {
                    "name": "plan_shake",
                    "cspace_controller": controller.rmp_controller,
                },
                "open": {
                    "name": "plan_open",
                    "cspace_controller": controller.rmp_controller,
                    "gripper": robot.gripper,
                },
                "close": {
                    "name": "plan_close",
                    "cspace_controller": controller.rmp_controller,
                    "gripper": robot.gripper,
                },
            }
            atomic = {}
            for action_name in ACTION_NAMES:
                self.assertEqual(len(records.atomic[action_name]), 1)
                atomic[action_name] = records.atomic[action_name][0]
                self.assertEqual(atomic[action_name].args, ())
                self.assertEqual(
                    atomic[action_name].kwargs,
                    expected_atomic_kwargs[action_name],
                )
                self.assertEqual(len(records.adapters[action_name]), 1)
                self.assertEqual(len(records.verifiers[action_name]), 1)

            expected_adapter_args = {
                "pick": (
                    atomic["pick"], controller.gripper_control, resolver, parameters
                ),
                "place": (
                    atomic["place"], controller.gripper_control, resolver, parameters
                ),
                "pour": (
                    atomic["pour"], robot, controller.gripper_control, resolver, parameters
                ),
                "press": (
                    atomic["press"], controller.gripper_control, resolver, parameters
                ),
                "press_z": (
                    atomic["press_z"], controller.gripper_control, resolver, parameters
                ),
                "shake": (
                    atomic["shake"], controller.gripper_control, resolver, parameters
                ),
                "open": (atomic["open"], resolver, parameters),
                "close": (atomic["close"], resolver, parameters),
            }
            for action_name in ACTION_NAMES:
                adapter = records.adapters[action_name][0]
                self.assertEqual(adapter.args, expected_adapter_args[action_name])
                self.assertEqual(adapter.kwargs, {})
                self.assertIs(runner.adapters[action_name], adapter)
                self.assertIs(
                    runner.verifiers[action_name], records.verifiers[action_name][0]
                )

    def test_forged_supported_report_cannot_bypass_current_sequence_validation(self):
        module, records = self.require_controller_module()
        plan = AgentPlan(
            plan_id="invalid_runtime",
            scene=ScenePlan(
                objects=[
                    SceneObject(
                        id="source",
                        asset_id="ErlenmeyerFlask",
                        instance_name="Source1",
                        role="source",
                    ),
                    SceneObject(
                        id="target",
                        asset_id="ErlenmeyerFlask",
                        instance_name="Target1",
                        role="target",
                    ),
                ]
            ),
            actions=[
                ActionStep(
                    id="step_001",
                    type=ActionType.POUR,
                    object="source",
                    target="target",
                )
            ],
        )
        current = PlanValidator(CapabilityRegistry.load_default(ROOT)).validate(plan)
        self.assertFalse(current.valid)
        self.assertIn("OBJECT_NOT_HELD", {issue.code for issue in current.issues})

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            _, paths = write_bundle(
                directory,
                validation=make_validation(plan),
                plan=plan,
            )
            records.validation_factory = lambda validated_plan: current

            with self.assertRaisesRegex(
                ValueError,
                "current plan validation failed.*recompile artifacts",
            ):
                module.PlanExecutorController(
                    relative_cfg(paths), SimpleNamespace(gripper=object())
                )

            self.assertEqual(records.validated_plans, [plan])
            self.assertEqual(records.runners, [])

    def test_current_validator_or_registry_outcome_drift_requires_recompile(self):
        module, records = self.require_controller_module()
        plan = make_plan().model_copy(deep=True)
        plan.actions[0].modifiers = {"carefully": True}
        stored = make_validation(plan).model_copy(update={"degraded_count": 1})
        current_coverage = stored.step_coverage.copy()
        current_coverage["step_001"] = CoverageLevel.DEGRADED
        current = make_validation(plan, coverage=current_coverage)

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            _, paths = write_bundle(directory, validation=stored, plan=plan)
            records.validation_factory = lambda validated_plan: current

            with self.assertRaisesRegex(
                ValueError,
                "current validation.*recompile artifacts",
            ):
                module.PlanExecutorController(
                    relative_cfg(paths), SimpleNamespace(gripper=object())
                )

            self.assertEqual(records.validated_plans, [plan])
            self.assertEqual(records.runners, [])

    def test_registry_manifest_drift_requires_recompile_before_live_validation(self):
        module, records = self.require_controller_module()
        plan = make_plan()

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            bundle_directory = directory / "bundle"
            bundle_directory.mkdir()
            base_registry = load_registry_copy(directory / "base_checkout")
            stored_validation = make_validation(plan, registry=base_registry)
            _, paths = write_bundle(
                bundle_directory,
                validation=stored_validation,
                plan=plan,
            )

            mutations = {
                "max_frames": lambda pick: pick.__setitem__(
                    "max_frames", pick["max_frames"] + 1
                ),
                "default_pre_offset_z": lambda pick: pick[
                    "default_parameters"
                ].__setitem__("pre_offset_z", 0.13),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    current_registry = load_registry_copy(
                        directory / f"current_{label}",
                        mutate,
                    )
                    self.assertNotEqual(
                        registry_fingerprint(current_registry),
                        stored_validation.registry_fingerprint,
                    )
                    records.registry = current_registry
                    records.validation_factory = lambda validated_plan: make_validation(
                        validated_plan,
                        registry=current_registry,
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        "registry fingerprint.*recompile artifacts",
                    ):
                        module.PlanExecutorController(
                            relative_cfg(paths),
                            SimpleNamespace(gripper=object()),
                        )

            self.assertEqual(records.validated_plans, [])
            self.assertEqual(records.registry_roots, [ROOT, ROOT])
            self.assertEqual(records.runners, [])

    def test_invalid_or_incomplete_validation_fails_before_runner_construction(self):
        module, records = self.require_controller_module()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            plan, paths = write_bundle(directory)
            cfg = relative_cfg(paths)
            robot = SimpleNamespace(gripper=object())
            supported = {
                step.id: CoverageLevel.SUPPORTED
                for step in plan.actions
            }
            cases = [
                (
                    "invalid report",
                    make_validation(plan, valid=False, coverage=supported),
                    "validation report is not valid",
                ),
                (
                    "missing coverage",
                    make_validation(
                        plan,
                        coverage={
                            key: value
                            for key, value in supported.items()
                            if key != "step_008"
                        },
                    ),
                    "missing validation coverage.*step_008",
                ),
                (
                    "blocked coverage",
                    make_validation(
                        plan,
                        coverage={
                            **supported,
                            "step_004": CoverageLevel.BLOCKED,
                        },
                    ),
                    "blocked validation coverage.*step_004",
                ),
            ]

            for label, validation, message in cases:
                with self.subTest(label=label):
                    paths.validation.write_text(
                        validation.model_dump_json(indent=2), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        module.PlanExecutorController(cfg, robot)

            self.assertEqual(records.registry_roots, [ROOT] * len(cases))
            self.assertEqual(records.runners, [])
            self.assertTrue(all(not values for values in records.atomic.values()))

    def test_stale_validation_rejects_semantic_plan_mutations(self):
        module, records = self.require_controller_module()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            plan, paths = write_bundle(directory)
            cfg = relative_cfg(paths)
            robot = SimpleNamespace(gripper=object())
            mutations = []
            action_changed = plan.model_copy(deep=True)
            action_changed.actions[0].type = ActionType.SHAKE
            mutations.append(action_changed)
            parameter_changed = plan.model_copy(deep=True)
            parameter_changed.actions[0].parameters["shake_distance"] = 0.2
            mutations.append(parameter_changed)
            order_changed = plan.model_copy(deep=True)
            order_changed.actions = list(reversed(order_changed.actions))
            mutations.append(order_changed)
            scene_changed = plan.model_copy(deep=True)
            scene_changed.scene.objects[0].instance_name = "ChangedSource"
            mutations.append(scene_changed)

            for mutated in mutations:
                with self.subTest(plan=mutated):
                    paths.plan.write_text(
                        mutated.model_dump_json(indent=2), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "fingerprint"):
                        module.PlanExecutorController(cfg, robot)

            self.assertEqual(records.registry_roots, [ROOT] * len(mutations))
            self.assertEqual(records.runners, [])

    def test_missing_fingerprint_and_contradictory_reports_fail_closed(self):
        module, records = self.require_controller_module()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            plan, paths = write_bundle(directory)
            cfg = relative_cfg(paths)
            robot = SimpleNamespace(gripper=object())
            coverage = {
                step.id: CoverageLevel.SUPPORTED
                for step in plan.actions
            }
            blocked_issue = ValidationIssue(
                code="FORGED_BLOCK",
                severity=IssueSeverity.ERROR,
                level=CoverageLevel.BLOCKED,
                message="forged blocked issue",
            )
            base = make_validation(plan, coverage=coverage)
            cases = [
                (
                    "blocked issue",
                    base.model_copy(update={"issues": [blocked_issue]}),
                    "blocked validation issue",
                ),
                (
                    "blocked count",
                    base.model_copy(update={"blocked_count": 1}),
                    "blocked_count",
                ),
                (
                    "negative blocked count",
                    base.model_copy(update={"blocked_count": -1}),
                    "blocked_count",
                ),
                (
                    "coverage count",
                    base.model_copy(update={"supported_count": 0}),
                    "coverage counts",
                ),
                (
                    "degraded count",
                    base.model_copy(update={"degraded_count": 1}),
                    "coverage counts",
                ),
                (
                    "extra coverage",
                    base.model_copy(
                        update={
                            "step_coverage": {
                                **coverage,
                                "step_999": CoverageLevel.SUPPORTED,
                            },
                            "supported_count": len(coverage) + 1,
                        }
                    ),
                    "coverage steps",
                ),
            ]

            for missing_fingerprint in (
                "plan_fingerprint",
                "registry_fingerprint",
            ):
                legacy_payload = base.model_dump(mode="json")
                legacy_payload.pop(missing_fingerprint)
                paths.validation.write_text(
                    json.dumps(legacy_payload), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, missing_fingerprint):
                    module.PlanExecutorController(cfg, robot)

            for label, validation, message in cases:
                with self.subTest(label=label):
                    paths.validation.write_text(
                        validation.model_dump_json(indent=2), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        module.PlanExecutorController(cfg, robot)

            unresolved = plan.model_copy(deep=True)
            unresolved.unresolved_capabilities = [
                UnresolvedCapability(
                    source_text="aspirate liquid",
                    missing_action="aspirate",
                    reason="not executable",
                )
            ]
            paths.plan.write_text(
                unresolved.model_dump_json(indent=2), encoding="utf-8"
            )
            paths.validation.write_text(
                make_validation(unresolved).model_dump_json(indent=2),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unresolved capabilities"):
                module.PlanExecutorController(cfg, robot)

            self.assertEqual(records.registry_roots, [ROOT] * (len(cases) + 3))
            self.assertEqual(records.runners, [])

    def test_step_preserves_final_action_and_persists_success_once(self):
        module, records = self.require_controller_module()
        collector = FakeCollector()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            _, paths = write_bundle(directory)
            cfg = relative_cfg(paths, mode="collect", collector=collector)
            controller = module.PlanExecutorController(
                cfg, SimpleNamespace(gripper=object())
            )
            runner = records.runners[0]
            runner.results = [
                ("move-action", False, False, {"execution_success": False}),
                (
                    "final-action",
                    True,
                    True,
                    {"execution_success": True, "steps": ["step_008"]},
                ),
            ]
            state = {
                "joint_positions": np.arange(9, dtype=float),
                "gripper_position": np.array([0.1, 0.2, 0.3]),
                "camera_data": {"front_rgb": "frame"},
            }

            self.assertEqual(controller.step(state), ("move-action", False, False))
            self.assertFalse(controller.reset_needed)
            self.assertEqual(controller.step(state), ("final-action", True, True))

            self.assertIs(controller.state, state)
            self.assertTrue(controller.reset_needed)
            self.assertTrue(controller._last_success)
            self.assertEqual(
                json.loads(paths.report.read_text(encoding="utf-8")),
                {"execution_success": True, "steps": ["step_008"]},
            )
            self.assertEqual(
                records.recorders[0].save_calls,
                [paths.trajectory],
            )
            self.assertTrue(paths.trajectory.is_file())
            self.assertEqual(len(collector.cache_calls), 2)
            np.testing.assert_array_equal(
                collector.cache_calls[0]["joint_angles"], np.arange(8, dtype=float)
            )
            self.assertEqual(
                collector.cache_calls[0]["language_instruction"],
                "execute the validated protocol",
            )
            self.assertEqual(collector.cache_calls[0]["camera_images"], {"front_rgb": "frame"})
            self.assertEqual(len(collector.write_calls), 1)
            np.testing.assert_array_equal(
                collector.write_calls[0], np.arange(8, dtype=float)
            )
            self.assertEqual(collector.clear_calls, 0)

            paths.report.write_text("terminal report must not be rewritten", encoding="utf-8")
            self.assertEqual(controller.step(state), (None, True, True))
            self.assertEqual(
                paths.report.read_text(encoding="utf-8"),
                "terminal report must not be rewritten",
            )
            self.assertEqual(records.recorders[0].save_calls, [paths.trajectory])
            self.assertEqual(len(collector.write_calls), 1)
            self.assertEqual(len(records.recorders[0].record_calls), 2)
            self.assertEqual(len(collector.cache_calls), 2)
            self.assertEqual(len(runner.step_calls), 2)

    def test_execute_mode_persists_reports_without_collecting_dataset(self):
        module, records = self.require_controller_module()
        collector = FakeCollector()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            _, paths = write_bundle(directory)
            controller = module.PlanExecutorController(
                relative_cfg(paths, mode="execute", collector=collector),
                SimpleNamespace(gripper=object()),
            )
            records.runners[0].results = [
                (
                    "final-action",
                    True,
                    True,
                    {"execution_success": True},
                )
            ]
            state = {
                "joint_positions": np.arange(9, dtype=float),
                "gripper_position": np.array([0.1, 0.2, 0.3]),
                "camera_data": {"front_rgb": "frame"},
            }

            self.assertEqual(
                controller.step(state), ("final-action", True, True)
            )
            self.assertEqual(collector.cache_calls, [])
            self.assertEqual(collector.write_calls, [])
            self.assertTrue(paths.report.is_file())
            self.assertTrue(paths.trajectory.is_file())

    def test_failure_clears_cache_and_reset_rearms_terminal_persistence(self):
        module, records = self.require_controller_module()
        collector = FakeCollector()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            _, paths = write_bundle(directory)
            cfg = relative_cfg(paths, mode="collect", collector=collector)
            controller = module.PlanExecutorController(
                cfg, SimpleNamespace(gripper=object())
            )
            runner = records.runners[0]
            recorder = records.recorders[0]
            runner.results = [
                (
                    "failed-action",
                    True,
                    False,
                    {"execution_success": False, "failed_step": "step_001"},
                )
            ]
            state = {
                "joint_positions": np.arange(9, dtype=float),
                "gripper_position": np.array([0.4, 0.5, 0.6]),
                "camera_data": {"front_rgb": "frame"},
            }

            self.assertEqual(controller.step(state), ("failed-action", True, False))
            self.assertEqual(collector.clear_calls, 1)
            self.assertEqual(collector.write_calls, [])
            self.assertFalse(controller._last_success)
            self.assertTrue(controller.reset_needed)

            controller.reset()

            self.assertEqual(controller.base_reset_count, 1)
            self.assertEqual(runner.reset_count, 1)
            self.assertEqual(recorder.reset_count, 1)
            self.assertFalse(controller.reset_needed)
            self.assertFalse(controller._last_success)
            self.assertEqual(controller.get_language_instruction(), "execute the validated protocol")

            runner.results = [
                (
                    "second-final-action",
                    True,
                    True,
                    {"execution_success": True, "steps": ["step_008"]},
                )
            ]
            self.assertEqual(
                controller.step(state), ("second-final-action", True, True)
            )
            self.assertEqual(recorder.save_calls, [paths.trajectory, paths.trajectory])
            self.assertEqual(len(collector.write_calls), 1)

    def test_failed_collect_terminals_count_completed_plan_runs_once(self):
        module, records = self.require_controller_module()
        collector = FakeCollector()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            directory = Path(temporary_directory)
            _, paths = write_bundle(directory)
            controller = module.PlanExecutorController(
                relative_cfg(paths, mode="collect", collector=collector),
                SimpleNamespace(gripper=object()),
            )
            runner = records.runners[0]
            state = {
                "joint_positions": np.arange(9, dtype=float),
                "camera_data": {},
            }
            runner.results = [
                (
                    "failed-action",
                    True,
                    False,
                    {"execution_success": False, "failed_step": "step_001"},
                )
            ]

            self.assertEqual(controller.episode_num(), 0)
            self.assertEqual(controller.step(state), ("failed-action", True, False))
            self.assertEqual(controller.episode_num(), 1)
            self.assertEqual(collector.episode_count, 0)

            self.assertEqual(controller.step(state), (None, True, False))
            self.assertEqual(controller.episode_num(), 1)

            controller.reset()
            self.assertEqual(controller.episode_num(), 1)
            runner.results = [
                (
                    "second-failed-action",
                    True,
                    False,
                    {"execution_success": False, "failed_step": "step_001"},
                )
            ]
            self.assertEqual(
                controller.step(state),
                ("second-failed-action", True, False),
            )
            self.assertEqual(controller.episode_num(), 2)
            self.assertEqual(collector.episode_count, 0)


if __name__ == "__main__":
    unittest.main()
