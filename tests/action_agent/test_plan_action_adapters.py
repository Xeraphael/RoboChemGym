import ast
import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from agent.planning.models import (
    ActionStep,
    ActionType,
    AgentPlan,
    SceneObject,
    ScenePlan,
)
from agent.planning.registry import (
    AssetDefinition,
    AssetRegistry,
    ActionRegistry,
    CapabilityRegistry,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIZE = np.array([0.04, 0.04, 0.08])


def rotation_from_wxyz(quaternion):
    quaternion = np.asarray(quaternion)
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]])


def wxyz_from_rotation(rotation):
    quaternion = rotation.as_quat()
    return quaternion[[3, 0, 1, 2]]


def clear_orientation_cache(module):
    cached_loader = getattr(module, "_load_orientation_profiles_cached", None)
    if cached_loader is not None:
        cached_loader.cache_clear()
    public_clear = getattr(module.load_orientation_profiles, "cache_clear", None)
    if public_clear is not None:
        public_clear()


@contextmanager
def temporary_orientation_yaml(content):
    module = adapter_module()
    original_path = module._ORIENTATION_PROFILE_PATH
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "orientation_profiles.yaml"
        path.write_text(content, encoding="utf-8")
        module._ORIENTATION_PROFILE_PATH = path
        clear_orientation_cache(module)
        try:
            yield module
        finally:
            module._ORIENTATION_PROFILE_PATH = original_path
            clear_orientation_cache(module)


@contextmanager
def loaded_atomic_controller_module(filename, *, stage_units=1.0):
    class StubBaseController:
        def __init__(self, name=None):
            self.name = name
            self.base_reset_count = 0

        def reset(self):
            self.base_reset_count += 1

    class StubArticulationAction:
        def __init__(self, joint_positions=None, joint_velocities=None):
            self.joint_positions = joint_positions
            self.joint_velocities = joint_velocities

    controllers_module = ModuleType("omni.isaac.core.controllers")
    controllers_module.BaseController = StubBaseController
    articulation_controller_module = ModuleType(
        "omni.isaac.core.controllers.articulation_controller"
    )
    articulation_controller_module.ArticulationController = object
    stage_module = ModuleType("omni.isaac.core.utils.stage")
    stage_module.get_stage_units = lambda: stage_units
    stage_module.get_current_stage = lambda: None
    types_module = ModuleType("omni.isaac.core.utils.types")
    types_module.ArticulationAction = StubArticulationAction
    rotations_module = ModuleType("omni.isaac.core.utils.rotations")
    rotations_module.euler_angles_to_quat = lambda *args, **kwargs: np.array(
        [1.0, 0.0, 0.0, 0.0]
    )
    gripper_module = ModuleType("controllers.robot_controllers.grapper_manager")
    gripper_module.Gripper = object
    isaac_gripper_module = ModuleType(
        "omni.isaac.manipulators.grippers.gripper"
    )
    isaac_gripper_module.Gripper = object
    object_utils_module = ModuleType("utils.object_utils")
    object_utils_module.ObjectUtils = object

    stubs = {
        "omni": ModuleType("omni"),
        "omni.isaac": ModuleType("omni.isaac"),
        "omni.isaac.core": ModuleType("omni.isaac.core"),
        "omni.isaac.core.controllers": controllers_module,
        "omni.isaac.core.controllers.articulation_controller": (
            articulation_controller_module
        ),
        "omni.isaac.core.utils": ModuleType("omni.isaac.core.utils"),
        "omni.isaac.core.utils.stage": stage_module,
        "omni.isaac.core.utils.types": types_module,
        "omni.isaac.core.utils.rotations": rotations_module,
        "omni.isaac.manipulators": ModuleType("omni.isaac.manipulators"),
        "omni.isaac.manipulators.grippers": ModuleType(
            "omni.isaac.manipulators.grippers"
        ),
        "omni.isaac.manipulators.grippers.gripper": isaac_gripper_module,
        "controllers.robot_controllers": ModuleType(
            "controllers.robot_controllers"
        ),
        "controllers.robot_controllers.grapper_manager": gripper_module,
        "utils": ModuleType("utils"),
        "utils.object_utils": object_utils_module,
    }
    units_suffix = str(stage_units).replace(".", "_")
    module_name = f"_task10_{Path(filename).stem}_{units_suffix}"
    path = ROOT / "controllers" / "atomic_actions" / filename
    with patch.dict(sys.modules, stubs, clear=False):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        yield module, StubArticulationAction


@lru_cache(maxsize=1)
def adapter_module():
    return importlib.import_module("controllers.plan_action_adapters")


def parameter_resolver(registry, plan):
    module = importlib.import_module("agent.action.plan_execution.parameter_resolver")
    return module.ParameterResolver(registry, plan)


def make_plan():
    return AgentPlan(
        plan_id="adapter_test",
        scene=ScenePlan(
            objects=[
                SceneObject(
                    id="liquid",
                    asset_id="ErlenmeyerFlask",
                    instance_name="LiquidFlask1",
                    role="container",
                    properties={"content_phase": "liquid"},
                ),
                SceneObject(
                    id="solid",
                    asset_id="ErlenmeyerFlask",
                    instance_name="SolidFlask1",
                    role="container",
                    properties={"content_phase": "solid"},
                ),
                SceneObject(
                    id="platform",
                    asset_id="TargetPlatform",
                    instance_name="TargetPlatform1",
                    role="placement_target",
                ),
                SceneObject(
                    id="plate",
                    asset_id="HeatingPlate",
                    instance_name="HeatingPlate1",
                    role="device",
                ),
                SceneObject(
                    id="scale",
                    asset_id="ElectronicScale",
                    instance_name="ElectronicScale1",
                    role="device",
                ),
                SceneObject(
                    id="door",
                    asset_id="DryingBox",
                    instance_name="DryingBox1",
                    role="door_device",
                ),
            ]
        ),
        actions=[],
    )


class FakeAtomicController:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.forward_calls = []
        self.reset_calls = []
        self.done = False
        self._event = 0

    def forward(self, **kwargs):
        self.events.append("controller.forward")
        self.forward_calls.append(kwargs)
        self.done = True
        return "robot-action"

    def is_done(self):
        return self.done

    def reset(self, **kwargs):
        self.events.append(("controller.reset", kwargs.copy()))
        self.reset_calls.append(kwargs.copy())
        self.done = False


class FakeGripperControl:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.position_updates = 0
        self.pose_updates = 0
        self.pose_tracking_calls = []
        self.position_tracking_starts = 0
        self.release_count = 0
        self.gripper_frame_path = "/World/Franka/panda_hand/tool_center"
        self.grasped_object_path = None
        self._relative_mat = None

    def begin_position_tracking(self):
        self.events.append("gripper.begin_position_tracking")
        self.position_tracking_starts += 1
        self._relative_mat = None

    def update_grasped_object_position(self):
        self.events.append("gripper.update_position")
        self.position_updates += 1

    def init_pose_tracking(self, object_path, gripper_frame_path):
        self.events.append("gripper.init_pose_tracking")
        self.pose_tracking_calls.append((object_path, gripper_frame_path))
        self._relative_mat = object()

    def update_grasped_object_pose(self):
        self.events.append("gripper.update_pose")
        self.pose_updates += 1

    def release_object(self):
        self.events.append("gripper.release")
        self.release_count += 1


class FakeRobot:
    def __init__(self):
        self.velocities = np.linspace(0.0, 0.8, 9)
        self.articulation_controller = object()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def get_articulation_controller(self):
        return self.articulation_controller


class FakeCspaceController:
    def __init__(self):
        self.forward_calls = []
        self.reset_count = 0

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        return "cspace-action"

    def reset(self):
        self.reset_count += 1


class FakeBindingGripper:
    def __init__(self):
        self.bindings = []
        self.resolve_rigid_body_flags = []
        self.grasped_object_path = None

    def add_object_to_gripper(
        self,
        object_path,
        gripper_frame_path,
        *,
        resolve_rigid_body=False,
    ):
        self.bindings.append((object_path, gripper_frame_path))
        self.resolve_rigid_body_flags.append(resolve_rigid_body)
        self.grasped_object_path = object_path


class ParameterResolverTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.registry = CapabilityRegistry.load_default(ROOT)
        self.resolver = parameter_resolver(self.registry, self.plan)

    def test_action_manifests_define_all_required_defaults(self):
        expected = {
            "pick": {
                "orientation_profile": "default",
                "pre_offset_z": 0.12,
                "after_offset_z": 0.15,
                "pre_offset_x": 0.1,
            },
            "place": {
                "orientation_profile": "default",
                "pre_place_z": 0.2,
                "place_offset_z": 0.05,
            },
            "pour": {"orientation_profile": "pour_default"},
            "press": {"orientation_profile": "default", "press_distance": 0.07},
            "press_z": {
                "orientation_profile": "press_z_default",
                "press_distance": 0.05,
            },
            "shake": {"orientation_profile": "default", "shake_distance": 0.1},
            "open": {
                "orientation_profile": "open_default",
                "furniture_type": "door",
                "angle": 50.0,
            },
            "close": {
                "orientation_profile": "close_default",
                "furniture_type": "door",
                "angle": 50.0,
            },
        }
        actual = {
            name: definition.default_parameters
            for name, definition in self.registry.actions.definitions.items()
        }
        self.assertEqual(actual, expected)

    def test_variant_defaults_cover_solid_and_liquid_protocol_objects(self):
        solid_pick = self.resolver.resolve(
            ActionStep(id="step_001", type=ActionType.PICK, object="solid")
        )
        solid_place = self.resolver.resolve(
            ActionStep(
                id="step_002",
                type=ActionType.PLACE,
                object="solid",
                target="platform",
            )
        )
        liquid_pick = self.resolver.resolve(
            ActionStep(id="step_003", type=ActionType.PICK, object="liquid")
        )
        liquid_pour = self.resolver.resolve(
            ActionStep(
                id="step_004",
                type=ActionType.POUR,
                object="liquid",
                target="solid",
            )
        )
        liquid_place = self.resolver.resolve(
            ActionStep(
                id="step_005",
                type=ActionType.PLACE,
                object="liquid",
                target="platform",
            )
        )

        self.assertEqual(solid_pick["orientation_profile"], "pick_solid")
        self.assertEqual(solid_pick["gripper_distance"], 0.012)
        self.assertEqual(solid_place["orientation_profile"], "default")
        self.assertEqual(solid_place["place_offset_z"], 0.12)
        self.assertEqual(liquid_pick["orientation_profile"], "default")
        self.assertEqual(liquid_pick["gripper_distance"], 0.008)
        self.assertEqual(liquid_pour["orientation_profile"], "pour_default")
        self.assertEqual(liquid_pour["pour_speed"], -1.0)
        self.assertEqual(liquid_place["orientation_profile"], "default")
        self.assertEqual(liquid_place["place_offset_z"], 0.08)

    def test_step_parameters_override_asset_and_action_defaults_without_mutation(self):
        step = ActionStep(
            id="step_001",
            type=ActionType.PICK,
            object="liquid",
            parameters={
                "orientation_profile": "pick_solid",
                "pre_offset_z": 0.2,
                "gripper_distance": 0.02,
            },
        )
        definition_before = deepcopy(
            self.registry.assets.get("ErlenmeyerFlask").model_dump(mode="python")
        )
        action_before = deepcopy(
            self.registry.actions.get("pick").model_dump(mode="python")
        )
        step_before = deepcopy(step.model_dump(mode="python"))

        first = self.resolver.resolve(step)
        first["pre_offset_z"] = 999.0
        second = self.resolver.resolve(step)

        self.assertEqual(second["orientation_profile"], "pick_solid")
        self.assertEqual(second["pre_offset_z"], 0.2)
        self.assertEqual(second["after_offset_z"], 0.15)
        self.assertEqual(second["gripper_distance"], 0.02)
        self.assertEqual(
            self.registry.assets.get("ErlenmeyerFlask").model_dump(mode="python"),
            definition_before,
        )
        self.assertEqual(
            self.registry.actions.get("pick").model_dump(mode="python"),
            action_before,
        )
        self.assertEqual(step.model_dump(mode="python"), step_before)

    def test_resolved_asset_defaults_are_detached_from_registry_models(self):
        definition = self.registry.assets.get("ErlenmeyerFlask")
        before = deepcopy(definition.model_dump(mode="python"))

        resolved = self.registry.assets.resolve(
            "ErlenmeyerFlask", {"content_phase": "liquid"}
        )
        resolved.action_defaults["pick"]["gripper_distance"] = 0.03
        again = self.registry.assets.resolve(
            "ErlenmeyerFlask", {"content_phase": "liquid"}
        )

        self.assertEqual(again.action_defaults["pick"]["gripper_distance"], 0.008)
        self.assertEqual(definition.model_dump(mode="python"), before)

    def test_variant_selection_uses_default_fallback_for_unknown_property_value(self):
        default_asset = self.registry.assets.resolve(
            "ErlenmeyerFlask", {"content_phase": "unexpected"}
        )
        expected_path = self.registry.assets.get("ErlenmeyerFlask").variants["default"]
        self.assertEqual(default_asset.usd_path, expected_path)
        self.assertEqual(default_asset.action_defaults, {})

    def test_missing_references_and_unknown_actions_fail_deterministically(self):
        cases = [
            (
                ActionStep(id="step_001", type=ActionType.PICK),
                ValueError,
                "requires object",
            ),
            (
                ActionStep(
                    id="step_002", type=ActionType.PLACE, object="solid"
                ),
                ValueError,
                "requires target",
            ),
            (
                ActionStep(
                    id="step_003", type=ActionType.PICK, object="missing"
                ),
                KeyError,
                "missing",
            ),
            (
                ActionStep(
                    id="step_004",
                    type=ActionType.PLACE,
                    object="solid",
                    target="missing",
                ),
                KeyError,
                "missing",
            ),
        ]
        for step, error_type, message in cases:
            with self.subTest(step=step.id):
                with self.assertRaisesRegex(error_type, message):
                    self.resolver.resolve(step)

        unknown = SimpleNamespace(
            type=SimpleNamespace(value="unknown_action"),
            object=None,
            target=None,
            parameters={},
        )
        with self.assertRaisesRegex(KeyError, "unknown_action"):
            self.resolver.resolve(unknown)

    def test_direct_resolver_rejects_unsupported_object_and_target_capabilities(self):
        cases = [
            (
                ActionStep(
                    id="step_001", type=ActionType.PICK, object="door"
                ),
                "does not support pick",
            ),
            (
                ActionStep(
                    id="step_002",
                    type=ActionType.PLACE,
                    object="solid",
                    target="liquid",
                ),
                "place_target",
            ),
        ]
        for step, message in cases:
            with self.subTest(step=step.id):
                with self.assertRaisesRegex(ValueError, message):
                    self.resolver.resolve(step)

    def test_direct_resolver_rejects_target_category_mismatch(self):
        assets = AssetRegistry(
            {
                "Source": AssetDefinition(
                    category="container",
                    usd_path="source.usd",
                    supported_actions=["pour"],
                ),
                "WrongTarget": AssetDefinition(
                    category="door_device",
                    usd_path="target.usd",
                    supported_actions=["pour"],
                ),
            }
        )
        registry = SimpleNamespace(
            assets=assets,
            actions=ActionRegistry(self.registry.actions.definitions),
            root=ROOT,
        )
        plan = AgentPlan(
            plan_id="category_mismatch",
            scene=ScenePlan(
                objects=[
                    SceneObject(
                        id="source",
                        asset_id="Source",
                        instance_name="Source1",
                        role="container",
                    ),
                    SceneObject(
                        id="target",
                        asset_id="WrongTarget",
                        instance_name="WrongTarget1",
                        role="target",
                    ),
                ]
            ),
            actions=[],
        )
        resolver = parameter_resolver(registry, plan)
        step = ActionStep(
            id="step_001",
            type=ActionType.POUR,
            object="source",
            target="target",
        )

        with self.assertRaisesRegex(ValueError, "target category"):
            resolver.resolve(step)


class StateAndOrientationResolverTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()

    def test_state_resolver_uses_exact_instance_keys_and_returns_copy(self):
        resolver = adapter_module().StateResolver(self.plan)
        source = np.array([0.1, 0.2, 0.3])
        state = {
            "LiquidFlask1_grisp_position": source,
            "LiquidFlask1_position": np.array([9.0, 9.0, 9.0]),
        }

        result = resolver.position(
            state, "liquid", "grisp_position", "position"
        )
        result[0] = 10.0

        np.testing.assert_allclose(source, [0.1, 0.2, 0.3])
        self.assertEqual(resolver.instance("liquid"), "LiquidFlask1")

    def test_state_resolver_reports_the_anchor_that_was_actually_selected(self):
        resolver = adapter_module().StateResolver(self.plan)

        position, anchor = resolver.position_with_anchor(
            {"LiquidFlask1_position": np.array([0.1, 0.2, 0.3])},
            "liquid",
            "grisp_position",
            "position",
        )

        np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
        self.assertEqual(anchor, "position")

    def test_state_resolver_does_not_fuzzy_match_and_reports_missing_keys(self):
        resolver = adapter_module().StateResolver(self.plan)
        with self.assertRaisesRegex(KeyError, "LiquidFlask1_position"):
            resolver.position(
                {"prefix_LiquidFlask1_position": np.zeros(3)},
                "liquid",
                "position",
            )
        with self.assertRaisesRegex(KeyError, "unknown"):
            resolver.position({}, "unknown", "position")

    def test_orientation_profiles_match_xyz_euler_degrees(self):
        module = adapter_module()
        expected_profiles = {
            "default": [0, 90, 30],
            "pick_solid": [0, 90, 20],
            "pour_default": [0, 90, 10],
            "press_z_default": [0, 80, 0],
            "open_default": [0, 90, 0],
            "close_default": [0, 110, 0],
        }
        self.assertEqual(module.load_orientation_profiles(), expected_profiles)
        for name, euler_degrees in expected_profiles.items():
            with self.subTest(profile=name):
                actual = module.orientation(
                    {"orientation_profile": name}
                )
                expected_rotation = Rotation.from_euler(
                    "xyz", euler_degrees, degrees=True
                )
                np.testing.assert_allclose(
                    rotation_from_wxyz(actual).as_matrix(),
                    expected_rotation.as_matrix(),
                    atol=1e-12,
                )

    def test_orientation_loading_is_cwd_independent_and_unknown_profiles_fail(self):
        module = adapter_module()
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                clear_orientation_cache(module)
                actual = module.orientation(
                    {"orientation_profile": "default"}
                )
            finally:
                os.chdir(old_cwd)
        expected = Rotation.from_euler("xyz", [0, 90, 30], degrees=True)
        np.testing.assert_allclose(
            rotation_from_wxyz(actual).as_matrix(),
            expected.as_matrix(),
            atol=1e-12,
        )
        with self.assertRaisesRegex(KeyError, "unknown orientation profile"):
            module.orientation({"orientation_profile": "missing"})

    def test_orientation_loader_returns_detached_nested_values(self):
        module = adapter_module()
        first = module.load_orientation_profiles()
        first["default"][0] = 999.0
        first["default"].append(123.0)

        second = module.load_orientation_profiles()
        self.assertEqual(second["default"], [0, 90, 30])
        actual = module.orientation({"orientation_profile": "default"})
        expected = Rotation.from_euler("xyz", [0, 90, 30], degrees=True)
        np.testing.assert_allclose(
            rotation_from_wxyz(actual).as_matrix(),
            expected.as_matrix(),
            atol=1e-12,
        )

    def test_identity_profile_is_scalar_first_without_180_degree_error(self):
        with temporary_orientation_yaml(
            "profiles:\n  identity: [0, 0, 0]\n"
        ) as module:
            actual = module.orientation({"orientation_profile": "identity"})
        np.testing.assert_allclose(actual, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            rotation_from_wxyz(actual).apply([1.0, 2.0, 3.0]),
            [1.0, 2.0, 3.0],
            atol=1e-12,
        )

    def test_orientation_yaml_duplicate_keys_fail_closed(self):
        content = """profiles:
  duplicate: [0, 0, 0]
  duplicate: [0, 90, 0]
"""
        with temporary_orientation_yaml(content) as module:
            with self.assertRaises(Exception) as caught:
                module.load_orientation_profiles()
        self.assertIsInstance(caught.exception, ValueError)
        self.assertIn("orientation profiles", str(caught.exception))
        self.assertIn("duplicate", str(caught.exception))

    def test_orientation_yaml_schema_rejects_bad_shapes_types_and_nonfinite_values(self):
        invalid_documents = {
            "missing profiles mapping": "wrong: {}\n",
            "empty profile name": 'profiles:\n  "": [0, 0, 0]\n',
            "wrong vector length": "profiles:\n  bad: [0, 90]\n",
            "boolean component": "profiles:\n  bad: [0, false, 0]\n",
            "string component": 'profiles:\n  bad: [0, "90", 0]\n',
            "nan component": "profiles:\n  bad: [0, .nan, 0]\n",
        }
        for label, content in invalid_documents.items():
            with self.subTest(label=label):
                with temporary_orientation_yaml(content) as module:
                    with self.assertRaises(Exception) as caught:
                        module.load_orientation_profiles()
                self.assertIsInstance(caught.exception, ValueError)
                self.assertIn("orientation profiles", str(caught.exception))


class PlanActionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.registry = CapabilityRegistry.load_default(ROOT)
        self.parameters = parameter_resolver(self.registry, self.plan)
        self.resolver = adapter_module().StateResolver(self.plan)
        self.joints = np.linspace(0.0, 0.8, 9)
        self.gripper_position = np.array([0.1, 0.2, 1.0])

    def test_pick_adapter_translates_typed_step_and_updates_after_forward(self):
        events = []
        atomic = FakeAtomicController(events)
        gripper = FakeGripperControl(events)
        adapter = adapter_module().PickActionAdapter(
            atomic, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001", type=ActionType.PICK, object="liquid"
        )
        adapter.prepare(step, SimpleNamespace())

        action = adapter.step(
            {
                "LiquidFlask1_grisp_position": np.array([0.3, 0.4, 0.8]),
                "LiquidFlask1_position": np.array([8.0, 8.0, 8.0]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        self.assertEqual(action, "robot-action")
        kwargs = atomic.forward_calls[-1]
        np.testing.assert_allclose(kwargs["picking_position"], [0.3, 0.4, 0.8])
        np.testing.assert_allclose(kwargs["current_joint_positions"], self.joints)
        self.assertEqual(kwargs["object_name"], "LiquidFlask1")
        np.testing.assert_allclose(kwargs["object_size"], DEFAULT_SIZE)
        self.assertIs(kwargs["gripper_control"], gripper)
        np.testing.assert_allclose(kwargs["gripper_position"], self.gripper_position)
        np.testing.assert_allclose(
            rotation_from_wxyz(
                kwargs["end_effector_orientation"]
            ).as_matrix(),
            Rotation.from_euler(
                "xyz", [0, 90, 30], degrees=True
            ).as_matrix(),
            atol=1e-12,
        )
        self.assertEqual(kwargs["pre_offset_z"], 0.12)
        self.assertEqual(kwargs["after_offset_z"], 0.15)
        self.assertEqual(kwargs["pre_offset_x"], 0.1)
        self.assertEqual(kwargs["gripper_distances"], 0.008)
        self.assertEqual(kwargs["object_prim_path"], "/World/LiquidFlask1")
        self.assertEqual(kwargs["pick_z_offset"], 0.0)
        self.assertEqual(
            events[-2:], ["controller.forward", "gripper.update_position"]
        )
        self.assertTrue(adapter.is_done())
        adapter.reset()
        self.assertEqual(atomic.reset_calls[-1], {})
        self.assertFalse(adapter.is_done())

    def test_pick_adapter_preserves_legacy_z_offset_for_position_fallback(self):
        atomic = FakeAtomicController()
        gripper = FakeGripperControl()
        adapter = adapter_module().PickActionAdapter(
            atomic, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001", type=ActionType.PICK, object="liquid"
        )
        adapter.prepare(step, SimpleNamespace())

        adapter.step(
            {
                "LiquidFlask1_position": np.array([0.3, 0.4, 0.8]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        kwargs = atomic.forward_calls[-1]
        np.testing.assert_allclose(kwargs["picking_position"], [0.3, 0.4, 0.8])
        self.assertIsNone(kwargs["pick_z_offset"])

    def test_pick_prepare_releases_stale_binding_before_every_attempt(self):
        events = []
        atomic = FakeAtomicController(events)
        gripper = FakeGripperControl(events)
        adapter = adapter_module().PickActionAdapter(
            atomic, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001", type=ActionType.PICK, object="solid"
        )

        adapter.prepare(step, SimpleNamespace())
        adapter.step(
            {
                "SolidFlask1_grisp_position": np.array([0.3, 0.4, 0.8]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )
        adapter.prepare(step, SimpleNamespace())
        adapter.step(
            {
                "SolidFlask1_grisp_position": np.array([0.3, 0.4, 0.8]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        self.assertEqual(gripper.release_count, 2)
        self.assertEqual(
            events,
            [
                "gripper.release",
                "controller.forward",
                "gripper.update_position",
                "gripper.release",
                "controller.forward",
                "gripper.update_position",
            ],
        )

    def test_successful_pick_reset_keeps_binding_for_following_place(self):
        events = []
        atomic = FakeAtomicController(events)
        gripper = FakeGripperControl(events)
        adapter = adapter_module().PickActionAdapter(
            atomic, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001", type=ActionType.PICK, object="solid"
        )
        adapter.prepare(step, SimpleNamespace())
        releases_after_prepare = gripper.release_count

        adapter.reset()

        self.assertEqual(gripper.release_count, releases_after_prepare)

    def test_place_adapter_updates_grasp_before_forward_and_uses_asset_offset(self):
        events = []
        atomic = FakeAtomicController(events)
        gripper = FakeGripperControl(events)
        adapter = adapter_module().PlaceActionAdapter(
            atomic, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001",
            type=ActionType.PLACE,
            object="solid",
            target="platform",
        )
        adapter.prepare(step, SimpleNamespace())

        self.assertEqual(gripper.position_tracking_starts, 1)

        adapter.step(
            {
                "SolidFlask1_position": np.array([0.2, 0.0, 0.8]),
                "TargetPlatform1_place_position": np.array([0.5, 0.1, 0.7]),
                "TargetPlatform1_position": np.array([9.0, 9.0, 9.0]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        kwargs = atomic.forward_calls[-1]
        np.testing.assert_allclose(kwargs["place_position"], [0.5, 0.1, 0.7])
        np.testing.assert_allclose(kwargs["current_joint_positions"], self.joints)
        self.assertIs(kwargs["gripper_control"], gripper)
        np.testing.assert_allclose(kwargs["gripper_position"], self.gripper_position)
        self.assertEqual(kwargs["pre_place_z"], 0.2)
        self.assertEqual(kwargs["place_offset_z"], 0.12)
        self.assertEqual(
            events[-2:], ["gripper.update_position", "controller.forward"]
        )

    def test_place_adapter_falls_back_to_target_root_position(self):
        atomic = FakeAtomicController()
        gripper = FakeGripperControl()
        adapter = adapter_module().PlaceActionAdapter(
            atomic, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001",
            type=ActionType.PLACE,
            object="solid",
            target="platform",
        )
        adapter.prepare(step, SimpleNamespace())

        adapter.step(
            {
                "SolidFlask1_position": np.array([0.2, 0.0, 0.8]),
                "TargetPlatform1_position": np.array([0.5, 0.1, 0.7]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        np.testing.assert_allclose(
            atomic.forward_calls[-1]["place_position"],
            [0.5, 0.1, 0.7],
        )

    def test_pour_adapter_uses_translation_tracking_during_approach(self):
        events = []
        atomic = FakeAtomicController(events)
        gripper = FakeGripperControl(events)
        gripper.grasped_object_path = "/World/LiquidFlask1/RigidBody"
        robot = FakeRobot()
        adapter = adapter_module().PourActionAdapter(
            atomic, robot, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001",
            type=ActionType.POUR,
            object="liquid",
            target="solid",
        )
        adapter.prepare(step, SimpleNamespace())

        action = adapter.step(
            {
                "SolidFlask1_position": np.array([0.2, 0.0, 0.8]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        self.assertEqual(action, "robot-action")
        self.assertEqual(gripper.pose_tracking_calls, [])
        self.assertEqual(gripper.position_updates, 1)
        self.assertEqual(gripper.pose_updates, 0)
        kwargs = atomic.forward_calls[-1]
        self.assertIs(
            kwargs["articulation_controller"], robot.articulation_controller
        )
        np.testing.assert_allclose(kwargs["source_size"], DEFAULT_SIZE)
        np.testing.assert_allclose(kwargs["target_position"], [0.2, 0.0, 0.8])
        np.testing.assert_allclose(
            kwargs["current_joint_velocities"], robot.velocities
        )
        np.testing.assert_allclose(kwargs["gripper_position"], self.gripper_position)
        self.assertEqual(kwargs["source_name"], "LiquidFlask1")
        self.assertEqual(kwargs["pour_speed"], -1.0)
        np.testing.assert_allclose(
            rotation_from_wxyz(
                kwargs["target_end_effector_orientation"]
            ).as_matrix(),
            Rotation.from_euler(
                "xyz", [0, 90, 10], degrees=True
            ).as_matrix(),
            atol=1e-12,
        )
        self.assertEqual(
            events[-2:], ["controller.forward", "gripper.update_position"]
        )

    def test_pour_adapter_tracks_the_bound_rigid_body_during_tilt(self):
        atomic = FakeAtomicController()
        atomic._event = 2
        gripper = FakeGripperControl()
        gripper.grasped_object_path = "/World/LiquidFlask1/RigidBody"
        adapter = adapter_module().PourActionAdapter(
            atomic, FakeRobot(), gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001",
            type=ActionType.POUR,
            object="liquid",
            target="solid",
        )
        adapter.prepare(step, SimpleNamespace())

        adapter.step(
            {
                "SolidFlask1_position": np.array([0.2, 0.0, 0.8]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        self.assertEqual(
            gripper.pose_tracking_calls,
            [
                (
                    "/World/LiquidFlask1/RigidBody",
                    "/World/Franka/panda_hand/tool_center",
                )
            ],
        )
        self.assertEqual(gripper.position_updates, 0)
        self.assertEqual(gripper.pose_updates, 1)

    def test_press_adapter_configures_distance_only_during_prepare(self):
        atomic = FakeAtomicController()
        gripper = FakeGripperControl()
        adapter = adapter_module().PressActionAdapter(
            atomic, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001",
            type=ActionType.PRESS,
            target="plate",
            parameters={"press_distance": 0.09},
        )
        adapter.prepare(step, SimpleNamespace())
        adapter.step(
            {
                "HeatingPlate1_press_position": np.array([0.4, 0.0, 0.75]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        self.assertEqual(atomic.reset_calls, [{"initial_offset": 0.09}])
        kwargs = atomic.forward_calls[-1]
        self.assertNotIn("press_distance", kwargs)
        np.testing.assert_allclose(kwargs["target_position"], [0.4, 0.0, 0.75])
        np.testing.assert_allclose(kwargs["current_joint_positions"], self.joints)
        self.assertIs(kwargs["gripper_control"], gripper)
        np.testing.assert_allclose(kwargs["gripper_position"], self.gripper_position)

    def test_press_z_adapter_configures_distance_only_during_prepare(self):
        atomic = FakeAtomicController()
        gripper = FakeGripperControl()
        adapter = adapter_module().PressZActionAdapter(
            atomic, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001", type=ActionType.PRESS_Z, target="scale"
        )
        adapter.prepare(step, SimpleNamespace())
        adapter.step(
            {
                "ElectronicScale1_pressz_position": np.array([0.6, 0.0, 0.72]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        self.assertEqual(atomic.reset_calls, [{"press_distance": 0.05}])
        kwargs = atomic.forward_calls[-1]
        self.assertNotIn("press_distance", kwargs)
        np.testing.assert_allclose(kwargs["target_position"], [0.6, 0.0, 0.72])
        np.testing.assert_allclose(
            rotation_from_wxyz(
                kwargs["end_effector_orientation"]
            ).as_matrix(),
            Rotation.from_euler(
                "xyz", [0, 80, 0], degrees=True
            ).as_matrix(),
            atol=1e-12,
        )

    def test_shake_adapter_uses_current_gripper_origin_and_updates_grasp(self):
        atomic = FakeAtomicController()
        gripper = FakeGripperControl()
        adapter = adapter_module().ShakeActionAdapter(
            atomic, gripper, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001",
            type=ActionType.SHAKE,
            object="liquid",
            parameters={"shake_distance": 0.12},
        )
        adapter.prepare(step, SimpleNamespace())
        adapter.step(
            {
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        self.assertEqual(atomic.reset_calls, [{"shake_distance": 0.12}])
        kwargs = atomic.forward_calls[-1]
        np.testing.assert_allclose(kwargs["current_joint_positions"], self.joints)
        np.testing.assert_allclose(kwargs["gripper_position"], self.gripper_position)
        self.assertEqual(gripper.position_updates, 1)

    def test_open_adapter_passes_exact_handle_and_revolute_anchors(self):
        atomic = FakeAtomicController()
        adapter = adapter_module().OpenActionAdapter(
            atomic, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001",
            type=ActionType.OPEN,
            target="door",
            parameters={"furniture_type": "drawer", "angle": 60.0},
        )
        adapter.prepare(step, SimpleNamespace())
        adapter.step(
            {
                "DryingBox1_handle_position": np.array([0.7, 0.1, 0.9]),
                "DryingBox1_revolute_joint_position": np.array([0.5, 0.1, 0.8]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        self.assertEqual(atomic.reset_calls, [{"furniture_type": "drawer"}])
        kwargs = atomic.forward_calls[-1]
        np.testing.assert_allclose(kwargs["handle_position"], [0.7, 0.1, 0.9])
        np.testing.assert_allclose(
            kwargs["revolute_joint_position"], [0.5, 0.1, 0.8]
        )
        self.assertEqual(kwargs["angle"], 60.0)
        np.testing.assert_allclose(
            rotation_from_wxyz(
                kwargs["end_effector_orientation"]
            ).as_matrix(),
            Rotation.from_euler(
                "xyz", [0, 90, 0], degrees=True
            ).as_matrix(),
            atol=1e-12,
        )

    def test_close_adapter_passes_optional_push_distance(self):
        atomic = FakeAtomicController()
        adapter = adapter_module().CloseActionAdapter(
            atomic, self.resolver, self.parameters
        )
        step = ActionStep(
            id="step_001",
            type=ActionType.CLOSE,
            target="door",
            parameters={"push_distance": 0.2},
        )
        adapter.prepare(step, SimpleNamespace())
        adapter.step(
            {
                "DryingBox1_handle_position": np.array([0.8, 0.2, 0.9]),
                "DryingBox1_revolute_joint_position": np.array([0.5, 0.2, 0.8]),
                "joint_positions": self.joints,
                "gripper_position": self.gripper_position,
            }
        )

        self.assertEqual(atomic.reset_calls, [{"furniture_type": "door"}])
        kwargs = atomic.forward_calls[-1]
        self.assertEqual(kwargs["push_distance"], 0.2)
        self.assertEqual(kwargs["angle"], 50.0)
        np.testing.assert_allclose(
            rotation_from_wxyz(
                kwargs["end_effector_orientation"]
            ).as_matrix(),
            Rotation.from_euler(
                "xyz", [0, 110, 0], degrees=True
            ).as_matrix(),
            atol=1e-12,
        )

    def test_close_adapter_cleanup_and_reprepare_preserve_custom_same_type_timing(self):
        custom_events = [0.1, 0.2, 0.3, 0.4]
        with loaded_atomic_controller_module("close_controller.py") as (
            atomic_module,
            _,
        ):
            controller = atomic_module.CloseController(
                name="close",
                cspace_controller=FakeCspaceController(),
                events_dt=custom_events,
                furniture_type="drawer",
            )
            parameters = SimpleNamespace(
                resolve=lambda step: {
                    "orientation_profile": "close_default",
                    "furniture_type": "drawer",
                    "angle": 50.0,
                }
            )
            adapter = adapter_module().CloseActionAdapter(
                controller, self.resolver, parameters
            )
            step = ActionStep(
                id="step_001",
                type=ActionType.CLOSE,
                target="door",
                parameters={"furniture_type": "drawer"},
            )

            adapter.prepare(step, SimpleNamespace())
            self.assertEqual(controller._events_dt, custom_events)
            adapter.reset()
            self.assertEqual(controller._events_dt, custom_events)
            adapter.prepare(step, SimpleNamespace())
            self.assertEqual(controller._events_dt, custom_events)

    def test_repeated_prepare_reconfigures_controller_for_each_typed_step(self):
        atomic = FakeAtomicController()
        adapter = adapter_module().PressZActionAdapter(
            atomic, FakeGripperControl(), self.resolver, self.parameters
        )
        first = ActionStep(
            id="step_001",
            type=ActionType.PRESS_Z,
            target="scale",
            parameters={"press_distance": 0.04},
        )
        second = ActionStep(
            id="step_002",
            type=ActionType.PRESS_Z,
            target="scale",
            parameters={"press_distance": 0.08},
        )

        adapter.prepare(first, SimpleNamespace())
        adapter.prepare(second, SimpleNamespace())

        self.assertEqual(
            atomic.reset_calls,
            [{"press_distance": 0.04}, {"press_distance": 0.08}],
        )


class PureImportTests(unittest.TestCase):
    def test_importing_plan_action_adapters_does_not_load_omni(self):
        code = """
import sys
import controllers
import controllers.plan_action_adapters

expected = [
    'PickController', 'PlaceController', 'StirController', 'PourController',
    'OpenController', 'CloseController', 'ShakeController',
    'PressController', 'PressZController',
]
assert controllers.__all__ == expected, controllers.__all__
loaded = sorted(name for name in sys.modules if name == 'omni' or name.startswith('omni.'))
assert not loaded, loaded
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


def _class_method(path, class_name, method_name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _default_for(function, argument_name):
    positional = function.args.posonlyargs + function.args.args
    default_offset = len(positional) - len(function.args.defaults)
    defaults = {
        argument.arg: default
        for argument, default in zip(
            positional[default_offset:], function.args.defaults
        )
    }
    return defaults[argument_name]


class AtomicControllerRuntimeRegressionTests(unittest.TestCase):
    def test_place_motion_event_does_not_advance_on_elapsed_time(self):
        with loaded_atomic_controller_module("place_controller.py") as (
            module,
            _,
        ):
            controller = module.PlaceController(
                name="place",
                cspace_controller=FakeCspaceController(),
                gripper=object(),
                events_dt=[1.0] * 6,
            )
            controller._start = False
            controller.forward(
                place_position=np.array([0.2, 0.1, 0.8]),
                current_joint_positions=np.zeros(9),
                gripper_control=FakeGripperControl(),
                gripper_position=np.array([0.0, 0.0, 1.0]),
            )

        self.assertEqual(controller._event, 0)

    def test_place_bounds_rmp_joint_targets_before_articulation(self):
        with loaded_atomic_controller_module("place_controller.py") as (
            module,
            articulation_action,
        ):
            class DivergingCspaceController(FakeCspaceController):
                def forward(self, **kwargs):
                    self.forward_calls.append(kwargs)
                    return articulation_action(
                        joint_positions=[
                            0.01,
                            0.20,
                            -0.20,
                            np.nan,
                            np.inf,
                            None,
                            -0.04,
                        ]
                    )

            controller = module.PlaceController(
                name="place",
                cspace_controller=DivergingCspaceController(),
                gripper=object(),
            )
            controller._start = False
            controller._event = 1
            action = controller.forward(
                place_position=np.array([0.2, -0.1, 0.8]),
                current_joint_positions=np.zeros(9),
                gripper_control=FakeGripperControl(),
                gripper_position=np.array([0.0, 0.0, 1.2]),
            )

        self.assertEqual(
            action.joint_positions,
            [0.01, 0.05, -0.05, 0.0, 0.0, None, -0.04],
        )

    def test_place_does_not_emit_targets_for_nonfinite_current_joints(self):
        with loaded_atomic_controller_module("place_controller.py") as (
            module,
            articulation_action,
        ):
            class FiniteCspaceController(FakeCspaceController):
                def forward(self, **kwargs):
                    self.forward_calls.append(kwargs)
                    return articulation_action(joint_positions=[0.02] * 7)

            controller = module.PlaceController(
                name="place",
                cspace_controller=FiniteCspaceController(),
                gripper=object(),
            )
            controller._start = False
            controller._event = 1
            current = np.zeros(9)
            current[2] = np.nan
            action = controller.forward(
                place_position=np.array([0.2, -0.1, 0.8]),
                current_joint_positions=current,
                gripper_control=FakeGripperControl(),
                gripper_position=np.array([0.0, 0.0, 1.2]),
            )

        self.assertIsNone(action.joint_positions[2])

    def test_pour_samples_each_approach_height_once_per_action(self):
        with loaded_atomic_controller_module("pour_controller.py") as (
            module,
            _,
        ):
            cspace_controller = FakeCspaceController()
            controller = module.PourController(
                name="pour", cspace_controller=cspace_controller
            )
            controller._random_height_1 = 0.33
            with patch.object(module.np.random, "uniform", return_value=0.39):
                for _ in range(2):
                    controller.forward(
                        articulation_controller=object(),
                        source_size=DEFAULT_SIZE.copy(),
                        target_position=np.array([0.2, 0.1, 0.8]),
                        current_joint_velocities=np.zeros(9),
                        gripper_position=np.array([2.0, 2.0, 1.0]),
                        source_name="LiquidFlask1",
                    )

        first_target, second_target = (
            call["target_end_effector_position"]
            for call in cspace_controller.forward_calls
        )
        np.testing.assert_allclose(first_target, second_target)

    def test_pour_bounds_rmp_joint_targets_during_approach(self):
        with loaded_atomic_controller_module("pour_controller.py") as (
            module,
            articulation_action,
        ):
            class DivergingCspaceController(FakeCspaceController):
                def forward(self, **kwargs):
                    self.forward_calls.append(kwargs)
                    return articulation_action(
                        joint_positions=[
                            0.01,
                            0.20,
                            -0.20,
                            np.nan,
                            np.inf,
                            None,
                            -2.0,
                        ]
                    )

            controller = module.PourController(
                name="pour", cspace_controller=DivergingCspaceController()
            )
            action = controller.forward(
                articulation_controller=object(),
                source_size=DEFAULT_SIZE.copy(),
                target_position=np.array([0.2, 0.1, 0.8]),
                current_joint_velocities=np.zeros(9),
                current_joint_positions=np.zeros(9),
                gripper_position=np.array([2.0, 2.0, 1.0]),
                source_name="LiquidFlask1",
            )

        self.assertEqual(
            action.joint_positions,
            [0.01, 0.05, -0.05, 0.0, 0.0, None, -0.05],
        )

    def test_pour_randomized_final_height_stays_inside_verifier_radius(self):
        with loaded_atomic_controller_module("pour_controller.py") as (
            module,
            _,
        ):
            with patch.object(
                module.np.random,
                "uniform",
                side_effect=lambda lower, upper: upper,
            ):
                cspace_controller = FakeCspaceController()
                controller = module.PourController(
                    name="pour", cspace_controller=cspace_controller
                )
            controller._event = 1
            target_position = np.array([0.2253, -0.1631, 0.9323])
            controller.forward(
                articulation_controller=object(),
                source_size=np.array([0.1051, -0.1051, 0.13103]),
                target_position=target_position,
                current_joint_velocities=np.zeros(9),
                gripper_position=np.array([2.0, 2.0, 1.0]),
                source_name="LiquidFlask1",
            )

        final_approach = cspace_controller.forward_calls[-1][
            "target_end_effector_position"
        ]
        self.assertLessEqual(
            np.linalg.norm(final_approach - target_position),
            0.25,
        )

    def test_pick_event_one_waits_when_xy_has_converged_but_z_has_not(self):
        with loaded_atomic_controller_module("pick_controller.py") as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 1
            controller.forward(
                picking_position=np.array([0.2, 0.3, 0.9]),
                current_joint_positions=np.zeros(9),
                object_name="SolidFlask1",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=FakeBindingGripper(),
                gripper_position=np.array([0.1, 0.3, 1.2]),
            )

        self.assertEqual(controller._event, 1)

    def test_pick_event_one_advances_when_xyz_have_converged(self):
        with loaded_atomic_controller_module("pick_controller.py") as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 1
            target_z = 0.9 + DEFAULT_SIZE[2] * 2 / 3
            controller.forward(
                picking_position=np.array([0.2, 0.3, 0.9]),
                current_joint_positions=np.zeros(9),
                object_name="SolidFlask1",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=FakeBindingGripper(),
                gripper_position=np.array([0.1, 0.3, target_z]),
            )

        self.assertEqual(controller._event, 2)

    def test_pick_explicit_anchor_has_no_legacy_z_offset_at_event_two(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            cspace_controller = FakeCspaceController()
            controller = module.PickController(
                name="pick", cspace_controller=cspace_controller
            )
            controller._start = False
            controller._event = 2
            controller.forward(
                picking_position=np.array([0.2, 0.3, 0.9]),
                current_joint_positions=np.zeros(9),
                object_name="SolidFlask1",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=FakeBindingGripper(),
                gripper_position=np.array([0.2, 0.3, 0.9]),
                pick_z_offset=0.0,
            )

        np.testing.assert_allclose(
            cspace_controller.forward_calls[-1]["target_end_effector_position"],
            [0.2, 0.3, 0.9],
        )

    def test_pick_legacy_default_keeps_computed_z_offset_at_event_two(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            cspace_controller = FakeCspaceController()
            controller = module.PickController(
                name="pick", cspace_controller=cspace_controller
            )
            controller._start = False
            controller._event = 2
            controller.forward(
                picking_position=np.array([0.2, 0.3, 0.9]),
                current_joint_positions=np.zeros(9),
                object_name="SolidFlask1",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=FakeBindingGripper(),
                gripper_position=np.zeros(3),
            )

        np.testing.assert_allclose(
            cspace_controller.forward_calls[-1]["target_end_effector_position"],
            [0.2, 0.3, 4.1],
        )

    def test_open_and_close_rotation_helpers_accept_and_return_wxyz(self):
        input_wxyz = wxyz_from_rotation(
            Rotation.from_euler("xyz", [15.0, 25.0, 35.0], degrees=True)
        )
        angle = 40.0
        expected = rotation_from_wxyz(input_wxyz) * Rotation.from_euler(
            "x", -angle, degrees=True
        )

        for filename, class_name in (
            ("open_controller.py", "OpenController"),
            ("close_controller.py", "CloseController"),
        ):
            with self.subTest(controller=class_name):
                with loaded_atomic_controller_module(filename) as (module, _):
                    kwargs = {
                        "name": "controller",
                        "cspace_controller": FakeCspaceController(),
                    }
                    if class_name == "OpenController":
                        kwargs["gripper"] = object()
                    controller = getattr(module, class_name)(**kwargs)
                    actual_wxyz = controller.rotate_quaternion_around_x(
                        input_wxyz, angle
                    )
                np.testing.assert_allclose(
                    rotation_from_wxyz(actual_wxyz).as_matrix(),
                    expected.as_matrix(),
                    atol=1e-12,
                )
                self.assertGreater(abs(actual_wxyz[0]), 0.1)

    def test_pick_explicit_gripper_width_is_scaled_once_at_controller_boundary(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 4
            action = controller.forward(
                picking_position=np.array([0.0, 0.0, 1.0]),
                current_joint_positions=np.zeros(9),
                object_name="container",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=FakeBindingGripper(),
                gripper_position=np.zeros(3),
                gripper_distances=0.008,
            )

        self.assertEqual(action.joint_positions[7], 0.8)
        self.assertEqual(action.joint_positions[8], 0.8)

    def test_pick_pre_offset_z_is_scaled_once_at_controller_boundary(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            cspace_controller = FakeCspaceController()
            controller = module.PickController(
                name="pick", cspace_controller=cspace_controller
            )
            controller._start = False
            controller.forward(
                picking_position=np.array([0.0, 0.0, 1.0]),
                current_joint_positions=np.zeros(9),
                object_name="container",
                object_size=np.array([4.0, 4.0, 8.0]),
                gripper_control=FakeBindingGripper(),
                gripper_position=np.zeros(3),
                pre_offset_z=0.12,
            )

        np.testing.assert_allclose(
            cspace_controller.forward_calls[0]["target_end_effector_position"],
            np.array([-10.0, 0.0, 21.0]),
        )

    def test_pick_plan_prim_path_overrides_legacy_binding_path_at_event_four(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 4
            gripper = FakeBindingGripper()
            controller.forward(
                picking_position=np.array([0.0, 0.0, 1.0]),
                current_joint_positions=np.zeros(9),
                object_name="LiquidFlask1",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=gripper,
                gripper_position=np.zeros(3),
                gripper_distances=0.008,
                object_prim_path="/World/LiquidFlask1",
            )

        self.assertEqual(
            gripper.bindings,
            [
                (
                    "/World/LiquidFlask1",
                    "/World/Franka/panda_hand/tool_center",
                )
            ],
        )
        self.assertEqual(gripper.resolve_rigid_body_flags, [True])

    def test_pick_plan_binding_waits_until_gripper_is_near_grasp_anchor(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 4
            gripper = FakeBindingGripper()

            controller.forward(
                picking_position=np.array([0.0, 0.0, 1.0]),
                current_joint_positions=np.zeros(9),
                object_name="LiquidFlask1",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=gripper,
                gripper_position=np.array([4.0, 0.0, 1.0]),
                object_prim_path="/World/LiquidFlask1",
            )

        self.assertEqual(gripper.bindings, [])
        self.assertEqual(controller._event, 2)

    def test_pick_plan_binding_accepts_gripper_within_proximity_gate(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 4
            gripper = FakeBindingGripper()

            controller.forward(
                picking_position=np.array([0.0, 0.0, 1.0]),
                current_joint_positions=np.zeros(9),
                object_name="LiquidFlask1",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=gripper,
                gripper_position=np.array([2.0, 0.0, 1.0]),
                object_prim_path="/World/LiquidFlask1",
            )

        self.assertEqual(len(gripper.bindings), 1)
        self.assertEqual(controller._event, 4)

    def test_pick_plan_binding_reuses_event_two_effective_grasp_target(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 2
            gripper = FakeBindingGripper()
            raw_position = np.array([0.0, 0.0, 1.0])
            effective_position = np.array([0.0, 0.0, 4.2])

            controller.forward(
                picking_position=raw_position.copy(),
                current_joint_positions=np.zeros(9),
                object_name="container",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=gripper,
                gripper_position=effective_position,
                object_prim_path="/World/Container",
            )
            controller._event = 4
            controller.forward(
                picking_position=raw_position.copy(),
                current_joint_positions=np.zeros(9),
                object_name="container",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=gripper,
                gripper_position=effective_position,
                object_prim_path="/World/Container",
            )

        self.assertEqual(len(gripper.bindings), 1)

    def test_pick_plan_binding_recomputes_target_if_object_moves_while_waiting(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 2
            gripper = FakeBindingGripper()
            old_raw_position = np.array([0.0, 0.0, 1.0])
            old_effective_position = np.array([0.0, 0.0, 4.2])

            controller.forward(
                picking_position=old_raw_position.copy(),
                current_joint_positions=np.zeros(9),
                object_name="container",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=gripper,
                gripper_position=old_effective_position,
                object_prim_path="/World/Container",
            )
            controller._event = 4
            controller.forward(
                picking_position=np.array([10.0, 0.0, 1.0]),
                current_joint_positions=np.zeros(9),
                object_name="container",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=gripper,
                gripper_position=old_effective_position,
                object_prim_path="/World/Container",
            )

        self.assertEqual(gripper.bindings, [])
        self.assertEqual(controller._event, 2)

    def test_pick_plan_prim_path_binds_only_once_across_event_four_frames(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 4
            gripper = FakeBindingGripper()

            for _ in range(2):
                controller.forward(
                    picking_position=np.array([0.0, 0.0, 1.0]),
                    current_joint_positions=np.zeros(9),
                    object_name="LiquidFlask1",
                    object_size=DEFAULT_SIZE.copy(),
                    gripper_control=gripper,
                    gripper_position=np.zeros(3),
                    object_prim_path="/World/LiquidFlask1",
                )

        self.assertEqual(len(gripper.bindings), 1)
        self.assertEqual(gripper.resolve_rigid_body_flags, [True])

    def test_pick_plan_prim_path_replaces_stale_gripper_binding(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 4
            gripper = FakeBindingGripper()
            gripper.grasped_object_path = "/World/StaleLegacyBinding"

            controller.forward(
                picking_position=np.array([0.0, 0.0, 1.0]),
                current_joint_positions=np.zeros(9),
                object_name="LiquidFlask1",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=gripper,
                gripper_position=np.zeros(3),
                object_prim_path="/World/LiquidFlask1",
            )

        self.assertEqual(
            gripper.bindings,
            [
                (
                    "/World/LiquidFlask1",
                    "/World/Franka/panda_hand/tool_center",
                )
            ],
        )
        self.assertEqual(gripper.resolve_rigid_body_flags, [True])

    def test_pick_reset_rebinds_the_same_explicit_target(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 4
            gripper = FakeBindingGripper()
            forward_kwargs = {
                "picking_position": np.array([0.0, 0.0, 1.0]),
                "current_joint_positions": np.zeros(9),
                "object_name": "LiquidFlask1",
                "object_size": DEFAULT_SIZE.copy(),
                "gripper_control": gripper,
                "gripper_position": np.zeros(3),
                "object_prim_path": "/World/LiquidFlask1",
            }

            controller.forward(**forward_kwargs)
            controller.reset()
            controller._start = False
            controller._event = 4
            forward_kwargs["picking_position"] = np.array([0.0, 0.0, 1.0])
            controller.forward(**forward_kwargs)

        self.assertEqual(len(gripper.bindings), 2)
        self.assertEqual(gripper.resolve_rigid_body_flags, [True, True])

    def test_pick_legacy_event_four_keeps_binding_each_frame(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 4
            gripper = FakeBindingGripper()

            for _ in range(2):
                controller.forward(
                    picking_position=np.array([0.0, 0.0, 1.0]),
                    current_joint_positions=np.zeros(9),
                    object_name="BeakerLegacy",
                    object_size=DEFAULT_SIZE.copy(),
                    gripper_control=gripper,
                    gripper_position=np.zeros(3),
                )

        self.assertEqual(len(gripper.bindings), 2)
        self.assertEqual(gripper.resolve_rigid_body_flags, [False, False])

    def test_pick_default_width_and_legacy_path_remain_compatible(self):
        with loaded_atomic_controller_module(
            "pick_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.PickController(
                name="pick", cspace_controller=FakeCspaceController()
            )
            controller._start = False
            controller._event = 4
            gripper = FakeBindingGripper()
            action = controller.forward(
                picking_position=np.array([0.0, 0.0, 1.0]),
                current_joint_positions=np.zeros(9),
                object_name="BeakerLegacy",
                object_size=DEFAULT_SIZE.copy(),
                gripper_control=gripper,
                gripper_position=np.zeros(3),
            )

        self.assertEqual(action.joint_positions[7], 2.0)
        self.assertEqual(
            gripper.bindings[0][0], "/World/LabScene/BeakerLegacy"
        )
        self.assertEqual(gripper.resolve_rigid_body_flags, [False])

    def test_close_push_distance_is_scaled_once_before_state_machine(self):
        with loaded_atomic_controller_module(
            "close_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.CloseController(
                name="close",
                cspace_controller=FakeCspaceController(),
                furniture_type="drawer",
            )
            captured = {}

            def capture_phase(*args):
                captured["push_distance"] = args[-1]
                return "close-action"

            controller._execute_phase = capture_phase
            controller.forward(
                handle_position=np.zeros(3),
                current_joint_positions=np.zeros(9),
                gripper_position=np.zeros(3),
                push_distance=0.2,
            )

        self.assertEqual(captured["push_distance"], 20.0)

    def test_close_does_not_treat_three_millimeters_as_twenty_centimeters(self):
        with loaded_atomic_controller_module(
            "close_controller.py", stage_units=0.01
        ) as (module, _):
            controller = module.CloseController(
                name="close",
                cspace_controller=FakeCspaceController(),
                events_dt=[0.1, 0.2, 0.3, 0.4],
                furniture_type="drawer",
            )
            controller._event = 1
            controller.init_handle_position = np.zeros(3)
            controller.forward(
                handle_position=np.array([0.3, 0.0, 0.0]),
                current_joint_positions=np.zeros(9),
                gripper_position=np.zeros(3),
                push_distance=0.2,
            )

        self.assertEqual(controller._event, 1)

    def test_close_reset_preserves_custom_timing_until_furniture_type_changes(self):
        custom_events = [0.1, 0.2, 0.3, 0.4]
        with loaded_atomic_controller_module("close_controller.py") as (module, _):
            controller = module.CloseController(
                name="close",
                cspace_controller=FakeCspaceController(),
                events_dt=custom_events,
                furniture_type="drawer",
            )
            controller.position_rotation_interp_iter = iter(())
            controller.init_handle_position = np.ones(3)

            controller.reset()
            self.assertEqual(controller._events_dt, custom_events)
            controller.reset(furniture_type="drawer")
            self.assertEqual(controller._events_dt, custom_events)
            controller.reset(furniture_type="door")

        self.assertEqual(controller._events_dt, [0.0025, 0.005, 0.005])
        self.assertEqual(controller._event, 0)
        self.assertEqual(controller._t, 0)
        self.assertIsNone(controller.position_rotation_interp_iter)
        self.assertIsNone(controller.init_handle_position)


class AtomicControllerSourceRegressionTests(unittest.TestCase):
    def test_press_distance_is_stage_scaled_during_construction_and_reset_only(self):
        init = _class_method(
            "controllers/atomic_actions/press_controller.py",
            "PressController",
            "__init__",
        )
        reset = _class_method(
            "controllers/atomic_actions/press_controller.py",
            "PressController",
            "reset",
        )
        forward = _class_method(
            "controllers/atomic_actions/press_controller.py",
            "PressController",
            "forward",
        )
        self.assertIn(
            "self._initial_offset = (initial_offset if initial_offset is not None else 0.07) / get_stage_units()",
            ast.unparse(init),
        )
        self.assertIn(
            "self._initial_offset = initial_offset / get_stage_units()",
            ast.unparse(reset),
        )
        self.assertIsNone(ast.literal_eval(_default_for(forward, "press_distance")))
        runtime_reads = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Name)
            and node.id == "press_distance"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(runtime_reads, [])

    def test_press_z_distance_drives_event_zero_and_is_stage_scaled(self):
        init = _class_method(
            "controllers/atomic_actions/pressZ_controller.py",
            "PressZController",
            "__init__",
        )
        reset = _class_method(
            "controllers/atomic_actions/pressZ_controller.py",
            "PressZController",
            "reset",
        )
        forward = _class_method(
            "controllers/atomic_actions/pressZ_controller.py",
            "PressZController",
            "forward",
        )
        self.assertIn(
            "self._press_distance = (press_distance if press_distance is not None else 0.05) / get_stage_units()",
            ast.unparse(init),
        )
        self.assertIn(
            "above_position[2] += self._press_distance", ast.unparse(forward)
        )
        self.assertIn(
            "self._press_distance = press_distance / get_stage_units()",
            ast.unparse(reset),
        )

    def test_shake_reset_reconfigures_amplitude_and_next_origin(self):
        forward = _class_method(
            "controllers/atomic_actions/shake_controller.py",
            "ShakeController",
            "forward",
        )
        reset = _class_method(
            "controllers/atomic_actions/shake_controller.py",
            "ShakeController",
            "reset",
        )
        self.assertIsNone(
            ast.literal_eval(_default_for(forward, "gripper_position"))
        )
        forward_source = ast.unparse(forward)
        self.assertIn(
            "self._initial_position = np.asarray(gripper_position).copy()",
            forward_source,
        )
        self.assertLess(
            forward_source.index("if not self._forward_start"),
            forward_source.index("if end_effector_orientation is None"),
        )
        self.assertIn(
            "self._shake_distance = shake_distance / get_stage_units()",
            ast.unparse(reset),
        )
        self.assertIn("self._forward_start = False", ast.unparse(reset))
        event_values = {
            comparator.value
            for node in ast.walk(forward)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "_event"
            for comparator in node.comparators
            if isinstance(comparator, ast.Constant)
            and isinstance(comparator.value, int)
        }
        self.assertTrue(set(range(9)).issubset(event_values))

    def test_open_reset_preserves_no_arg_call_and_reconfigures_furniture(self):
        reset = _class_method(
            "controllers/atomic_actions/open_controller.py",
            "OpenController",
            "reset",
        )
        source = ast.unparse(reset)
        self.assertIsNone(ast.literal_eval(_default_for(reset, "furniture_type")))
        self.assertIn("BaseController.reset(self)", source)
        self.assertIn("self._event = 0", source)
        self.assertIn("self._t = 0", source)
        self.assertIn("self.position_rotation_interp_iter = None", source)
        self.assertIn("if furniture_type is not None", source)
        self.assertIn("self.furniture_type = furniture_type", source)

    def test_close_reset_preserves_no_arg_api_and_clears_door_state(self):
        reset = _class_method(
            "controllers/atomic_actions/close_controller.py",
            "CloseController",
            "reset",
        )
        source = ast.unparse(reset)
        self.assertIsNone(ast.literal_eval(_default_for(reset, "furniture_type")))
        self.assertIn("BaseController.reset(self)", source)
        self.assertIn("if furniture_type is not None", source)
        self.assertIn("self.furniture_type = furniture_type", source)
        self.assertIn("self.position_rotation_interp_iter = None", source)
        self.assertIn("self.init_handle_position = None", source)


if __name__ == "__main__":
    unittest.main()
