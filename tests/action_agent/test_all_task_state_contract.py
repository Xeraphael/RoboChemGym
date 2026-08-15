import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
STATE_CONTRACT_PATH = (
    ROOT / "agent" / "action" / "plan_execution" / "state_contract.py"
)
ALL_TASK_PATH = ROOT / "tasks" / "all_task.py"


def load_state_contract_module():
    spec = importlib.util.spec_from_file_location(
        "_task11_state_contract", STATE_CONTRACT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_all_task_module():
    base_task_module = ModuleType("tasks.base_task")

    class StubBaseTask:
        pass

    base_task_module.BaseTask = StubBaseTask
    pxr_module = ModuleType("pxr")
    pxr_module.Usd = FakeUsd
    module_name = "tasks._task11_all_task"
    with patch.dict(
        sys.modules,
        {"tasks.base_task": base_task_module, "pxr": pxr_module},
        clear=False,
    ):
        spec = importlib.util.spec_from_file_location(module_name, ALL_TASK_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


class FakePrim:
    def __init__(self, valid):
        self.valid = valid

    def IsValid(self):
        return self.valid


class FakePath:
    def __init__(self, path):
        self.pathString = path


class TreePrim(FakePrim):
    def __init__(self, path, *, valid=True, type_name="Xform"):
        super().__init__(valid)
        self.path = path
        self.type_name = type_name
        self.subtree = [self]

    def __bool__(self):
        return self.valid

    def GetName(self):
        return self.path.rsplit("/", 1)[-1]

    def GetPath(self):
        return FakePath(self.path)

    def GetTypeName(self):
        return self.type_name


class FakeUsd:
    proxy_predicate = object()

    @staticmethod
    def TraverseInstanceProxies():
        return FakeUsd.proxy_predicate

    @staticmethod
    def PrimRange(prim, predicate):
        if predicate is not FakeUsd.proxy_predicate:
            raise AssertionError("instance proxy traversal is required")
        return prim.subtree


class TreeStage:
    def __init__(self, prims):
        self.prims = {prim.path: prim for prim in prims}
        self.requests = []

    def GetPrimAtPath(self, path):
        self.requests.append(path)
        return self.prims.get(path, TreePrim(path, valid=False))


class FakeStage:
    def __init__(self, valid_paths):
        self.valid_paths = set(valid_paths)
        self.requests = []

    def GetPrimAtPath(self, path):
        self.requests.append(path)
        return FakePrim(path in self.valid_paths)


class FakeObjectUtils:
    def __init__(self):
        self.geometry_calls = []
        self.xform_calls = []
        self.size_calls = []
        self.joint_calls = []

    def get_geometry_center(self, *, object_path):
        self.geometry_calls.append(object_path)
        if object_path == "/World/Beaker":
            raise RuntimeError("geometry center unavailable")
        return np.array([len(self.geometry_calls), 0.0, 1.0])

    def get_object_xform_position(self, *, object_path):
        self.xform_calls.append(object_path)
        values = {
            "/World/Beaker": np.array([3.0, 0.0, 1.0]),
            "/World/DoorOne/handle": np.array([1.0, 1.0, 1.0]),
            "/World/DoorTwo/handle": np.array([2.0, 2.0, 2.0]),
            "/World/Beaker/grisp_position": np.array([3.0, 3.0, 3.0]),
        }
        if object_path == "/World/Beaker/exploding":
            raise RuntimeError("broken anchor")
        return values.get(object_path)

    def get_object_size(self, *, object_path):
        self.size_calls.append(object_path)
        return np.array([0.1, 0.2, 0.3])

    def get_revolute_joint_positions(self, *, joint_path):
        self.joint_calls.append(joint_path)
        values = {
            "/World/DoorOne/RevoluteJoint": np.array([1.0, 0.0, 0.0]),
            "/World/DoorTwo/RevoluteJoint": np.array([2.0, 0.0, 0.0]),
        }
        return values[joint_path]


class FakeRobot:
    def get_joint_positions(self):
        return np.arange(9, dtype=float)

    def get_gripper_position(self):
        return np.array([0.4, 0.5, 0.6])


class AllTaskStateContractTests(unittest.TestCase):
    def test_state_contract_module_exists(self):
        self.assertTrue(
            STATE_CONTRACT_PATH.is_file(),
            "state_contract.py must provide the pure anchor naming contract",
        )

    @unittest.skipUnless(STATE_CONTRACT_PATH.is_file(), "state contract not implemented")
    def test_anchor_names_map_to_stable_state_suffixes(self):
        anchor_state_suffix = load_state_contract_module().anchor_state_suffix

        self.assertEqual(anchor_state_suffix("grisp_position"), "grisp_position")
        self.assertEqual(anchor_state_suffix("handle"), "handle_position")
        self.assertEqual(
            anchor_state_suffix("RevoluteJoint"), "revolute_joint_position"
        )

    @unittest.skipUnless(STATE_CONTRACT_PATH.is_file(), "state contract not implemented")
    def test_anchor_suffix_rejects_non_strings_deterministically(self):
        anchor_state_suffix = load_state_contract_module().anchor_state_suffix

        for value in (None, 7, ["handle"]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "anchor must be a string"):
                    anchor_state_suffix(value)

    def test_all_task_collects_only_configured_valid_anchors(self):
        module = load_all_task_module()
        task = module.AllTask.__new__(module.AllTask)
        door_one = TreePrim("/World/DoorOne")
        door_one_handle = TreePrim("/World/DoorOne/handle")
        door_one_joint = TreePrim(
            "/World/DoorOne/RevoluteJoint",
            type_name="PhysicsRevoluteJoint",
        )
        door_one.subtree = [door_one, door_one_handle, door_one_joint]
        door_two = TreePrim("/World/DoorTwo")
        door_two_handle = TreePrim("/World/DoorTwo/handle")
        door_two_joint = TreePrim(
            "/World/DoorTwo/RevoluteJoint",
            type_name="PhysicsRevoluteJoint",
        )
        door_two.subtree = [door_two, door_two_handle, door_two_joint]
        beaker = TreePrim("/World/Beaker")
        beaker_grisp = TreePrim("/World/Beaker/grisp_position")
        beaker_exploding = TreePrim("/World/Beaker/exploding")
        beaker.subtree = [beaker, beaker_grisp, beaker_exploding]
        unconfigured = TreePrim("/World/Unconfigured")
        stage = TreeStage(
            [
                *door_one.subtree,
                *door_two.subtree,
                *beaker.subtree,
                unconfigured,
            ]
        )
        object_utils = FakeObjectUtils()
        task.cfg = SimpleNamespace(
            task=SimpleNamespace(max_steps=100),
            agent=SimpleNamespace(
                state_anchors={
                    "DoorOne": ["handle", "RevoluteJoint", "missing"],
                    "DoorTwo": ["handle", "RevoluteJoint"],
                    "Beaker": ["grisp_position", "exploding"],
                }
            ),
        )
        task.object_paths = [
            "/World/DoorOne",
            "/World/DoorTwo",
            "/World/Beaker",
            "/World/Unconfigured",
        ]
        task.object_utils = object_utils
        task.stage = stage
        task.robot = FakeRobot()
        task.frame_idx = 4
        task.reset_needed = False
        task.get_camera_data = lambda: ({"front_rgb": "image"}, {"front": "display"})
        task.on_task_complete = lambda success: self.fail(
            "max-step completion must not run in this fixture"
        )

        state = task.step()

        np.testing.assert_array_equal(state["DoorOne_position"], [1.0, 0.0, 1.0])
        np.testing.assert_array_equal(state["DoorTwo_position"], [2.0, 0.0, 1.0])
        np.testing.assert_array_equal(state["Beaker_position"], [3.0, 0.0, 1.0])
        np.testing.assert_array_equal(state["DoorOne_size"], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(state["DoorOne_handle_position"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(state["DoorTwo_handle_position"], [2.0, 2.0, 2.0])
        np.testing.assert_array_equal(
            state["DoorOne_revolute_joint_position"], [1.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(
            state["DoorTwo_revolute_joint_position"], [2.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(
            state["Beaker_grisp_position"], [3.0, 3.0, 3.0]
        )

        self.assertNotIn("DoorOne_missing_position", state)
        self.assertNotIn("Beaker_exploding_position", state)
        self.assertNotIn("Unconfigured_grisp_position", state)
        self.assertNotIn("revolute_joint_position", state)
        self.assertNotIn("DryingBox_revolute_joint_position", state)
        self.assertNotIn("DryingBox_handle_position", state)
        self.assertEqual(
            stage.requests,
            [
                "/World/DoorOne",
                "/World/DoorTwo",
                "/World/Beaker",
                "/World/Unconfigured",
            ],
        )
        self.assertEqual(
            object_utils.joint_calls,
            [
                "/World/DoorOne/RevoluteJoint",
                "/World/DoorTwo/RevoluteJoint",
            ],
        )

    def test_missing_agent_metadata_preserves_legacy_observation_contract(self):
        module = load_all_task_module()
        task = module.AllTask.__new__(module.AllTask)
        valid_paths = {
            "/World/Beaker/grisp_position",
            "/World/Beaker/press_position",
        }
        stage = FakeStage(valid_paths)

        class LegacyObjectUtils(FakeObjectUtils):
            def get_object_xform_position(self, *, object_path):
                self.xform_calls.append(object_path)
                values = {
                    "/World/Beaker/grisp_position": np.array([1.0, 2.0, 3.0]),
                    "/World/Beaker/press_position": np.array([4.0, 5.0, 6.0]),
                }
                return values.get(object_path)

            def get_revolute_joint_positions(self, *, joint_path):
                self.joint_calls.append(joint_path)
                return np.array([7.0, 8.0, 9.0])

        object_utils = LegacyObjectUtils()
        task.cfg = SimpleNamespace(task=SimpleNamespace(max_steps=100))
        task.object_paths = ["/World/Beaker", "/World/handle"]
        task.object_utils = object_utils
        task.stage = stage
        task.robot = FakeRobot()
        task.frame_idx = 4
        task.reset_needed = False
        task.get_camera_data = lambda: ({}, {})
        task.on_task_complete = lambda success: None

        state = task.step()

        np.testing.assert_array_equal(
            state["Beaker_grisp_position"], [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(
            state["Beaker_press_position"], [4.0, 5.0, 6.0]
        )
        np.testing.assert_array_equal(
            state["DryingBox_handle_position"], state["handle_position"]
        )
        np.testing.assert_array_equal(
            state["revolute_joint_position"], [7.0, 8.0, 9.0]
        )
        np.testing.assert_array_equal(
            state["DryingBox_revolute_joint_position"], [7.0, 8.0, 9.0]
        )

    def test_null_omegaconf_anchor_metadata_uses_legacy_fallback(self):
        module = load_all_task_module()
        cfg = OmegaConf.create({"agent": {"state_anchors": None}})

        self.assertEqual(module._state_anchor_config(cfg), (False, {}))

    def test_explicit_anchor_metadata_uses_preflight_resolution_and_prim_type(self):
        module = load_all_task_module()
        task = module.AllTask.__new__(module.AllTask)
        instance = TreePrim("/World/Device")
        direct = TreePrim("/World/Device/direct_anchor")
        nested_handle = TreePrim("/World/Device/assembly/handle")
        nested_press = TreePrim("/World/Device/buttons/press_position")
        ambiguous_one = TreePrim("/World/Device/left/ambiguous")
        ambiguous_two = TreePrim("/World/Device/right/ambiguous")
        typed_joint = TreePrim(
            "/World/Device/joints/joint_alias",
            type_name="PhysicsRevoluteJoint",
        )
        named_joint_xform = TreePrim(
            "/World/Device/RevoluteJoint",
            type_name="Xform",
        )
        instance.subtree = [
            instance,
            direct,
            nested_handle,
            nested_press,
            ambiguous_one,
            ambiguous_two,
            typed_joint,
            named_joint_xform,
        ]
        stage = TreeStage(instance.subtree)

        class TypedObjectUtils(FakeObjectUtils):
            def get_object_xform_position(self, *, object_path):
                self.xform_calls.append(object_path)
                return np.array([float(len(self.xform_calls)), 0.0, 0.0])

            def get_revolute_joint_positions(self, *, joint_path):
                self.joint_calls.append(joint_path)
                return np.array([9.0, 9.0, 9.0])

        object_utils = TypedObjectUtils()
        task.cfg = SimpleNamespace(
            task=SimpleNamespace(max_steps=100),
            agent=SimpleNamespace(
                state_anchors={
                    "Device": [
                        "direct_anchor",
                        "assembly/handle",
                        "press_position",
                        "ambiguous",
                        "missing",
                        "joint_alias",
                        "RevoluteJoint",
                    ]
                }
            ),
        )
        task.object_paths = ["/World/Device"]
        task.object_utils = object_utils
        task.stage = stage
        task.robot = FakeRobot()
        task.frame_idx = 4
        task.reset_needed = False
        task.get_camera_data = lambda: ({}, {})
        task.on_task_complete = lambda success: None

        state = task.step()

        self.assertIn("Device_direct_anchor_position", state)
        self.assertIn("Device_handle_position", state)
        self.assertIn("Device_press_position", state)
        self.assertIn("Device_joint_alias_position", state)
        self.assertIn("Device_revolute_joint_position", state)
        self.assertFalse(any("/" in key for key in state))
        self.assertNotIn("Device_ambiguous_position", state)
        self.assertNotIn("Device_missing_position", state)
        self.assertEqual(
            object_utils.joint_calls,
            ["/World/Device/joints/joint_alias"],
        )
        self.assertIn(
            "/World/Device/RevoluteJoint", object_utils.xform_calls
        )

    def test_explicit_empty_anchor_metadata_does_not_enable_legacy_fallback(self):
        module = load_all_task_module()
        task = module.AllTask.__new__(module.AllTask)
        stage = FakeStage({"/World/Beaker/grisp_position"})
        object_utils = FakeObjectUtils()
        task.cfg = SimpleNamespace(
            task=SimpleNamespace(max_steps=100),
            agent=SimpleNamespace(state_anchors={}),
        )
        task.object_paths = ["/World/Beaker"]
        task.object_utils = object_utils
        task.stage = stage
        task.robot = FakeRobot()
        task.frame_idx = 4
        task.reset_needed = False
        task.get_camera_data = lambda: ({}, {})
        task.on_task_complete = lambda success: None

        state = task.step()

        self.assertNotIn("Beaker_grisp_position", state)
        self.assertEqual(object_utils.joint_calls, [])

    def test_all_task_source_keeps_legacy_logic_out_of_configured_anchor_branch(self):
        source = ALL_TASK_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertNotIn("common_child_nodes", source)
        self.assertFalse(
            any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                for function in ast.walk(tree)
                if isinstance(function, ast.FunctionDef) and function.name == "step"
                for node in ast.walk(function)
            )
        )


if __name__ == "__main__":
    unittest.main()
