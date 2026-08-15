import importlib.util
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class FakeQuaternion:
    def GetReal(self):
        return 1.0

    def GetImaginary(self):
        return np.zeros(3)


class FakeRotationMatrix:
    def __mul__(self, vector):
        return np.asarray(vector, dtype=float)


class FakeMatrix:
    def __init__(self, values=None):
        if isinstance(values, FakeMatrix):
            values = values.values
        self.values = np.array(
            np.eye(4) if values is None else values,
            dtype=float,
            copy=True,
        )

    def GetInverse(self):
        return FakeMatrix(np.linalg.inv(self.values))

    def __mul__(self, other):
        return FakeMatrix(np.matmul(self.values, other.values))

    def TransformAffine(self, position):
        homogeneous = np.append(np.asarray(position, dtype=float), 1.0)
        return np.matmul(homogeneous, self.values)[:3]

    def ExtractTranslation(self):
        return self.values[3, :3].copy()

    def ExtractRotationQuat(self):
        return FakeQuaternion()

    def ExtractRotationMatrix(self):
        return FakeRotationMatrix()

    def SetTranslateOnly(self, position):
        self.values[3, :3] = np.asarray(position, dtype=float)
        return self


class FakeOp:
    def __init__(self, op_type, name, value, *, inverse=False):
        self.op_type = op_type
        self.name = name
        self.value = value
        self.inverse = inverse
        self.set_calls = []

    def GetOpType(self):
        return self.op_type

    def GetOpName(self):
        return self.name

    def IsInverseOp(self):
        return self.inverse

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = value
        self.set_calls.append(value)


class FakeAttribute:
    def __init__(self, value, *, fail_on_set_attempts=()):
        self.value = value
        self.fail_on_set_attempts = set(fail_on_set_attempts)
        self.set_attempts = []
        self.set_calls = []

    def Get(self):
        return self.value

    def Set(self, value):
        self.set_attempts.append(value)
        attempt = len(self.set_attempts)
        if attempt in self.fail_on_set_attempts:
            raise RuntimeError(f"attribute write failed on attempt {attempt}")
        self.value = value
        self.set_calls.append(value)
        return True


class FakePrim:
    def __init__(
        self,
        path,
        *,
        rigid_body_enabled=None,
        kinematic_enabled=False,
        velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        rigid_attribute_failures=None,
        children=(),
        instance_proxy=False,
        visible_children=True,
        xform_ops=(),
        local_matrix=None,
        world_matrix=None,
        parent=None,
    ):
        self.path = path
        self.rigid_body_enabled = rigid_body_enabled
        rigid_attribute_failures = rigid_attribute_failures or {}
        self.rigid_body_enabled_attr = FakeAttribute(rigid_body_enabled)
        self.kinematic_enabled_attr = FakeAttribute(
            kinematic_enabled,
            fail_on_set_attempts=rigid_attribute_failures.get("kinematic", ()),
        )
        self.velocity_attr = FakeAttribute(
            np.asarray(velocity, dtype=float),
            fail_on_set_attempts=rigid_attribute_failures.get("velocity", ()),
        )
        self.angular_velocity_attr = FakeAttribute(
            np.asarray(angular_velocity, dtype=float),
            fail_on_set_attempts=rigid_attribute_failures.get(
                "angular_velocity", ()
            ),
        )
        self.children = list(children)
        self.instance_proxy = instance_proxy
        self.visible_children = visible_children
        self.xform_ops = list(xform_ops)
        self.local_matrix = local_matrix or FakeMatrix()
        self.world_matrix = world_matrix or FakeMatrix()
        self.parent = parent
        self.clear_xform_count = 0
        self.transform_ops = []
        self.reset_xform_stack = False

    def IsValid(self):
        return True

    def GetChildren(self):
        return list(self.children) if self.visible_children else []

    def GetPath(self):
        return self.path

    def HasAPI(self, schema):
        del schema
        return self.rigid_body_enabled is not None

    def IsInstanceProxy(self):
        return self.instance_proxy

    def GetParent(self):
        return self.parent or SimpleNamespace(IsValid=lambda: False)


class FakeRigidBodyAPI:
    def __init__(self, prim):
        self.prim = prim

    def GetRigidBodyEnabledAttr(self):
        return self.prim.rigid_body_enabled_attr

    def GetKinematicEnabledAttr(self):
        return self.prim.kinematic_enabled_attr

    def GetVelocityAttr(self):
        return self.prim.velocity_attr

    def GetAngularVelocityAttr(self):
        return self.prim.angular_velocity_attr


@contextmanager
def loaded_gripper_module(root_prim, *, extra_prims=()):
    class FakeXformOp:
        TypeOrient = 0
        TypeRotateXYZ = 1
        TypeRotateXZY = 2
        TypeRotateYXZ = 3
        TypeRotateYZX = 4
        TypeRotateZXY = 5
        TypeRotateZYX = 6
        TypeTranslate = 7
        TypeScale = 8
        TypeTransform = 9

    class FakeXformable:
        def __init__(self, prim):
            self.prim = prim

        def ComputeLocalToWorldTransform(self, time):
            del time
            return self.prim.world_matrix

        def GetOrderedXformOps(self):
            return list(self.prim.xform_ops)

        def GetLocalTransformation(self):
            return FakeMatrix(self.prim.local_matrix)

        def ClearXformOpOrder(self):
            self.prim.clear_xform_count += 1
            self.prim.xform_ops = []
            self.prim.reset_xform_stack = False

        def AddTranslateOp(self):
            op = FakeOp(
                FakeXformOp.TypeTranslate,
                "xformOp:translate",
                np.zeros(3),
            )
            self.prim.xform_ops.append(op)
            return op

        def AddTransformOp(self):
            op = FakeOp(
                FakeXformOp.TypeTransform,
                "xformOp:transform",
                FakeMatrix(),
            )
            self.prim.xform_ops.append(op)
            self.prim.transform_ops.append(op)
            return op

        def GetResetXformStack(self):
            return self.prim.reset_xform_stack

        def SetResetXformStack(self, value):
            self.prim.reset_xform_stack = value

        def MakeMatrixXform(self):
            self.ClearXformOpOrder()
            return self.AddTransformOp()

    prims = {prim.path: prim for prim in extra_prims}
    stack = [root_prim, *extra_prims]
    while stack:
        prim = stack.pop()
        prims[prim.path] = prim
        stack.extend(prim.children)

    prims_module = ModuleType("omni.isaac.core.utils.prims")
    prims_module.get_prim_at_path = lambda path: prims.get(
        path, SimpleNamespace(IsValid=lambda: False)
    )
    traversal_calls = []

    def traverse_instance_proxies():
        return "instance-proxy-predicate"

    def prim_range(root, predicate=None):
        traversal_calls.append((root, predicate))
        pending = [root]
        result = []
        while pending:
            prim = pending.pop()
            result.append(prim)
            pending.extend(reversed(prim.children))
        return result

    usd_geom = SimpleNamespace(XformOp=FakeXformOp, Xformable=FakeXformable)
    pxr_module = ModuleType("pxr")
    pxr_module.Gf = SimpleNamespace(
        Vec3d=lambda *values: np.asarray(values, dtype=float),
        Vec3f=lambda *values: np.asarray(values, dtype=float),
        Quatf=lambda real, imaginary: FakeQuaternion(),
        Matrix4d=FakeMatrix,
    )
    pxr_module.UsdGeom = usd_geom
    pxr_module.Usd = SimpleNamespace(
        PrimRange=prim_range,
        TraverseInstanceProxies=traverse_instance_proxies,
    )
    pxr_module.Sdf = SimpleNamespace()
    pxr_module.UsdPhysics = SimpleNamespace(RigidBodyAPI=FakeRigidBodyAPI)
    stubs = {
        "omni": ModuleType("omni"),
        "omni.isaac": ModuleType("omni.isaac"),
        "omni.isaac.core": ModuleType("omni.isaac.core"),
        "omni.isaac.core.utils": ModuleType("omni.isaac.core.utils"),
        "omni.isaac.core.utils.prims": prims_module,
        "pxr": pxr_module,
    }
    path = ROOT / "controllers" / "robot_controllers" / "grapper_manager.py"
    module_name = "_protocol1_gripper_manager"
    with patch.dict(sys.modules, stubs, clear=False):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        module._test_traversal_calls = traversal_calls
        module._test_xform_op = FakeXformOp
        yield module
    sys.modules.pop(module_name, None)


class GripperRigidBodyResolutionTests(unittest.TestCase):
    def test_strict_binding_resolves_unique_enabled_rigid_body_below_wrapper(self):
        rigid = FakePrim(
            "/World/Flask/FlaskBody", rigid_body_enabled=True
        )
        wrapper = FakePrim("/World/Flask", children=[rigid])
        with loaded_gripper_module(wrapper) as module:
            gripper = module.Gripper()

            gripper.add_object_to_gripper(
                "/World/Flask",
                "/World/Franka/tool_center",
                resolve_rigid_body=True,
            )

        self.assertEqual(gripper.grasped_object_path, rigid.path)
        self.assertEqual(
            module._test_traversal_calls,
            [(wrapper, "instance-proxy-predicate")],
        )

    def test_strict_binding_rejects_unique_enabled_instance_proxy_atomically(self):
        proxy = FakePrim(
            "/World/Flask/FlaskBody",
            rigid_body_enabled=True,
            instance_proxy=True,
        )
        wrapper = FakePrim(
            "/World/Flask",
            children=[proxy],
            visible_children=False,
        )
        with loaded_gripper_module(wrapper) as module:
            gripper = module.Gripper()

            with self.assertRaisesRegex(
                ValueError, "instance proxy.*cannot be authored"
            ):
                gripper.add_object_to_gripper(
                    wrapper.path,
                    "/World/Franka/tool_center",
                    resolve_rigid_body=True,
                )

        self.assertIsNone(gripper.grasped_object_path)
        self.assertIsNone(gripper.gripper_frame_path)
        self.assertIsNone(gripper.inverse_transform_matrix)
        self.assertIsNone(gripper.position_offest)
        self.assertIsNone(gripper._managed_rigid_body_api)
        self.assertIsNone(gripper._managed_rigid_body_original_kinematic)

    def test_strict_binding_ignores_disabled_rigid_body(self):
        disabled = FakePrim(
            "/World/Flask/DisabledBody", rigid_body_enabled=False
        )
        enabled = FakePrim(
            "/World/Flask/EnabledBody", rigid_body_enabled=True
        )
        wrapper = FakePrim("/World/Flask", children=[disabled, enabled])
        with loaded_gripper_module(wrapper) as module:
            gripper = module.Gripper()

            gripper.add_object_to_gripper(
                wrapper.path,
                "/World/Franka/tool_center",
                resolve_rigid_body=True,
            )

        self.assertEqual(gripper.grasped_object_path, enabled.path)

    def test_strict_binding_fails_closed_without_enabled_rigid_body(self):
        disabled = FakePrim(
            "/World/Flask/DisabledBody", rigid_body_enabled=False
        )
        wrapper = FakePrim("/World/Flask", children=[disabled])
        with loaded_gripper_module(wrapper) as module:
            gripper = module.Gripper()

            with self.assertRaisesRegex(
                ValueError, "requires exactly one enabled rigid body; found 0"
            ):
                gripper.add_object_to_gripper(
                    wrapper.path,
                    "/World/Franka/tool_center",
                    resolve_rigid_body=True,
                )

    def test_strict_binding_fails_closed_for_ambiguous_rigid_bodies(self):
        first = FakePrim("/World/Flask/First", rigid_body_enabled=True)
        second = FakePrim("/World/Flask/Second", rigid_body_enabled=True)
        wrapper = FakePrim("/World/Flask", children=[first, second])
        with loaded_gripper_module(wrapper) as module:
            gripper = module.Gripper()

            with self.assertRaisesRegex(
                ValueError, "requires exactly one enabled rigid body; found 2"
            ):
                gripper.add_object_to_gripper(
                    wrapper.path,
                    "/World/Franka/tool_center",
                    resolve_rigid_body=True,
                )

    def test_legacy_binding_keeps_wrapper_path_by_default(self):
        rigid = FakePrim(
            "/World/Flask/FlaskBody", rigid_body_enabled=True
        )
        wrapper = FakePrim("/World/Flask", children=[rigid])
        with loaded_gripper_module(wrapper) as module:
            gripper = module.Gripper()

            gripper.add_object_to_gripper(
                wrapper.path, "/World/Franka/tool_center"
            )

        self.assertEqual(gripper.grasped_object_path, wrapper.path)

    def test_reset_releases_current_binding(self):
        wrapper = FakePrim("/World/Flask", rigid_body_enabled=True)
        with loaded_gripper_module(wrapper) as module:
            gripper = module.Gripper()
            gripper.add_object_to_gripper(
                wrapper.path, "/World/Franka/tool_center"
            )

            gripper.reset()

        self.assertIsNone(gripper.grasped_object_path)
        self.assertIsNone(gripper.gripper_frame_path)


class GripperManagedRigidBodyLifecycleTests(unittest.TestCase):
    def test_strict_binding_makes_dynamic_body_kinematic_and_release_restores_it(self):
        rigid = FakePrim(
            "/World/Flask/Body",
            rigid_body_enabled=True,
            kinematic_enabled=False,
            velocity=(1.0, -2.0, 3.0),
            angular_velocity=(-4.0, 5.0, -6.0),
        )
        wrapper = FakePrim("/World/Flask", children=[rigid])
        with loaded_gripper_module(wrapper) as module:
            gripper = module.Gripper()

            gripper.add_object_to_gripper(
                wrapper.path,
                "/World/Franka/tool_center",
                resolve_rigid_body=True,
            )

            self.assertTrue(rigid.kinematic_enabled_attr.Get())
            np.testing.assert_allclose(rigid.velocity_attr.Get(), np.zeros(3))
            np.testing.assert_allclose(
                rigid.angular_velocity_attr.Get(), np.zeros(3)
            )

            gripper.release_object()
            write_counts = (
                len(rigid.kinematic_enabled_attr.set_attempts),
                len(rigid.velocity_attr.set_attempts),
                len(rigid.angular_velocity_attr.set_attempts),
            )

            gripper.release_object()

        self.assertFalse(rigid.kinematic_enabled_attr.Get())
        np.testing.assert_allclose(rigid.velocity_attr.Get(), np.zeros(3))
        np.testing.assert_allclose(rigid.angular_velocity_attr.Get(), np.zeros(3))
        self.assertIsNone(gripper.grasped_object_path)
        self.assertIsNone(gripper.gripper_frame_path)
        self.assertEqual(
            (
                len(rigid.kinematic_enabled_attr.set_attempts),
                len(rigid.velocity_attr.set_attempts),
                len(rigid.angular_velocity_attr.set_attempts),
            ),
            write_counts,
        )

    def test_strict_binding_release_preserves_original_kinematic_true(self):
        rigid = FakePrim(
            "/World/Flask", rigid_body_enabled=True, kinematic_enabled=True
        )
        with loaded_gripper_module(rigid) as module:
            gripper = module.Gripper()
            gripper.add_object_to_gripper(
                rigid.path,
                "/World/Franka/tool_center",
                resolve_rigid_body=True,
            )

            gripper.release_object()

        self.assertTrue(rigid.kinematic_enabled_attr.Get())
        self.assertEqual(rigid.kinematic_enabled_attr.set_calls, [True, True])

    def test_legacy_binding_never_writes_rigid_body_attributes(self):
        rigid = FakePrim(
            "/World/Flask",
            rigid_body_enabled=True,
            velocity=(1.0, 2.0, 3.0),
            angular_velocity=(4.0, 5.0, 6.0),
        )
        with loaded_gripper_module(rigid) as module:
            gripper = module.Gripper()

            gripper.add_object_to_gripper(
                rigid.path, "/World/Franka/tool_center"
            )
            gripper.release_object()

        self.assertEqual(rigid.kinematic_enabled_attr.set_attempts, [])
        self.assertEqual(rigid.velocity_attr.set_attempts, [])
        self.assertEqual(rigid.angular_velocity_attr.set_attempts, [])

    def test_legacy_binding_releases_managed_rigid_body_before_replacement(self):
        managed = FakePrim(
            "/World/ManagedFlask",
            rigid_body_enabled=True,
            kinematic_enabled=False,
            velocity=(1.0, 2.0, 3.0),
            angular_velocity=(4.0, 5.0, 6.0),
        )
        legacy = FakePrim("/World/LegacyBeaker")
        with loaded_gripper_module(managed, extra_prims=[legacy]) as module:
            gripper = module.Gripper()
            gripper.add_object_to_gripper(
                managed.path,
                "/World/Franka/tool_center",
                resolve_rigid_body=True,
            )

            gripper.add_object_to_gripper(
                legacy.path, "/World/Franka/tool_center"
            )

        self.assertFalse(managed.kinematic_enabled_attr.Get())
        np.testing.assert_allclose(managed.velocity_attr.Get(), np.zeros(3))
        np.testing.assert_allclose(
            managed.angular_velocity_attr.Get(), np.zeros(3)
        )
        self.assertEqual(gripper.grasped_object_path, legacy.path)
        self.assertEqual(
            gripper.gripper_frame_path, "/World/Franka/tool_center"
        )
        self.assertIsNone(gripper._managed_rigid_body_api)
        self.assertIsNone(gripper._managed_rigid_body_original_kinematic)

    def test_legacy_binding_does_not_replace_managed_body_when_release_fails(self):
        managed = FakePrim(
            "/World/ManagedFlask",
            rigid_body_enabled=True,
            kinematic_enabled=False,
            rigid_attribute_failures={"velocity": {2}},
        )
        legacy = FakePrim("/World/LegacyBeaker")
        with loaded_gripper_module(managed, extra_prims=[legacy]) as module:
            gripper = module.Gripper()
            gripper.add_object_to_gripper(
                managed.path,
                "/World/Franka/tool_center",
                resolve_rigid_body=True,
            )

            with self.assertRaisesRegex(RuntimeError, "attribute write failed"):
                gripper.add_object_to_gripper(
                    legacy.path, "/World/Franka/tool_center"
                )

        self.assertFalse(managed.kinematic_enabled_attr.Get())
        self.assertIsNone(gripper.grasped_object_path)
        self.assertIsNone(gripper.gripper_frame_path)
        self.assertIsNone(gripper.inverse_transform_matrix)
        self.assertIsNone(gripper._managed_rigid_body_api)
        self.assertIsNone(gripper._managed_rigid_body_original_kinematic)

    def test_legacy_binding_still_replaces_another_legacy_binding_directly(self):
        first = FakePrim("/World/FirstLegacy")
        second = FakePrim("/World/SecondLegacy")
        with loaded_gripper_module(first, extra_prims=[second]) as module:
            gripper = module.Gripper()
            gripper.add_object_to_gripper(
                first.path, "/World/Franka/tool_center"
            )

            gripper.add_object_to_gripper(
                second.path, "/World/Franka/tool_center"
            )

        self.assertEqual(gripper.grasped_object_path, second.path)
        self.assertEqual(first.kinematic_enabled_attr.set_attempts, [])
        self.assertEqual(second.kinematic_enabled_attr.set_attempts, [])

    def test_strict_binding_failure_rolls_back_physics_and_binding_state(self):
        original_velocity = np.array([1.0, 2.0, 3.0])
        original_angular_velocity = np.array([4.0, 5.0, 6.0])
        rigid = FakePrim(
            "/World/Flask",
            rigid_body_enabled=True,
            kinematic_enabled=False,
            velocity=original_velocity,
            angular_velocity=original_angular_velocity,
            rigid_attribute_failures={"angular_velocity": {1}},
        )
        with loaded_gripper_module(rigid) as module:
            gripper = module.Gripper()

            with self.assertRaisesRegex(RuntimeError, "attribute write failed"):
                gripper.add_object_to_gripper(
                    rigid.path,
                    "/World/Franka/tool_center",
                    resolve_rigid_body=True,
                )

        self.assertFalse(rigid.kinematic_enabled_attr.Get())
        np.testing.assert_allclose(rigid.velocity_attr.Get(), original_velocity)
        np.testing.assert_allclose(
            rigid.angular_velocity_attr.Get(), original_angular_velocity
        )
        self.assertIsNone(gripper.grasped_object_path)
        self.assertIsNone(gripper.gripper_frame_path)
        self.assertIsNone(gripper.inverse_transform_matrix)
        self.assertIsNone(gripper.position_offest)
        self.assertIsNone(gripper._managed_rigid_body_api)
        self.assertIsNone(gripper._managed_rigid_body_original_kinematic)

    def test_release_failure_restores_kinematic_and_always_clears_state(self):
        rigid = FakePrim(
            "/World/Flask",
            rigid_body_enabled=True,
            kinematic_enabled=False,
            rigid_attribute_failures={"velocity": {2}},
        )
        with loaded_gripper_module(rigid) as module:
            gripper = module.Gripper()
            gripper.add_object_to_gripper(
                rigid.path,
                "/World/Franka/tool_center",
                resolve_rigid_body=True,
            )

            with self.assertRaisesRegex(RuntimeError, "attribute write failed"):
                gripper.release_object()

            gripper.release_object()

        self.assertFalse(rigid.kinematic_enabled_attr.Get())
        self.assertEqual(len(rigid.angular_velocity_attr.set_attempts), 2)
        self.assertIsNone(gripper.grasped_object_path)
        self.assertIsNone(gripper.gripper_frame_path)
        self.assertIsNone(gripper.inverse_transform_matrix)
        self.assertIsNone(gripper.position_offest)
        self.assertIsNone(gripper._managed_rigid_body_api)
        self.assertIsNone(gripper._managed_rigid_body_original_kinematic)


class GripperTranslationUpdateTests(unittest.TestCase):
    def _update_fixture(self, xform_ops, *, local_matrix=None):
        object_prim = FakePrim(
            "/World/Flask",
            xform_ops=xform_ops,
            local_matrix=local_matrix,
        )
        gripper_frame = FakePrim(
            "/World/Franka/tool_center",
            world_matrix=FakeMatrix(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [4.0, 5.0, 6.0, 1.0],
                ]
            ),
        )
        return object_prim, gripper_frame

    def test_update_normalizes_rotate_or_scale_before_translate(self):
        cases = (
            (
                "rotate_z_90",
                FakeOp(1, "xformOp:rotateZ", 90.0),
                FakeMatrix(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [-2.0, 1.0, 3.0, 1.0],
                    ]
                ),
            ),
            (
                "scale",
                FakeOp(8, "xformOp:scale", np.array([2.0, 3.0, 4.0])),
                FakeMatrix(
                    [
                        [2.0, 0.0, 0.0, 0.0],
                        [0.0, 3.0, 0.0, 0.0],
                        [0.0, 0.0, 4.0, 0.0],
                        [2.0, 6.0, 12.0, 1.0],
                    ]
                ),
            ),
        )
        for label, leading_op, local_matrix in cases:
            with self.subTest(label=label):
                translate_op = FakeOp(
                    7, "xformOp:translate", np.array([1.0, 2.0, 3.0])
                )
                object_prim, frame = self._update_fixture(
                    [leading_op, translate_op], local_matrix=local_matrix
                )
                with loaded_gripper_module(
                    object_prim, extra_prims=[frame]
                ) as module:
                    gripper = module.Gripper()
                    gripper.add_object_to_gripper(
                        object_prim.path, frame.path
                    )
                    gripper.position_offest = np.zeros(3)

                    gripper.update_grasped_object_position()

                self.assertEqual(leading_op.set_calls, [])
                self.assertEqual(translate_op.set_calls, [])
                self.assertEqual(object_prim.clear_xform_count, 1)
                matrix = object_prim.transform_ops[0].set_calls[-1]
                np.testing.assert_allclose(
                    matrix.values[:3, :3], local_matrix.values[:3, :3]
                )
                np.testing.assert_allclose(
                    matrix.ExtractTranslation(), [4.0, 5.0, 6.0]
                )

    def test_update_keeps_direct_path_for_translate_first_stack(self):
        translate_op = FakeOp(
            7, "xformOp:translate", np.array([1.0, 2.0, 3.0])
        )
        object_prim, frame = self._update_fixture([translate_op])
        with loaded_gripper_module(
            object_prim, extra_prims=[frame]
        ) as module:
            gripper = module.Gripper()
            gripper.add_object_to_gripper(object_prim.path, frame.path)
            gripper.position_offest = np.array([0.5, 1.0, 1.5])

            gripper.update_grasped_object_position()

        self.assertEqual(object_prim.clear_xform_count, 0)
        self.assertEqual(object_prim.transform_ops, [])
        self.assertEqual(len(translate_op.set_calls), 1)
        np.testing.assert_allclose(
            translate_op.set_calls[0], [4.5, 6.0, 7.5]
        )

    def test_update_failure_clears_binding_state_atomically(self):
        object_prim = FakePrim("/World/Flask")
        with loaded_gripper_module(object_prim) as module:
            gripper = module.Gripper()
            gripper.add_object_to_gripper(
                object_prim.path, "/World/MissingFrame"
            )

            with self.assertRaisesRegex(ValueError, "Gripper frame"):
                gripper.update_grasped_object_position()

        self.assertIsNone(gripper.grasped_object_path)
        self.assertIsNone(gripper.gripper_frame_path)
        self.assertIsNone(gripper.position_offest)
        self.assertIsNone(gripper.inverse_transform_matrix)

    def test_update_without_translate_normalizes_matrix_and_preserves_linear_part(self):
        orient_op = FakeOp(1, "xformOp:rotateXYZ", np.array([0.0, 0.0, 30.0]))
        scale_op = FakeOp(8, "xformOp:scale", np.array([2.0, 3.0, 4.0]))
        local_matrix = FakeMatrix(
            [
                [0.0, 2.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [1.0, 2.0, 3.0, 1.0],
            ]
        )
        object_prim, frame = self._update_fixture(
            [orient_op, scale_op], local_matrix=local_matrix
        )
        object_prim.reset_xform_stack = True
        with loaded_gripper_module(
            object_prim, extra_prims=[frame]
        ) as module:
            gripper = module.Gripper()
            gripper.add_object_to_gripper(object_prim.path, frame.path)
            gripper.position_offest = np.array([0.5, 1.0, 1.5])

            gripper.update_grasped_object_position()

        self.assertEqual(orient_op.set_calls, [])
        self.assertEqual(scale_op.set_calls, [])
        self.assertEqual(object_prim.clear_xform_count, 1)
        self.assertEqual(len(object_prim.transform_ops), 1)
        self.assertTrue(object_prim.reset_xform_stack)
        matrix = object_prim.transform_ops[0].set_calls[-1]
        np.testing.assert_allclose(
            matrix.values[:3, :3], local_matrix.values[:3, :3]
        )
        np.testing.assert_allclose(matrix.ExtractTranslation(), [4.5, 6.0, 7.5])


class GripperPoseUpdateTests(unittest.TestCase):
    def test_pose_update_replaces_composite_ops_with_exact_local_matrix(self):
        translate_op = FakeOp(7, "xformOp:translate", np.array([1.0, 2.0, 3.0]))
        rotate_op = FakeOp(1, "xformOp:rotateXYZ", np.array([10.0, 20.0, 30.0]))
        scale_op = FakeOp(8, "xformOp:scale", np.array([2.0, 3.0, 4.0]))
        parent = FakePrim(
            "/World/Flask",
            world_matrix=FakeMatrix(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.5, -0.25, 0.1, 1.0],
                ]
            ),
        )
        object_prim = FakePrim(
            "/World/Flask/Body",
            xform_ops=[translate_op, rotate_op, scale_op],
            world_matrix=FakeMatrix(
                [
                    [0.0, 2.0, 0.0, 0.0],
                    [-3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 0.0],
                    [1.0, 2.0, 3.0, 1.0],
                ]
            ),
            parent=parent,
        )
        parent.children.append(object_prim)
        initial_gripper = FakeMatrix(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.2, 0.3, 0.4, 1.0],
            ]
        )
        gripper_frame = FakePrim(
            "/World/Franka/tool_center", world_matrix=initial_gripper
        )
        with loaded_gripper_module(parent, extra_prims=[gripper_frame]) as module:
            gripper = module.Gripper()
            gripper.init_pose_tracking(object_prim.path, gripper_frame.path)
            gripper_frame.world_matrix = FakeMatrix(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.8, 0.9, 1.0, 1.0],
                ]
            )
            expected = (
                object_prim.world_matrix
                * initial_gripper.GetInverse()
                * gripper_frame.world_matrix
                * parent.world_matrix.GetInverse()
            )

            gripper.update_grasped_object_pose()

        self.assertEqual(translate_op.set_calls, [])
        self.assertEqual(rotate_op.set_calls, [])
        self.assertEqual(scale_op.set_calls, [])
        self.assertEqual(object_prim.clear_xform_count, 1)
        self.assertEqual(len(object_prim.transform_ops), 1)
        matrix = object_prim.transform_ops[0].set_calls[-1]
        np.testing.assert_allclose(matrix.values, expected.values)


if __name__ == "__main__":
    unittest.main()
