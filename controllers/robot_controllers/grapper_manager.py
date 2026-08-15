from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, UsdGeom, Usd, Sdf, UsdPhysics



class Gripper:
    def __init__(self):
        self.grasped_object_path = None
        self.gripper_frame_path = None
        self.position_offest = None
        self.inverse_transform_matrix = None
        self._managed_rigid_body_api = None
        self._managed_rigid_body_original_kinematic = None
        # 6DoF 跟随用的状态（仅 pour 阶段使用）
        self._relative_mat      = None
        self._pose_obj_path     = None
        self._pose_gripper_path = None

    # ------------------------------------------------------------------
    # 夹取 / 释放
    # ------------------------------------------------------------------
    def reset(self):
        self.release_object()

    def add_object_to_gripper(
        self,
        object_path,
        gripper_frame_path,
        *,
        resolve_rigid_body=False,
    ):
        if resolve_rigid_body:
            self.release_object()
            rigid_body_api = None
            original_motion_state = None
            try:
                transform_prim = get_prim_at_path(object_path)
                if not transform_prim.IsValid():
                    raise ValueError(
                        f"Object at path '{object_path}' is not valid for "
                        "rigid-body resolution."
                    )
                transform_prim = self._resolve_unique_enabled_rigid_body(
                    transform_prim, object_path
                )
                inverse_transform_matrix = (
                    UsdGeom.Xformable(transform_prim)
                    .ComputeLocalToWorldTransform(0)
                    .GetInverse()
                )
                rigid_body_api = UsdPhysics.RigidBodyAPI(transform_prim)
                original_motion_state = (
                    bool(rigid_body_api.GetKinematicEnabledAttr().Get()),
                    rigid_body_api.GetVelocityAttr().Get(),
                    rigid_body_api.GetAngularVelocityAttr().Get(),
                )
                zero_velocity = Gf.Vec3f(0.0, 0.0, 0.0)
                self._set_physics_attribute(
                    rigid_body_api.GetKinematicEnabledAttr(),
                    True,
                    "kinematicEnabled",
                )
                self._set_physics_attribute(
                    rigid_body_api.GetVelocityAttr(),
                    zero_velocity,
                    "velocity",
                )
                self._set_physics_attribute(
                    rigid_body_api.GetAngularVelocityAttr(),
                    zero_velocity,
                    "angularVelocity",
                )
            except Exception as bind_error:
                rollback_error = None
                try:
                    if (
                        rigid_body_api is not None
                        and original_motion_state is not None
                    ):
                        self._write_rigid_body_motion_state(
                            rigid_body_api, *original_motion_state
                        )
                except Exception as exc:
                    rollback_error = exc
                finally:
                    self._clear_binding_state()
                if rollback_error is not None:
                    raise RuntimeError(
                        "Rigid-body binding failed and its physics state could "
                        "not be rolled back."
                    ) from rollback_error
                raise

            self.grasped_object_path = str(transform_prim.GetPath())
            self.gripper_frame_path = gripper_frame_path
            self.inverse_transform_matrix = inverse_transform_matrix
            self._managed_rigid_body_api = rigid_body_api
            self._managed_rigid_body_original_kinematic = original_motion_state[0]
            return

        if self._managed_rigid_body_api is not None:
            self.release_object()

        self.grasped_object_path = object_path
        self.gripper_frame_path = gripper_frame_path

        transform_prim = get_prim_at_path(object_path)
        if not transform_prim.IsValid():
            print(f"[Gripper] Warning: Object at path '{object_path}' is not valid, skipping gripper binding.")
            self.grasped_object_path = None
            return

        self.inverse_transform_matrix = UsdGeom.Xformable(transform_prim).ComputeLocalToWorldTransform(0).GetInverse()

    def _resolve_unique_enabled_rigid_body(self, root_prim, object_path):
        enabled_rigid_bodies = []
        for prim in Usd.PrimRange(
            root_prim, Usd.TraverseInstanceProxies()
        ):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                enabled = (
                    UsdPhysics.RigidBodyAPI(prim)
                    .GetRigidBodyEnabledAttr()
                    .Get()
                )
                if bool(enabled):
                    enabled_rigid_bodies.append(prim)

        if len(enabled_rigid_bodies) != 1:
            raise ValueError(
                f"Rigid-body resolution for '{object_path}' requires exactly "
                f"one enabled rigid body; found {len(enabled_rigid_bodies)}."
            )
        rigid_body = enabled_rigid_bodies[0]
        if rigid_body.IsInstanceProxy():
            raise ValueError(
                f"Rigid-body resolution for '{object_path}' selected instance "
                f"proxy '{rigid_body.GetPath()}', which cannot be authored."
            )
        return rigid_body

    @staticmethod
    def _set_physics_attribute(attribute, value, name):
        if not attribute.Set(value):
            raise RuntimeError(f"Failed to author rigid-body {name} attribute.")

    def _write_rigid_body_motion_state(
        self,
        rigid_body_api,
        kinematic_enabled,
        velocity,
        angular_velocity,
    ):
        first_error = None
        for attribute, value, name in (
            (
                rigid_body_api.GetVelocityAttr(),
                velocity,
                "velocity",
            ),
            (
                rigid_body_api.GetAngularVelocityAttr(),
                angular_velocity,
                "angularVelocity",
            ),
            (
                rigid_body_api.GetKinematicEnabledAttr(),
                kinematic_enabled,
                "kinematicEnabled",
            ),
        ):
            try:
                self._set_physics_attribute(attribute, value, name)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    # ------------------------------------------------------------------
    # pick/place 阶段：仅平移跟随
    # ------------------------------------------------------------------
    def begin_position_tracking(self):
        """Rebase translation tracking without releasing the grasped object."""
        self.position_offest = None
        self._relative_mat = None
        self._pose_obj_path = None
        self._pose_gripper_path = None

    def update_grasped_object_position(self):
        if not self.grasped_object_path or not self.gripper_frame_path:
            return

        try:
            self._update_grasped_object_position()
        except Exception:
            self.release_object()
            raise

    def _update_grasped_object_position(self):

        target_frame_prim = get_prim_at_path(self.gripper_frame_path)
        if not target_frame_prim.IsValid():
            raise ValueError(f"Gripper frame at path {self.gripper_frame_path} is not valid.")

        # 夹爪世界坐标
        gripper_world_mat = UsdGeom.Xformable(target_frame_prim).ComputeLocalToWorldTransform(0)
        gripper_world_pos = gripper_world_mat.ExtractTranslation()
        gripper_world_quat = gripper_world_mat.ExtractRotationQuat()

        object_prim = get_prim_at_path(self.grasped_object_path)
        if not object_prim.IsValid():
            raise ValueError(f"Object at path {self.grasped_object_path} is not valid.")

        # 将世界坐标转换到物体父节点的局部坐标系
        parent_prim = object_prim.GetParent()
        if parent_prim and parent_prim.IsValid():
            parent_world_mat = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(0)
            local_position = parent_world_mat.GetInverse().TransformAffine(gripper_world_pos)
        else:
            local_position = gripper_world_pos

        xformable = UsdGeom.Xformable(object_prim)
        xform_ops = xformable.GetOrderedXformOps()
        translate_op = next(
            (
                op
                for op in xform_ops
                if not op.IsInverseOp()
                and op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                and "pivot" not in str(op.GetOpName())
            ),
            None,
        )
        direct_translate = (
            translate_op is not None
            and bool(xform_ops)
            and translate_op is xform_ops[0]
        )

        if self.position_offest is None:
            if direct_translate:
                current_translate = translate_op.Get()
            else:
                local_transform = xformable.GetLocalTransformation()
                if isinstance(local_transform, tuple):
                    local_transform = local_transform[0]
                current_translate = local_transform.ExtractTranslation()
            self.position_offest = Gf.Vec3d(
                *[
                    float(current_translate[index] - local_position[index])
                    for index in range(3)
                ]
            )

        target_local_position = Gf.Vec3d(
            *[
                float(local_position[index] + self.position_offest[index])
                for index in range(3)
            ]
        )
        if direct_translate:
            translate_op.Set(target_local_position)
        else:
            local_transform = xformable.GetLocalTransformation()
            if isinstance(local_transform, tuple):
                local_transform = local_transform[0]
            normalized_transform = Gf.Matrix4d(local_transform)
            normalized_transform.SetTranslateOnly(target_local_position)
            reset_xform_stack = xformable.GetResetXformStack()
            matrix_op = xformable.MakeMatrixXform()
            matrix_op.Set(normalized_transform)
            xformable.SetResetXformStack(reset_xform_stack)

    # ------------------------------------------------------------------
    # pour 阶段：完整 6DoF 跟随
    # ------------------------------------------------------------------
    def init_pose_tracking(self, object_path, gripper_frame_path):
        """
        在开始"完整 6DoF 跟随"前调用一次（pour 阶段开始时）。
        计算并保存 夹爪→物体 的初始相对变换矩阵。
        """
        obj_prim     = get_prim_at_path(object_path)
        gripper_prim = get_prim_at_path(gripper_frame_path)
        if not obj_prim.IsValid() or not gripper_prim.IsValid():
            print(f"[Gripper] init_pose_tracking: invalid prim, skipping. obj={object_path}")
            self._relative_mat      = None
            self._pose_obj_path     = None
            self._pose_gripper_path = None
            return

        gripper_world_mat = UsdGeom.Xformable(gripper_prim).ComputeLocalToWorldTransform(0)
        obj_world_mat     = UsdGeom.Xformable(obj_prim).ComputeLocalToWorldTransform(0)
        # Gf 为行向量(Row-Major)约定：p' = p * M，乘法顺序左=先、右=后
        # 刚体约束：new_obj = relative * new_gripper，故 relative = obj * gripper^-1
        self._relative_mat      = obj_world_mat * gripper_world_mat.GetInverse()
        self._pose_obj_path     = object_path
        self._pose_gripper_path = gripper_frame_path

    def update_grasped_object_pose(self) -> None:
        """
        每帧调用：同时更新物体的 平移 + 旋转，实现完整 6DoF 跟随夹爪。
        仅在调用过 init_pose_tracking 之后才生效。
        """
        if self._relative_mat is None:
            return

        gripper_prim = get_prim_at_path(self._pose_gripper_path)
        if not gripper_prim.IsValid():
            return

        gripper_world_mat = UsdGeom.Xformable(gripper_prim).ComputeLocalToWorldTransform(0)

        # Row-Major：new_obj_world = relative * new_gripper
        new_obj_world_mat = self._relative_mat * gripper_world_mat

        obj_prim = get_prim_at_path(self._pose_obj_path)
        if not obj_prim.IsValid():
            return

        # 将世界变换转到父节点局部坐标系
        # Row-Major：local = world * parent^-1
        parent_prim = obj_prim.GetParent()
        if parent_prim and parent_prim.IsValid():
            parent_world_mat = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(0)
            new_obj_local_mat = new_obj_world_mat * parent_world_mat.GetInverse()
        else:
            parent_world_mat  = Gf.Matrix4d(1)   # identity
            new_obj_local_mat = new_obj_world_mat

        xformable = UsdGeom.Xformable(obj_prim)
        reset_xform_stack = xformable.GetResetXformStack()
        matrix_op = xformable.MakeMatrixXform()
        matrix_op.Set(new_obj_local_mat)
        xformable.SetResetXformStack(reset_xform_stack)

    # ------------------------------------------------------------------
    def release_object(self):
        rigid_body_api = self._managed_rigid_body_api
        original_kinematic = self._managed_rigid_body_original_kinematic
        release_error = None
        try:
            if rigid_body_api is not None:
                try:
                    self._write_rigid_body_motion_state(
                        rigid_body_api,
                        original_kinematic,
                        Gf.Vec3f(0.0, 0.0, 0.0),
                        Gf.Vec3f(0.0, 0.0, 0.0),
                    )
                except Exception as exc:
                    release_error = exc
        finally:
            self._clear_binding_state()
        if release_error is not None:
            raise release_error

    def _clear_binding_state(self):
        self.grasped_object_path = None
        self.gripper_frame_path  = None
        self.position_offest     = None
        self.inverse_transform_matrix = None
        self._managed_rigid_body_api = None
        self._managed_rigid_body_original_kinematic = None
        self._relative_mat       = None
        self._pose_obj_path      = None
        self._pose_gripper_path  = None
