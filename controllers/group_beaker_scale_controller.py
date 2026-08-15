from typing import Optional, Tuple, Any, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R

from controllers.base_controller import BaseController
from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from controllers.atomic_actions.pour_controller import PourController

from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom


class GroupBeakerScaleController(BaseController):
    """
    实验流程：
    1. 抓取 ErlenmeyerFlask_01
    2. 向 Beaker_01 倾倒
    3. 将 ErlenmeyerFlask_01 放到 target_plate_01 上
    4. 抓取 Beaker_01
    5. 放到 ElectronicScale_02 上称重
    6. 再次抓取 Beaker_01
    7. 放到 target_plate_04 上
    8. 抓取 ErlenmeyerFlask_02
    9. 向 Beaker_02 倾倒
    10. 将 ErlenmeyerFlask_02 放到 target_plate_02 上
    11. 抓取 Beaker_02
    12. 放到 ElectronicScale_02 上称重
    13. 再次抓取 Beaker_02
    14. 放到 target_plate_03 上
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        # 严格遵守初始化签名：只用 self.rmp_controller 和 robot.gripper
        self.pick_controller = PickController(name="pick", cspace_controller=self.rmp_controller)
        self.place_controller = PlaceController(
            name="place", cspace_controller=self.rmp_controller, gripper=robot.gripper
        )
        self.pour_controller = PourController(name="pour", cspace_controller=self.rmp_controller)

        self._event = 0
        self.reset_needed = False
        self._final_joint_positions = None  # 用于 collect 模式下延迟写入

    def reset(self):
        super().reset()
        self._event = 0
        self.reset_needed = False
        self._final_joint_positions = None

        self.pick_controller.reset()
        self.place_controller.reset()
        self.pour_controller.reset()

    def _cache_step_if_needed(self, state: Dict[str, Any]) -> None:
        """在 collect 模式下缓存一步数据（安全地检查 camera_data 存在）。"""
        if self.mode == "collect" and "camera_data" in state:
            self.data_collector.cache_step(
                camera_images=state["camera_data"],
                joint_angles=state["joint_positions"][:-1],
                language_instruction=self.get_language_instruction(),
            )

    def _default_size(self) -> np.ndarray:
        return np.array([0.04, 0.04, 0.08])

    def _get_place_xform_position(self, target_name: str) -> Optional[np.ndarray]:
        """从 /World/LabScene/<target_name>/place_xform 读取世界坐标。"""
        prim_path = f"/World/LabScene/{target_name}/place_xform"
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            return None
        world_mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        t = world_mat.ExtractTranslation()
        return np.array([t[0], t[1], t[2]])

    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        self.state = state

        # 为了安全，先确保关键字段存在；若不存在则等待下一帧
        if "joint_positions" not in state or "gripper_position" not in state:
            return None, False, False

        joint_positions = state["joint_positions"]
        gripper_position = state["gripper_position"]
        default_orientation = R.from_euler("xyz", np.radians([0, 90, 30])).as_quat()

        # 事件 0：抓取 ErlenmeyerFlask_01
        if self._event == 0:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "ErlenmeyerFlask_01"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False

                size = state.get(f"{obj_name}_size", self._default_size())

                # 在原始抓取点基础上，将抓取位置沿 Z 轴下移 2cm
                picking_pos = pos.copy()
                picking_pos[2] -= 0.01
                self.current_target_position = picking_pos

                action = self.pick_controller.forward(
                    picking_position=picking_pos,
                    current_joint_positions=joint_positions,
                    object_name=obj_name,
                    object_size=size,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                    gripper_distances=0.018,
                )

                # 夹爪中有烧瓶，需要维护物体位姿
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.pour_controller.reset()
                self._event = 1
                return None, False, False

        # 事件 1：向 Beaker_01 倾倒
        elif self._event == 1:
            self.current_action_type = "pour"
            if not self.pour_controller.is_done():
                source_name = "ErlenmeyerFlask_01"
                target_name = "Beaker_01"
                target_pos = state.get(f"{target_name}_position")
                if target_pos is None:
                    return None, False, False

                source_size = state.get(f"{source_name}_size", self._default_size())
                self.current_target_position = target_pos

                # 第一次进入 pour 阶段：初始化 6DoF pose 跟随
                if self.pour_controller._event == 0 and self.gripper_control._relative_mat is None:
                    self.gripper_control.init_pose_tracking(
                        f"/World/LabScene/{source_name}",
                        "/World/Franka/panda_hand/tool_center"
                    )

                action = self.pour_controller.forward(
                    articulation_controller=self.robot.get_articulation_controller(),
                    source_size=source_size,
                    target_position=target_pos,
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    gripper_position=gripper_position,
                    source_name=source_name,
                    pour_speed=-1,
                )
                # 倾倒全程：完整 6DoF 跟随（含旋转）
                self.gripper_control.update_grasped_object_pose(
                    pour_event=self.pour_controller._event
                )
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.gripper_control._relative_mat = None  # 清空，让下次 pour 重新初始化
                self.place_controller.reset()
                self._event = 2
                return None, False, False

        # 事件 2：将 ErlenmeyerFlask_01 放到 target_plate_01 上
        elif self._event == 2:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "target_plate_01"
                pos = state.get(f"{target_name}_place_position", state.get(f"{target_name}_position"))
                if pos is None:
                    return None, False, False

                self.current_target_position = pos
                action = self.place_controller.forward(
                    place_position=pos,
                    current_joint_positions=joint_positions,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                    place_offset_z=0.1,
                )
                # 放置过程物体仍在夹爪中，需要维护物体位姿（至少平移跟随）
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                # PlaceController 内部已经 release_object，这里只切换阶段
                self.pick_controller.reset()
                self._event = 3
                return None, False, False

        # 事件 3：抓取 Beaker_01
        elif self._event == 3:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "Beaker_01"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False

                size = state.get(f"{obj_name}_size", self._default_size())
                self.current_target_position = pos

                action = self.pick_controller.forward(
                    picking_position=pos,
                    current_joint_positions=joint_positions,
                    object_name=obj_name,
                    object_size=size,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.place_controller.reset()
                self._event = 4
                return None, False, False

        # 事件 4：Beaker_01 放到 ElectronicScale_02
        elif self._event == 4:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "ElectronicScale_02"
                pos = self._get_place_xform_position(target_name)
                if pos is None:
                    return None, False, False

                self.current_target_position = pos
                action = self.place_controller.forward(
                    place_position=pos,
                    current_joint_positions=joint_positions,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    pre_place_z=0.05,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.pick_controller.reset()
                self._event = 5
                return None, False, False

        # 事件 5：再次抓取 Beaker_01
        elif self._event == 5:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "Beaker_01"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False

                size = state.get(f"{obj_name}_size", self._default_size())
                self.current_target_position = pos

                action = self.pick_controller.forward(
                    picking_position=pos,
                    current_joint_positions=joint_positions,
                    object_name=obj_name,
                    object_size=size,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    pre_offset_z=0.10,
                    after_offset_z=0.09,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.place_controller.reset()
                self._event = 6
                return None, False, False

        # 事件 6：Beaker_01 放到 target_plate_03
        elif self._event == 6:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "target_plate_03"
                pos = state.get(f"{target_name}_place_position", state.get(f"{target_name}_position"))
                if pos is None:
                    return None, False, False

                self.current_target_position = pos
                action = self.place_controller.forward(
                    place_position=pos,
                    current_joint_positions=joint_positions,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                     place_offset_z=0.07,
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.pick_controller.reset()
                self._event = 7
                return None, False, False

        # 事件 7：抓取 ErlenmeyerFlask_02
        elif self._event == 7:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "ErlenmeyerFlask_02"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False

                size = state.get(f"{obj_name}_size", self._default_size())
                self.current_target_position = pos

                action = self.pick_controller.forward(
                    picking_position=pos,
                    current_joint_positions=joint_positions,
                    object_name=obj_name,
                    object_size=size,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.pour_controller.reset()
                self._event = 8
                return None, False, False

        # 事件 8：向 Beaker_02 倾倒
        elif self._event == 8:
            self.current_action_type = "pour"
            if not self.pour_controller.is_done():
                source_name = "ErlenmeyerFlask_02"
                target_name = "Beaker_02"
                target_pos = state.get(f"{target_name}_position")
                if target_pos is None:
                    return None, False, False

                source_size = state.get(f"{source_name}_size", self._default_size())
                self.current_target_position = target_pos

                # 第一次进入 pour 阶段：初始化 6DoF pose 跟随
                if self.pour_controller._event == 0 and self.gripper_control._relative_mat is None:
                    self.gripper_control.init_pose_tracking(
                        f"/World/LabScene/{source_name}",
                        "/World/Franka/panda_hand/tool_center"
                    )

                action = self.pour_controller.forward(
                    articulation_controller=self.robot.get_articulation_controller(),
                    source_size=source_size,
                    target_position=target_pos,
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    gripper_position=gripper_position,
                    source_name=source_name,
                    pour_speed=-1,
                )
                # 倾倒全程：完整 6DoF 跟随（含旋转）
                self.gripper_control.update_grasped_object_pose(
                    pour_event=self.pour_controller._event
                )
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.place_controller.reset()
                self._event = 9
                return None, False, False

        # 事件 9：将 ErlenmeyerFlask_02 放到 target_plate_02 上
        elif self._event == 9:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "target_plate_02"
                pos = state.get(f"{target_name}_place_position", state.get(f"{target_name}_position"))
                if pos is None:
                    return None, False, False

                self.current_target_position = pos
                action = self.place_controller.forward(
                    place_position=pos,
                    current_joint_positions=joint_positions,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    place_offset_z=0.1,
                )
                # 放置过程物体仍在夹爪中，需要维护物体位姿（至少平移跟随）
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                # PlaceController 内部已经 release_object，这里只切换阶段
                self.pick_controller.reset()
                self._event = 10
                return None, False, False

        # 事件 10：抓取 Beaker_02
        elif self._event == 10:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "Beaker_02"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False

                size = state.get(f"{obj_name}_size", self._default_size())
                self.current_target_position = pos

                action = self.pick_controller.forward(
                    picking_position=pos,
                    current_joint_positions=joint_positions,
                    object_name=obj_name,
                    object_size=size,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    end_effector_orientation=default_orientation,
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.place_controller.reset()
                self._event = 11
                return None, False, False

        # 事件 11：Beaker_02 放到 ElectronicScale_02
        elif self._event == 11:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "ElectronicScale_02"
                pos = self._get_place_xform_position(target_name)
                if pos is None:
                    return None, False, False

                self.current_target_position = pos
                action = self.place_controller.forward(
                    place_position=pos,
                    current_joint_positions=joint_positions,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    pre_place_z=0.05,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.pick_controller.reset()
                self._event = 12
                return None, False, False

        # 事件 12：再次抓取 Beaker_02
        elif self._event == 12:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "Beaker_02"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False

                size = state.get(f"{obj_name}_size", self._default_size())
                self.current_target_position = pos

                action = self.pick_controller.forward(
                    picking_position=pos,
                    current_joint_positions=joint_positions,
                    object_name=obj_name,
                    object_size=size,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    pre_offset_z=0.10,
                    after_offset_z=0.09,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.place_controller.reset()
                self._event = 13
                return None, False, False

        # 事件 13：Beaker_02 放到 target_plate_03 （最后一步）
        elif self._event == 13:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "target_plate_04"
                pos = state.get(f"{target_name}_place_position", state.get(f"{target_name}_position"))
                if pos is None:
                    return None, False, False

                self.current_target_position = pos
                action = self.place_controller.forward(
                    place_position=pos,
                    current_joint_positions=joint_positions,
                    gripper_control=self.gripper_control,
                    gripper_position=gripper_position,
                    end_effector_orientation=default_orientation,
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                # 所有阶段完成
                self._last_success = True
                self.reset_needed = True

                # 保存最终状态用于后续数据写入
                if self.mode == "collect":
                    self._final_joint_positions = joint_positions[:-1]

                self.gripper_control.release_object()
                return None, True, True

        # 默认兜底
        return None, False, False

