from typing import Optional, Tuple, Any, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R

from controllers.base_controller import BaseController
from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from controllers.atomic_actions.pour_controller import PourController

from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom


class BeakerFlaskExperimentController(BaseController):
    """
    实验流程：
    1.  (event 0) 抓取 Beaker_03
    2.  (event 1) Beaker_03 向 ErlenmeyerFlask_04 倾倒
    3.  (event 2) 将 Beaker_03 放到 target_plate_05 上
    4.  (event 3) 抓取 Beaker_04
    5.  (event 4) Beaker_04 向 ErlenmeyerFlask_04 倾倒
    6.  (event 5) 将 Beaker_04 放到 target_plate_06 上
    7.  (event 6) 抓取 ErlenmeyerFlask_04
    8.  (event 7) 将 ErlenmeyerFlask_04 放到 instrument_21 上
    9.  (event 8) 再次抓取 ErlenmeyerFlask_04
    10. (event 9) 将 ErlenmeyerFlask_04 放到 target_plate_07 上
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
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
        """在 collect 模式下缓存一步数据。"""
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

        if "joint_positions" not in state or "gripper_position" not in state:
            return None, False, False

        joint_positions = state["joint_positions"]
        gripper_position = state["gripper_position"]
        default_orientation = R.from_euler("xyz", np.radians([0, 90, 30])).as_quat()

        # ── 事件 0：抓取 Beaker_01 ───────────────────────────────────────────
        if self._event == 0:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "Beaker_03"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False
                pos = pos.copy()
                pos[2] += 0.02  # 抓取位置向上偏移 2cm

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
                self.pour_controller.reset()
                self._event = 1
                return None, False, False

        # ── 事件 1：Beaker_01 向 ErlenmeyerFlask_04 倾倒 ────────────────────
        elif self._event == 1:
            self.current_action_type = "pour"
            if not self.pour_controller.is_done():
                source_name = "Beaker_03"
                target_name = "ErlenmeyerFlask_04"
                target_pos = state.get(f"{target_name}_position")
                if target_pos is None:
                    return None, False, False

                source_size = state.get(f"{source_name}_size", self._default_size())
                self.current_target_position = target_pos

                # 靠近阶段（event 0-1）仅做位置跟随，避免 IK 重定向导致烧杯"乱转"。
                # 到实际倾倒开始（event 2）时才初始化完整 6DoF 跟随。
                if self.pour_controller._event == 2 and self.gripper_control._relative_mat is None:
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
                # event 0-1（靠近阶段）：仅位置跟随，不更新旋转
                # event 2+（倾倒阶段）：完整 6DoF 跟随
                if self.pour_controller._event >= 2:
                    self.gripper_control.update_grasped_object_pose()
                else:
                    self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.gripper_control._relative_mat = None  # 清空，下次 pour 重新初始化
                self.place_controller.reset()
                self._event = 2
                return None, False, False

        # ── 事件 2：将 Beaker_01 放到 target_plate_05 上 ─────────────────────
        elif self._event == 2:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "target_plate_05"
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
                self.pick_controller.reset()
                self._event = 3
                return None, False, False

        # ── 事件 3：抓取 Beaker_02 ───────────────────────────────────────────
        elif self._event == 3:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "Beaker_04"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False
                pos = pos.copy()
                pos[2] += 0.02  # 抓取位置向上偏移 2cm

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
                self.pour_controller.reset()
                self._event = 4
                return None, False, False

        # ── 事件 4：Beaker_02 向 ErlenmeyerFlask_04 倾倒 ────────────────────
        elif self._event == 4:
            self.current_action_type = "pour"
            if not self.pour_controller.is_done():
                source_name = "Beaker_04"
                target_name = "ErlenmeyerFlask_04"
                target_pos = state.get(f"{target_name}_position")
                if target_pos is None:
                    return None, False, False

                source_size = state.get(f"{source_name}_size", self._default_size())
                self.current_target_position = target_pos

                # 靠近阶段（event 0-1）仅做位置跟随，避免 IK 重定向导致烧杯"乱转"。
                # 到实际倾倒开始（event 2）时才初始化完整 6DoF 跟随。
                if self.pour_controller._event == 2 and self.gripper_control._relative_mat is None:
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
                # event 0-1（靠近阶段）：仅位置跟随，不更新旋转
                # event 2+（倾倒阶段）：完整 6DoF 跟随
                if self.pour_controller._event >= 2:
                    self.gripper_control.update_grasped_object_pose()
                else:
                    self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.gripper_control._relative_mat = None  # 清空，下次 pour 重新初始化
                self.place_controller.reset()
                self._event = 5
                return None, False, False

        # ── 事件 5：将 Beaker_02 放到 target_plate_06 上 ─────────────────────
        elif self._event == 5:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "target_plate_06"
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
                self.pick_controller.reset()
                self._event = 6
                return None, False, False

        # ── 事件 6：抓取 ErlenmeyerFlask_04 ─────────────────────────────────
        elif self._event == 6:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "ErlenmeyerFlask_04"
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
                    gripper_distances=0.018,
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.place_controller.reset()
                self._event = 7
                return None, False, False

        # ── 事件 7：将 ErlenmeyerFlask_04 放到 instrument_21 上 ──────────────
        elif self._event == 7:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "instrument_21"
                # 从 /World/LabScene/World/instrument_21/place_xform 读取世界坐标
                _place_prim = get_prim_at_path("/World/LabScene/World/instrument_21/place_xform")
                if _place_prim.IsValid():
                    _t = UsdGeom.Xformable(_place_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
                    pos = np.array([_t[0], _t[1], _t[2]])
                else:
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
                    place_offset_z=0.1,
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.pick_controller.reset()
                self._event = 8
                return None, False, False

        # ── 事件 8：再次抓取 ErlenmeyerFlask_04 ─────────────────────────────
        elif self._event == 8:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "ErlenmeyerFlask_04"
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
                    gripper_distances=0.018,
                )
                self.gripper_control.update_grasped_object_position()
                self._cache_step_if_needed(state)
                return action, False, False
            else:
                self.place_controller.reset()
                self._event = 9
                return None, False, False

        # ── 事件 9：将 ErlenmeyerFlask_04 放到 target_plate_07（最后一步）────
        elif self._event == 9:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "target_plate_07"
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
                    place_offset_z=0.1,
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
