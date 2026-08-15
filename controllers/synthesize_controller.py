from typing import Optional, Tuple, Any, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R
from controllers.base_controller import BaseController
from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from controllers.atomic_actions.press_controller import PressController
from controllers.atomic_actions.pour_controller import PourController


class SynthesizeController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        # 初始化所有需要的控制器
        self.pick_controller = PickController(name="pick", cspace_controller=self.rmp_controller)
        self.place_controller = PlaceController(name="place", cspace_controller=self.rmp_controller, gripper=robot.gripper)
        self.press_controller = PressController(name="press", cspace_controller=self.rmp_controller, gripper=robot.gripper)
        self.pour_controller = PourController(name="pour", cspace_controller=self.rmp_controller)
        
        # 状态机
        self._event = 0
        self.reset_needed = False

    def reset(self):
        super().reset()
        self._event = 0
        self.reset_needed = False
        self.pick_controller.reset()
        self.place_controller.reset()
        self.press_controller.reset()
        self.pour_controller.reset()

    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        self.state = state
        
        if self.reset_needed:
            return None, True, True
        
        # 抓取装有苯甲酸的容器
        if self._event == 0:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "ErlenmeyerFlask_Solid1"
                pos = state.get(f"{obj_name}_position")
                if pos is None: 
                    return None, False, False
                
                self.current_target_position = pos
                action = self.pick_controller.forward(
                    picking_position=pos,
                    current_joint_positions=state['joint_positions'],
                    object_name=obj_name,
                    object_size=state.get(f"{obj_name}_size", np.array([0.04, 0.04, 0.08])),
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    gripper_distances=0.012  # 特殊处理ErlenmeyerFlask_Solid
                )
                return action, False, False
            else:
                self.place_controller.reset()
                self._event = 1
                return None, False, False
        
        # 将装有苯甲酸的容器放到加热板上
        elif self._event == 1:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                obj_name = "HeatingPlate"
                pos = state.get(f"{obj_name}_place_position", state.get(f"{obj_name}_position"))
                if pos is None: 
                    return None, False, False
                
                self.current_target_position = pos
                action = self.place_controller.forward(
                    place_position=pos,
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    place_offset_z=0.12  # 特殊处理ErlenmeyerFlask_Solid
                )
                return action, False, False
            else:
                self.press_controller.reset()
                self._event = 2
                return None, False, False
        
        # 按下加热板按钮启动搅拌
        elif self._event == 2:
            self.current_action_type = "press"
            if not self.press_controller.is_done():
                obj_name = "HeatingPlate"
                pos = state.get(f"{obj_name}_press_position")
                if pos is None: 
                    return None, False, False
                
                self.current_target_position = pos
                action = self.press_controller.forward(
                    target_position=pos,
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
                )
                return action, False, False
            else:
                self.pick_controller.reset()
                self._event = 3
                return None, False, False
                
        # 抓取装有浓硫酸的锥形瓶
        elif self._event == 3:
            self.current_action_type = "pick"
            if not self.pick_controller.is_done():
                obj_name = "ErlenmeyerFlask_Liquid1"
                pos = state.get(f"{obj_name}_position")
                if pos is None: 
                    return None, False, False
                
                self.current_target_position = pos
                action = self.pick_controller.forward(
                    picking_position=pos,
                    current_joint_positions=state['joint_positions'],
                    object_name=obj_name,
                    object_size=state.get(f"{obj_name}_size", np.array([0.04, 0.04, 0.08])),
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    gripper_distances=0.008  # 特殊处理ErlenmeyerFlask_Liquid
                )
                return action, False, False
            else:
                self.pour_controller.reset()
                self._event = 4
                return None, False, False
                
        # 将浓硫酸倒入苯甲酸容器中
        elif self._event == 4:
            self.current_action_type = "pour"
            if not self.pour_controller.is_done():
                self.gripper_control.update_grasped_object_position()
                
                source_name = "ErlenmeyerFlask_Liquid1"
                target_name = "ErlenmeyerFlask_Solid1"
                
                target_pos = state.get(f"{target_name}_position")
                if target_pos is None: 
                    return None, False, False
                
                self.current_target_position = target_pos
                action = self.pour_controller.forward(
                    articulation_controller=self.robot.get_articulation_controller(),
                    source_size=state.get(f"{source_name}_size", np.array([0.04, 0.04, 0.08])),
                    target_position=target_pos,
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    gripper_position=state['gripper_position'],
                    source_name=source_name,
                    pour_speed=-1  # 强制参数
                )
                return action, False, False
            else:
                self.place_controller.reset()
                self._event = 5
                return None, False, False
                
        # 将浓硫酸瓶放回原处
        elif self._event == 5:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                self.gripper_control.update_grasped_object_position()
                
                obj_name = "TargetPlatform1"
                pos = state.get(f"{obj_name}_place_position", state.get(f"{obj_name}_position"))
                if pos is None: 
                    return None, False, False
                
                self.current_target_position = pos
                action = self.place_controller.forward(
                    place_position=pos,
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    place_offset_z=0.08  # 特殊处理ErlenmeyerFlask_Liquid
                )
                return action, False, False
            else:
                self.reset_needed = True
                return None, True, True