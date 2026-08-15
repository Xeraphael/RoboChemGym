from typing import Optional, Tuple, Any, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R
from controllers.base_controller import BaseController
from controllers.atomic_actions.pick_controller import PickController


class BeakerPickTaskController(BaseController):
    """
    Controller for picking up Beaker_01 task with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous picking
    """
    
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        self._event = 0
        self.reset_needed = False
        self._final_joint_positions = None
        
    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )
    
    def _init_infer_mode(self, cfg, robot):
        """Initialize controller for inference mode."""
        super()._init_infer_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        )
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._event = 0
        self.reset_needed = False
        self._final_joint_positions = None
        self.initial_position = None
        
        if self.mode == "collect":
            self.pick_controller.reset()
        else:
            self.inference_engine.reset()
            self.pick_controller.reset()
    
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state
        
        # 初始化初始位置
        if self.initial_position is None:
            beaker_pos = state.get("Beaker_01_position")
            if beaker_pos is not None:
                self.initial_position = np.array(beaker_pos)
        
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
    
    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute collection mode step."""
        if self._event == 0:
            self.current_action_type = "pick"
            
            if not self.pick_controller.is_done():
                obj_name = "Beaker_01"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False
                
                obj_size = state.get(f"{obj_name}_size", np.array([0.04, 0.04, 0.08]))
                
                self.current_target_position = pos
                action = self.pick_controller.forward(
                    picking_position=pos,
                    current_joint_positions=state['joint_positions'],
                    object_name=obj_name,
                    object_size=obj_size,
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    pre_offset_x=0.05,
                    pre_offset_z=0.05
                )
                
                # 更新抓取物体位置
                self.gripper_control.update_grasped_object_position()
                
                # 缓存数据
                if self.mode == "collect" and 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        language_instruction=self.get_language_instruction()
                    )
                
                return action, False, False
            else:
                # Pick动作完成，检查是否成功
                beaker_pos = state.get("Beaker_01_position")
                if beaker_pos is not None and self.initial_position is not None:
                    # 检查物体是否被成功抓取（高度增加）
                    if beaker_pos[2] > self.initial_position[2] + 0.1:
                        self._last_success = True
                        self.reset_needed = True
                        
                        # 保存最终状态用于后续数据写入
                        if self.mode == "collect":
                            self._final_joint_positions = state['joint_positions'][:-1]
                            self.data_collector.write_cached_data(self._final_joint_positions)
                        
                        return None, True, True
                    else:
                        # 抓取失败
                        self._last_success = False
                        self.reset_needed = True
                        if self.mode == "collect":
                            self.data_collector.clear_cache()
                        return None, True, False
                else:
                    # 无法判断，默认失败
                    self._last_success = False
                    self.reset_needed = True
                    if self.mode == "collect":
                        self.data_collector.clear_cache()
                    return None, True, False
        
        return None, False, False
    
    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute inference mode step."""
        if self._event == 0:
            self.current_action_type = "pick"
            
            # 在inference模式下，如果pick controller未完成，使用它
            if not self.pick_controller.is_done():
                obj_name = "Beaker_01"
                pos = state.get(f"{obj_name}_position")
                if pos is None:
                    return None, False, False
                
                obj_size = state.get(f"{obj_name}_size", np.array([0.04, 0.04, 0.08]))
                
                self.current_target_position = pos
                action = self.pick_controller.forward(
                    picking_position=pos,
                    current_joint_positions=state['joint_positions'],
                    object_name=obj_name,
                    object_size=obj_size,
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    pre_offset_x=0.05,
                    pre_offset_z=0.05
                )
                
                # 更新抓取物体位置
                self.gripper_control.update_grasped_object_position()
                
                return action, False, False
            else:
                # Pick动作完成，检查是否成功
                beaker_pos = state.get("Beaker_01_position")
                if beaker_pos is not None and self.initial_position is not None:
                    # 检查物体是否被成功抓取（高度增加）
                    if beaker_pos[2] > self.initial_position[2] + 0.1:
                        self._last_success = True
                        self.reset_needed = True
                        return None, True, True
                    else:
                        # 抓取失败
                        self._last_success = False
                        self.reset_needed = True
                        return None, True, False
                else:
                    # 使用推理引擎继续
                    language_instruction = self.get_language_instruction()
                    state['language_instruction'] = language_instruction
                    action = self.inference_engine.step_inference(state)
                    
                    # 检查成功条件
                    if beaker_pos is not None and self.initial_position is not None:
                        if beaker_pos[2] > self.initial_position[2] + 0.1:
                            self.check_success_counter += 1
                        else:
                            self.check_success_counter = 0
                        
                        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
                        if self._last_success:
                            self.reset_needed = True
                            return action, True, True
                    
                    return action, False, False
        
        return None, False, False
    
    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        
        Returns:
            Optional[str]: The language instruction
        """
        self._language_instruction = "Pick up the Beaker_01 from the table"
        return self._language_instruction
