import re
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from .inference_engines.inference_engine_factory import InferenceEngineFactory


class GraspObjectTaskController(BaseController):
    """
    复合操作: 抓取物体任务
    调用的原子操作: PickController
    
    功能说明:
    - 使用PickController完成抓取物体的全过程，包括:
      - 识别目标物体的位置和尺寸
      - 移动到物体上方（保持X轴偏移0.05米）
      - 调整末端执行器到抓取姿态
      - 下降到物体抓取高度
      - 关闭夹爪抓取物体
      - 垂直抬起物体0.25米
      - 保持抓取状态并检查高度
    """
    
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        
    def _init_collect_mode(self, cfg, robot):
        """
        初始化数据收集模式的组件
        设置pick controller、夹爪控制和数据收集器
        
        Args:
            cfg: 包含收集设置的配置对象
            robot: 要控制的机器人实例
        """
        super()._init_collect_mode(cfg, robot)
        
        # 创建PickController原子操作控制器
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02],
            position_threshold=0.01
        )

    def reset(self):
        """重置控制器状态"""
        super().reset()
        if self.mode == "collect":
            self.pick_controller.reset()
        else:
            self.inference_engine.reset()
        self.initial_position = None
    
    def step(self, state):
        """
        执行一步控制
        
        Args:
            state: 当前环境状态字典
            
        Returns:
            tuple: (action, done, success) 指示控制输出和episode状态
        """
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        """检查任务是否成功完成"""
        return self.state['object_position'][2] > self.initial_position[2] + 0.1

    def _init_infer_mode(self, cfg, robot):
        """
        初始化推理模式的组件
        创建推理引擎和轨迹控制器
        
        Args:
            cfg: 包含模型路径和设置的配置对象
            robot: 要控制的机器人实例
        """
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )
        
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )
        
    def _step_collect(self, state):
        """
        在收集模式下执行一步
        记录演示并管理episode转换
        
        Args:
            state (dict): 当前环境状态
            
        Returns:
            tuple: (action, done, success) 指示控制输出和episode状态
        """
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        
        if not self.pick_controller.is_done():
            # 设置默认的末端执行器姿态（向下抓取姿态）
            end_effector_orientation = R.from_euler('xyz', np.radians([0, 90, 0])).as_quat()
            
            # 设置机器人的位置用于计算接近方向（可选，如果未设置则使用默认值）
            try:
                if hasattr(self.robot, 'get_world_pose'):
                    robot_pose = self.robot.get_world_pose()
                    if robot_pose is not None and len(robot_pose) > 0:
                        self.pick_controller.set_robot_position(np.array(robot_pose[0]))
            except Exception:
                pass  # 如果获取失败，使用PickController的默认值
            
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=end_effector_orientation,
                pre_offset_x=0.05,
                pre_offset_z=0.12,
                after_offset_z=0.25,
                gripper_distances=None
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False
        
        # 检查是否成功
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self.reset_needed = True
            return None, True, True

        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        return None, True, False
        
    def _step_infer(self, state):
        """
        在推理模式下执行一步
        使用推理引擎处理观察并生成动作
        
        Args:
            state (dict): 当前环境状态
            
        Returns:
            tuple: (action, done, success) 指示控制输出和episode状态
        """
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
            
        action = self.inference_engine.step_inference(state)
        
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        """获取当前任务的语言指令
        
        Returns:
            Optional[str]: 语言指令，如果不可用则返回None
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ', ' ').lower()
        self._language_instruction = f"Pick up the {object_name} from the table"
        return self._language_instruction
