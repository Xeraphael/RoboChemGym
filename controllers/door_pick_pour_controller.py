import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.core.utils.types import ArticulationAction

from .base_controller import BaseController
from .atomic_actions.open_controller import OpenController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.pour_controller import PourController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from .inference_engines.inference_engine_factory import InferenceEngineFactory
from utils.object_utils import ObjectUtils


class TaskPhase(Enum):
    """任务阶段枚举"""
    OPENING = "opening"
    PICKING_BEAKER = "picking_beaker"
    PLACING_BEAKER = "placing_beaker"
    PICKING_BOTTLE = "picking_bottle"
    POURING = "pouring"
    PLACING_BOTTLE = "placing_bottle"
    FINISHED = "finished"


class DoorPickPourTaskController(BaseController):
    """
    复合操作: 开门取物倒液任务
    
    调用的原子操作: 
    - OpenController: 打开门
    - PickController: 抓取烧杯和锥形瓶
    - PlaceController: 放置烧杯和锥形瓶
    - PourController: 执行倒液操作
    
    功能说明:
    - Phase 0: 打开门 - 使用OpenController顺时针旋转门
    - Phase 1: 抓取烧杯 - 使用PickController从柜子内抓取烧杯
    - Phase 2: 放置烧杯 - 使用PlaceController将烧杯放在目标平台上
    - Phase 3: 抓取锥形瓶 - 使用PickController抓取锥形瓶
    - Phase 4: 倒液 - 使用PourController将锥形瓶内液体倒入烧杯
    - Phase 5: 放置锥形瓶 - 使用PlaceController将锥形瓶放回指定平台
    """
    
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.current_phase = TaskPhase.OPENING
        self.initial_beaker_position = None
        self.initial_bottle_position = None
        self.beaker_place_position = None
        self.bottle_place_position = None
        
    def _init_collect_mode(self, cfg, robot):
        """
        初始化数据收集模式的组件
        设置各个原子操作控制器
        
        Args:
            cfg: 包含收集设置的配置对象
            robot: 要控制的机器人实例
        """
        super()._init_collect_mode(cfg, robot)
        
        # 开门控制器 - 使用door类型，顺时针旋转
        self.open_controller = OpenController(
            name="open_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.0025, 0.005, 0.08, 0.002, 0.05, 0.05, 0.01, 0.008],
            furniture_type="door",
            door_open_direction="clockwise"
        )
        
        # 抓取烧杯控制器
        self.pick_beaker_controller = PickController(
            name="pick_beaker_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        
        # 放置烧杯控制器
        self.place_beaker_controller = PlaceController(
            name="place_beaker_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.008, 0.02, 0.05, 0.01, 0.02]
        )
        
        # 抓取锥形瓶控制器
        self.pick_bottle_controller = PickController(
            name="pick_bottle_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        
        # 倒液控制器 - 使用特定的pour_speed以实现约20°的倾角
        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.002, 0.009, 0.01, 0.009, 0.01],
            speed=0.35  # 设置倒液速度以实现约20°的倾角
        )
        
        # 放置锥形瓶控制器
        self.place_bottle_controller = PlaceController(
            name="place_bottle_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.008, 0.02, 0.05, 0.01, 0.02]
        )
        
        self.current_phase = TaskPhase.OPENING
        
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

    def reset(self):
        """重置控制器状态"""
        super().reset()
        self.current_phase = TaskPhase.OPENING
        self.initial_beaker_position = None
        self.initial_bottle_position = None
        self.beaker_place_position = None
        self.bottle_place_position = None
        
        if self.mode == "collect":
            self.open_controller.reset()
            self.pick_beaker_controller.reset()
            self.place_beaker_controller.reset()
            self.pick_bottle_controller.reset()
            self.pour_controller.reset()
            self.place_bottle_controller.reset()
        else:
            self.inference_engine.reset()
    
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        执行一步控制
        
        Args:
            state: 当前环境状态字典
            
        Returns:
            tuple: (action, done, success) 指示控制输出和episode状态
        """
        self.state = state
        
        # 初始化位置信息
        if self.initial_beaker_position is None and 'object_position' in state:
            self.initial_beaker_position = state['object_position'].copy()
        if self.initial_bottle_position is None and 'target_position' in state:
            self.initial_bottle_position = state['target_position'].copy()
        
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        在收集模式下执行一步
        记录演示并管理episode转换
        
        Args:
            state (dict): 当前环境状态
            
        Returns:
            tuple: (action, done, success) 指示控制输出和episode状态
        """
        # 验证必要的状态信息
        if 'joint_positions' not in state or state['joint_positions'] is None:
            return None, False, False
        
        if self.current_phase == TaskPhase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        # 检查当前阶段是否完成，如果完成则切换阶段
        if self._is_current_phase_done():
            self.current_phase = self._get_next_phase()
            if self.current_phase == TaskPhase.FINISHED:
                # 所有阶段完成
                if 'joint_positions' in state:
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.reset_needed = True
                # 返回一个空的但有效的动作
                empty_action = ArticulationAction(
                    joint_positions=[None] * len(state['joint_positions']) if 'joint_positions' in state else None
                )
                return empty_action, True, True
            else:
                # 切换到下一阶段，重置控制器
                self._reset_current_controller()
                # 获取新阶段的动作
                action = self._get_phase_action(state)
                if action is None:
                    action = ArticulationAction(
                        joint_positions=[None] * len(state['joint_positions'])
                    )
                return action, False, False

        # 获取当前活动控制器并执行动作
        action = self._get_phase_action(state)
        
        # 如果动作为None，创建一个空的但有效的动作
        if action is None:
            action = ArticulationAction(
                joint_positions=[None] * len(state['joint_positions'])
            )
        
        # 验证动作的有效性（检查是否有NaN或无效值）
        try:
            if hasattr(action, 'joint_positions') and action.joint_positions is not None:
                joint_positions = np.array([p for p in action.joint_positions if p is not None], dtype=np.float64)
                if len(joint_positions) > 0 and np.any(np.isnan(joint_positions)):
                    print("Warning: Action contains NaN values, using empty action")
                    action = ArticulationAction(joint_positions=[None] * len(state['joint_positions']))
        except Exception as e:
            print(f"Warning: Error validating action: {e}")
            action = ArticulationAction(joint_positions=[None] * len(state['joint_positions']))
        
        # 收集数据
        if 'camera_data' in state and action is not None:
            self.data_collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=state['joint_positions'][:-1],
                language_instruction=self.get_language_instruction()
            )
        
        return action, False, False
    
    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
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
        
        # 检查任务成功
        success = self._check_task_success(state)
        if success:
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True
        return action, False, False
    
    def _get_phase_action(self, state: Dict[str, Any]):
        """根据当前阶段获取相应的动作"""
        # 验证状态
        if state is None or 'joint_positions' not in state or 'gripper_position' not in state:
            return None
        
        try:
            object_utils = ObjectUtils.get_instance()
        except Exception as e:
            print(f"Error getting ObjectUtils: {e}")
            return None
        
        if self.current_phase == TaskPhase.OPENING:
            # 开门阶段
            door_path = self.cfg.task.get("door_paths", [{}])[0].get("path", "/World/MuffleFurnace")
            handle_path = f"{door_path}/handle"
            revolute_joint_path = f"{door_path}/RevoluteJoint"
            
            door_orientation = euler_angles_to_quats([0, 90, 0], degrees=True, extrinsic=False)
            
            try:
                handle_position = object_utils.get_geometry_center(object_path=handle_path)
                revolute_joint_position = object_utils.get_revolute_joint_positions(joint_path=revolute_joint_path)
                
                if handle_position is None or np.any(np.isnan(handle_position)):
                    print(f"Warning: Invalid handle position for {handle_path}")
                    return None
                
                return self.open_controller.forward(
                    handle_position=handle_position,
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=door_orientation,
                    angle=110.0,
                    revolute_joint_position=revolute_joint_position
                )
            except Exception as e:
                print(f"Error in OPENING phase: {e}")
                return None
            
        elif self.current_phase == TaskPhase.PICKING_BEAKER:
            # 抓取烧杯阶段
            try:
                beaker_path = self.cfg.task.obj_paths[0]["path"]
                beaker_position = object_utils.get_geometry_center(object_path=beaker_path)
                beaker_size = object_utils.get_object_size(object_path=beaker_path)
                
                if beaker_position is None or np.any(np.isnan(beaker_position)):
                    print(f"Warning: Invalid beaker position for {beaker_path}")
                    return None
                if beaker_size is None or np.any(np.isnan(beaker_size)):
                    print(f"Warning: Invalid beaker size for {beaker_path}")
                    return None
                
                # 设置机器人的位置用于计算接近方向
                try:
                    if hasattr(self.robot, 'get_world_pose'):
                        robot_pose = self.robot.get_world_pose()
                        if robot_pose is not None and len(robot_pose) > 0:
                            self.pick_beaker_controller.set_robot_position(np.array(robot_pose[0]))
                except Exception:
                    pass
                
                end_effector_orientation = R.from_euler('xyz', np.radians([0, 90, 30])).as_quat()
                
                return self.pick_beaker_controller.forward(
                    picking_position=beaker_position,
                    current_joint_positions=state['joint_positions'],
                    object_name=beaker_path.split("/")[-1],
                    object_size=beaker_size,
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=end_effector_orientation,
                    pre_offset_z=0.1,
                    after_offset_z=0.1
                )
            except Exception as e:
                print(f"Error in PICKING_BEAKER phase: {e}")
                return None
            
        elif self.current_phase == TaskPhase.PLACING_BEAKER:
            # 放置烧杯阶段
            try:
                target_plat_path = self.cfg.task.obj_paths[1]["path"]
                beaker_place_pos = object_utils.get_geometry_center(object_path=target_plat_path)
                
                if beaker_place_pos is None or np.any(np.isnan(beaker_place_pos)):
                    print(f"Warning: Invalid target platform position for {target_plat_path}")
                    return None
                
                self.beaker_place_position = beaker_place_pos.copy()
                
                end_effector_orientation = R.from_euler('xyz', np.radians([0, 90, 30])).as_quat()
                
                return self.place_beaker_controller.forward(
                    place_position=beaker_place_pos,
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=end_effector_orientation,
                    gripper_position=state['gripper_position'],
                    pre_place_z=0.2,
                    place_offset_z=0.1
                )
            except Exception as e:
                print(f"Error in PLACING_BEAKER phase: {e}")
                return None
            
        elif self.current_phase == TaskPhase.PICKING_BOTTLE:
            # 抓取锥形瓶阶段
            try:
                bottle_path = self.cfg.task.obj_paths[2]["path"] if len(self.cfg.task.obj_paths) > 2 else self.cfg.task.obj_paths[0]["path"]
                bottle_position = object_utils.get_geometry_center(object_path=bottle_path)
                bottle_size = object_utils.get_object_size(object_path=bottle_path)
                
                if bottle_position is None or np.any(np.isnan(bottle_position)):
                    print(f"Warning: Invalid bottle position for {bottle_path}")
                    return None
                if bottle_size is None or np.any(np.isnan(bottle_size)):
                    print(f"Warning: Invalid bottle size for {bottle_path}")
                    return None
                
                # 设置机器人的位置用于计算接近方向
                try:
                    if hasattr(self.robot, 'get_world_pose'):
                        robot_pose = self.robot.get_world_pose()
                        if robot_pose is not None and len(robot_pose) > 0:
                            self.pick_bottle_controller.set_robot_position(np.array(robot_pose[0]))
                except Exception:
                    pass
                
                end_effector_orientation = R.from_euler('xyz', np.radians([0, 90, 40])).as_quat()
                
                return self.pick_bottle_controller.forward(
                    picking_position=bottle_position,
                    current_joint_positions=state['joint_positions'],
                    object_name=bottle_path.split("/")[-1],
                    object_size=bottle_size,
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=end_effector_orientation,
                    pre_offset_z=0.1,
                    after_offset_z=0.1
                )
            except Exception as e:
                print(f"Error in PICKING_BOTTLE phase: {e}")
                return None
            
        elif self.current_phase == TaskPhase.POURING:
            # 倒液阶段
            try:
                if self.beaker_place_position is None:
                    target_plat_path = self.cfg.task.obj_paths[1]["path"]
                    self.beaker_place_position = object_utils.get_geometry_center(object_path=target_plat_path)
                
                if self.beaker_place_position is None or np.any(np.isnan(self.beaker_place_position)):
                    print("Warning: Invalid beaker place position for pouring")
                    return None
                
                bottle_path = self.cfg.task.obj_paths[2]["path"] if len(self.cfg.task.obj_paths) > 2 else self.cfg.task.obj_paths[0]["path"]
                bottle_size = object_utils.get_object_size(object_path=bottle_path)
                
                if bottle_size is None or np.any(np.isnan(bottle_size)):
                    print(f"Warning: Invalid bottle size for {bottle_path}")
                    return None
                
                return self.pour_controller.forward(
                    articulation_controller=self.robot.get_articulation_controller(),
                    source_size=bottle_size,
                    target_position=self.beaker_place_position,
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    gripper_position=state['gripper_position'],
                    source_name=bottle_path.split("/")[-1]
                )
            except Exception as e:
                print(f"Error in POURING phase: {e}")
                return None
            
        elif self.current_phase == TaskPhase.PLACING_BOTTLE:
            # 放置锥形瓶阶段
            try:
                bottle_place_path = self.cfg.task.obj_paths[3]["path"] if len(self.cfg.task.obj_paths) > 3 else self.cfg.task.obj_paths[1]["path"]
                bottle_place_pos = object_utils.get_geometry_center(object_path=bottle_place_path)
                
                if bottle_place_pos is None or np.any(np.isnan(bottle_place_pos)):
                    print(f"Warning: Invalid bottle place position for {bottle_place_path}")
                    return None
                
                self.bottle_place_position = bottle_place_pos.copy()
                
                end_effector_orientation = R.from_euler('xyz', np.radians([0, 90, 40])).as_quat()
                
                return self.place_bottle_controller.forward(
                    place_position=bottle_place_pos,
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=end_effector_orientation,
                    gripper_position=state['gripper_position'],
                    pre_place_z=0.2,
                    place_offset_z=0.1
                )
            except Exception as e:
                print(f"Error in PLACING_BOTTLE phase: {e}")
                return None
        
        return None
    
    def _is_current_phase_done(self) -> bool:
        """检查当前阶段是否完成"""
        phase_controller_map = {
            TaskPhase.OPENING: self.open_controller,
            TaskPhase.PICKING_BEAKER: self.pick_beaker_controller,
            TaskPhase.PLACING_BEAKER: self.place_beaker_controller,
            TaskPhase.PICKING_BOTTLE: self.pick_bottle_controller,
            TaskPhase.POURING: self.pour_controller,
            TaskPhase.PLACING_BOTTLE: self.place_bottle_controller,
        }
        
        controller = phase_controller_map.get(self.current_phase)
        return controller is not None and controller.is_done()
    
    def _get_next_phase(self) -> TaskPhase:
        """获取下一阶段"""
        phase_sequence = [
            TaskPhase.OPENING,
            TaskPhase.PICKING_BEAKER,
            TaskPhase.PLACING_BEAKER,
            TaskPhase.PICKING_BOTTLE,
            TaskPhase.POURING,
            TaskPhase.PLACING_BOTTLE,
            TaskPhase.FINISHED
        ]
        
        try:
            current_idx = phase_sequence.index(self.current_phase)
            if current_idx < len(phase_sequence) - 1:
                return phase_sequence[current_idx + 1]
        except ValueError:
            pass
        return TaskPhase.FINISHED
    
    def _reset_current_controller(self):
        """重置当前阶段的控制器"""
        phase_controller_map = {
            TaskPhase.OPENING: self.open_controller,
            TaskPhase.PICKING_BEAKER: self.pick_beaker_controller,
            TaskPhase.PLACING_BEAKER: self.place_beaker_controller,
            TaskPhase.PICKING_BOTTLE: self.pick_bottle_controller,
            TaskPhase.POURING: self.pour_controller,
            TaskPhase.PLACING_BOTTLE: self.place_bottle_controller,
        }
        
        controller = phase_controller_map.get(self.current_phase)
        if controller is not None:
            controller.reset()
    
    def _check_task_success(self, state: Dict[str, Any]) -> bool:
        """检查任务是否成功完成"""
        # 检查所有阶段是否都完成了
        return self.current_phase == TaskPhase.FINISHED
    
    def get_language_instruction(self) -> Optional[str]:
        """获取当前任务的语言指令"""
        phase_instructions = {
            TaskPhase.OPENING: "Open the door of the device",
            TaskPhase.PICKING_BEAKER: "Pick up the beaker from inside the cabinet",
            TaskPhase.PLACING_BEAKER: "Place the beaker on the target platform",
            TaskPhase.PICKING_BOTTLE: "Pick up the conical bottle",
            TaskPhase.POURING: "Pour the liquid from the bottle into the beaker",
            TaskPhase.PLACING_BOTTLE: "Place the bottle back on the platform"
        }
        return phase_instructions.get(self.current_phase, "Complete the door pick pour task")
