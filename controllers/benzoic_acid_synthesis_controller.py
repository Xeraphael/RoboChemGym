from typing import Optional, Tuple, Any, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R
from controllers.base_controller import BaseController
from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from datetime import datetime
import os


class BenzoicAcidSynthesisController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        # 初始化各控制器（只保留pick和place）
        self.pick_controller = PickController(name="pick", cspace_controller=self.rmp_controller)
        self.place_controller = PlaceController(name="place", cspace_controller=self.rmp_controller, gripper=robot.gripper)
        
        # 初始化状态机
        self._event = 0
        self.reset_needed = False
        self._final_joint_positions = None  # 用于延迟数据写入
        
        # 初始化调试日志文件
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"gripper_debug_{timestamp}.txt")
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        self._log(f"调试日志开始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"日志文件路径: {self.log_file_path}\n")
        # 在控制台也输出一次日志位置，方便用户查找
        print(f"\n{'='*70}")
        print(f"[夹爪调试] 日志文件已创建: {self.log_file_path}")
        print(f"{'='*70}\n")
    
    def _log(self, message):
        """写入日志文件"""
        self.log_file.write(message + '\n')
        self.log_file.flush()  # 立即写入磁盘
    
    def __del__(self):
        """析构函数，确保关闭日志文件"""
        if hasattr(self, 'log_file') and self.log_file:
            self._log(f"\n调试日志结束 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_file.close()

    def reset(self):
        super().reset()
        self._event = 0
        self.reset_needed = False
        self._final_joint_positions = None
        self.pick_controller.reset()
        self.place_controller.reset()

    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        # ===== 调试：记录夹爪状态 =====
        self._log(f"\n{'='*60}")
        self._log(f"[DEBUG] 当前阶段: Event {self._event}")
        # 使用正确的属性名
        if hasattr(self.gripper_control, 'grasped_object_path'):
            self._log(f"[DEBUG] 夹爪抓取的物体路径: {self.gripper_control.grasped_object_path}")
        if hasattr(self.gripper_control, 'gripper_frame_path'):
            self._log(f"[DEBUG] 夹爪框架路径: {self.gripper_control.gripper_frame_path}")
        if hasattr(self.gripper_control, 'position_offest'):
            self._log(f"[DEBUG] 位置偏移: {self.gripper_control.position_offest}")
        self._log(f"{'='*60}\n")
        # ===== 调试结束 =====
        
        self.state = state
        
        # 第1阶段：抓取装有苯甲酸的锥形瓶 ErlenmeyerFlask_Solid1
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
                    object_size=state.get(f"{obj_name}_size", np.array([0.04, 0.04, 0.12])),
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                    gripper_distances=0.012  # 特殊设置，ErlenmeyerFlask_Solid 需要
                )
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
                self._log(f"[DEBUG] *** 阶段1完成，准备进入阶段2 ***")
                self.place_controller.reset()
                self._event = 1
                return None, False, False
        
        # 第2阶段：将锥形瓶放到加热板上（最后阶段）
        elif self._event == 1:
            self.current_action_type = "place"
            if not self.place_controller.is_done():
                target_name = "HeatingPlate"
                pos = state.get(f"{target_name}_place_position", state.get(f"{target_name}_position"))
                if pos is None:
                    return None, False, False
                
                self.current_target_position = pos
                action = self.place_controller.forward(
                    place_position=pos,
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    place_offset_z=0.12  # 特殊设置，ErlenmeyerFlask_Solid 需要
                )
                
                # 缓存数据
                if self.mode == "collect" and 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        language_instruction=self.get_language_instruction()
                    )
                
                return action, False, False
            else:
                # 所有步骤完成
                self._log(f"[DEBUG] *** 阶段2完成（放置固体瓶到加热板），所有流程结束 ***")
                
                # 标记任务成功完成（数据写入将在外部确认成功后进行）
                self._last_success = True
                self.reset_needed = True
                
                # 保存最终状态用于后续数据写入
                if self.mode == "collect":
                    self._final_joint_positions = state['joint_positions'][:-1]
                
                self.gripper_control.release_object()
                self._log(f"[DEBUG] 已清理夹爪状态")
                return None, True, True
        
        # 默认情况
        return None, False, False