"""
Execution Monitor - 执行监控器

包装controller，监控执行过程：
1. 检测每个原子动作是否成功
2. 记录场景状态（每N帧）
3. 记录失败信息
4. 失败时强制终止
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

# 尝试导入轨迹记录器
try:
    from agent.action.rating.trajectory_recorder import TrajectoryRecorder
    TRAJECTORY_RECORDING_AVAILABLE = True
except ImportError:
    TRAJECTORY_RECORDING_AVAILABLE = False
    TrajectoryRecorder = None


class ExecutionMonitor:
    """执行监控器 - 包装controller并记录执行过程"""
    
    def __init__(
        self,
        controller,
        log_file: str,
        frame_interval: int = 10,
        enable_verification: bool = True,
        strict_mode: bool = True,
        exit_on_failure: bool = True  # 新增：控制失败时是否退出程序
    ):
        self.controller = controller
        self.log_file = Path(log_file)
        self.frame_interval = frame_interval
        self.enable_verification = enable_verification
        self.strict_mode = strict_mode
        self.exit_on_failure = exit_on_failure  # True: 单次执行模式，False: 多episode采集模式
        
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(self.log_file, 'a', encoding='utf-8')
        
        self.frame_count = 0
        self.last_stage = -1
        self.stage_start_time = None
        self.current_action_type = None
        self.last_action_type = None  # 缓存上一个阶段的动作类型，用于阶段结束验证
        self.grasped_object_name = None
        self.initial_object_positions = {}
        self.force_terminate = False
        self.action_failed = 0
        
        self.trajectory_recorder = None
        if TRAJECTORY_RECORDING_AVAILABLE:
            self.trajectory_recorder = TrajectoryRecorder(frame_interval=1)
        
        self._log(f"=== Monitor Initialized: {controller.__class__.__name__} ===")

    def _log(self, text: str):
        """写入日志"""
        try:
            self.log_handle.write(text + '\n')
            self.log_handle.flush()
        except Exception as e:
            print(f"[Monitor] Error writing log: {e}")

    def _get_current_stage(self) -> int:
        return getattr(self.controller, '_event', -1)

    def _get_action_type(self) -> str:
        """根据控制器状态推断动作类型"""
        # 1. 优先使用控制器显式设置的动作类型
        explicit_type = getattr(self.controller, 'current_action_type', None)
        if explicit_type:
            # self._log(f"[Monitor] Using explicit action type: {explicit_type}")
            return explicit_type.lower()

        # 2. 备选：根据原子控制器状态推断
        actions = ["pick", "place", "press", "pressz", "pour", "stir", "open", "close", "shake"]
        for action in actions:
            ctrl = getattr(self.controller, f"{action}_controller", None)
            if ctrl and not ctrl.is_done():
                self._log(f"[Monitor] Inferred action type from controller state: {action}")
                return action
        
        # 3. 备选：根据类名推断
        name = self.controller.__class__.__name__.lower()
        for action in actions:
            if action in name: 
                self._log(f"[Monitor] Inferred action type from class name: {action}")
                return action
        return "unknown"

    def _record_scene_state(self, state: Dict):
        """记录场景状态"""
        parts = []
        gripper_pos = state.get('gripper_position')
        if isinstance(gripper_pos, np.ndarray):
            parts.append(f"gripper: ({gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f})")
            if self.trajectory_recorder:
                self.trajectory_recorder.record(gripper_pos)
        
        for k, v in state.items():
            if k.endswith('_position') and not k.startswith(('joint', 'gripper')) and isinstance(v, np.ndarray):
                parts.append(f"{k.replace('_position', '')}: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")
        
        if parts:
            self._log(f"[Frame {self.frame_count}] {', '.join(parts)}")

    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if self.force_terminate: return None, True, False
        
        self.frame_count += 1
        
        # --- 1. 执行控制器逻辑 ---
        try:
            action, done, success = self.controller.step(state)
        except Exception as e:
            self._log(f"[Exception] {e}")
            self._log("ACTION_FAILED=1")
            self.action_failed = 1
            
            if self.exit_on_failure:
                # 单次执行模式：异常时退出程序
                sys.exit(1)
            else:
                # 多episode采集模式：返回失败状态，允许继续
                return None, True, False  # done=True, success=False

        # 记录场景状态
        if self.frame_count % self.frame_interval == 0:
            self._record_scene_state(state)
        
        current_stage = self._get_current_stage()
        
        # --- 2. 检测阶段切换 ---
        if current_stage != self.last_stage:
            # 处理旧阶段的结束验证
            if self.last_stage >= 0 and self.enable_verification:
                # 使用缓存的、确保是该阶段正确的动作类型进行验证
                prev_action_type = self.last_action_type or "unknown"
                success_prev_stage, error = self._verify_action(self.last_stage, state, prev_action_type)
                self._record_stage_end(self.last_stage, prev_action_type, success_prev_stage, error)
                
                if not success_prev_stage:
                    self.action_failed = 1
                    if self.strict_mode:
                        self._terminate_simulation(self.last_stage, error)
                        return None, True, False

            # 初始化新阶段记录状态
            if current_stage >= 0:
                self.stage_start_time = time.time()
                self.current_action_type = None # 强制重置，等待下一行重新获取
                # 记录初始位置
                self.initial_object_positions = {}
                for k, v in state.items():
                    if k.endswith('_position') and isinstance(v, np.ndarray):
                        self.initial_object_positions[k.replace('_position', '')] = v.copy()
            
            self.last_stage = current_stage
            self.last_action_type = None # 清除旧的缓存
            self.current_action_type = None
            
            # 强制清除控制器的“粘性”标签属性，防止新阶段在第一帧误用旧标签
            if hasattr(self.controller, 'current_action_type'):
                try:
                    self.controller.current_action_type = None
                except:
                    pass # 防止某些实现将该属性设为只读的情况

        # --- 3. 获取并记录当前阶段的动作类型 ---
        # 如果当前没有动作标签，尝试获取一个。只有获取到标签后才记录 START
        if current_stage >= 0 and self.last_action_type is None:
            new_type = self._get_action_type()
            # 允许记录 unknown，但如果有更好的标签就使用它
            if new_type:
                self.current_action_type = new_type
                self.last_action_type = new_type # 锁定本阶段动作类型
                self._log(f"[Stage {current_stage} - {self.current_action_type}] START")
            elif self.frame_count > 100: # 兜底：如果 100 帧都没拿到标签
                self.current_action_type = "unknown"
                self.last_action_type = "unknown"
                self._log(f"[Stage {current_stage} - unknown] START (Timeout waiting for tag)")

        if done:
            if self.enable_verification:
                action_type = self.current_action_type or "unknown"
                success, error = self._verify_action(current_stage, state, action_type)
                self._record_stage_end(current_stage, action_type, success, error)
                if not success: self.action_failed = 1
            
            # 如果检测到任何失败，覆盖controller返回的success
            if self.action_failed:
                success = False
            
            self._log(f"=== Task Complete: {'SUCCESS' if success else 'FAILED'} - Frames: {self.frame_count} ===")
            self._log(f"ACTION_FAILED={self.action_failed}")
            
            if self.trajectory_recorder:
                path = self.log_file.parent / "trajectory.json"
                self.trajectory_recorder.save(str(path))
            
            if self.exit_on_failure:
                # 单次执行模式：任务完成后退出程序
                sys.exit(0 if success else 1)
            # else: 多episode采集模式：返回状态，让外部循环继续
        
        return action, done, success

    def _verify_action(self, stage: int, state: Dict, action_type: str) -> Tuple[bool, Optional[Dict]]:
        """动作验证分发器"""
        if hasattr(self.controller, '_check_phase_success'):
            try: return self.controller._check_phase_success(), getattr(self.controller, 'last_error_info', None)
            except: pass
            
        if action_type == "pick": return self._verify_pick(state)
        if action_type == "place": return self._verify_place(state)
        return True, None

    def _infer_object(self, state: Dict, action_type: str) -> Optional[str]:
        """推断当前操作的物体"""
        if action_type == "place" and self.grasped_object_name:
            return self.grasped_object_name
            
        target_pos = getattr(self.controller, 'current_target_position', None)
        if target_pos is not None:
            for k, v in state.items():
                if k.endswith('_position') and isinstance(v, np.ndarray):
                    if np.allclose(v, target_pos, atol=1e-4):
                        name = k.replace('_position', '')
                        if name not in ['gripper', 'table', 'GroundPlane']: return name
        
        return getattr(self.controller, 'target_object_name', None)

    def _verify_pick(self, state: Dict) -> Tuple[bool, Optional[Dict]]:
        obj_name = self._infer_object(state, "pick")
        if not obj_name: 
            self._log("[Verify Pick] WARNING: Cannot infer object name")
            return True, None
        
        self._log(f"[Verify Pick] Object: {obj_name}")
        gripper_pos = state.get('gripper_position')
        # 优先使用 grisp_position 子节点，如果没有则用重心 position
        target_pos = state.get(f"{obj_name}_grisp_position")
        target_type = "grisp_position"
        if target_pos is None: 
            target_pos = state.get(f"{obj_name}_position")
            target_type = "position"
        
        if gripper_pos is not None and target_pos is not None:
            dist = np.linalg.norm(gripper_pos - target_pos)
            threshold = 0.10
            self._log(f"[Verify Pick] Gripper: ({gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f})")
            self._log(f"[Verify Pick] Target ({target_type}): ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
            self._log(f"[Verify Pick] Result: Distance={dist:.4f}m, Expected: <{threshold}m")
            if dist < threshold:
                self.grasped_object_name = obj_name
                return True, None
            return False, {"error_type": "grasp_failed", "dist": float(dist), "target": obj_name}
        return True, None

    def _verify_place(self, state: Dict) -> Tuple[bool, Optional[Dict]]:
        obj_name = self._infer_object(state, "place")
        if not obj_name: 
            self._log("[Verify Place] WARNING: Cannot infer object name")
            return True, None
        
        self._log(f"[Verify Place] Object: {obj_name}")
        obj_pos = state.get(f"{obj_name}_position")
        if obj_pos is None: 
            self._log(f"[Verify Place] ERROR: Missing position for {obj_name}")
            return False, {"error_type": "missing_pos", "object": obj_name}
        
        # 寻找放置目标点
        target_pos = None
        target_name = "unknown"
        for suffix in ['_place_position', '_position']:
            for cand in ['HeatingPlate', 'TargetPlatform1', 'table']:
                target_pos = state.get(f"{cand}{suffix}")
                if target_pos is not None: 
                    target_name = f"{cand}{suffix}"
                    break
            if target_pos is not None: break
        
        if target_pos is not None:
            dist = np.linalg.norm(obj_pos - target_pos)
            threshold = 0.08
            self._log(f"[Verify Place] Object: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
            self._log(f"[Verify Place] Target ({target_name}): ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
            self._log(f"[Verify Place] Result: Distance={dist:.4f}m, Expected: <{threshold}m")
            if dist < threshold:
                self.grasped_object_name = None
                return True, None
            return False, {"error_type": "place_failed", "dist": float(dist), "object": obj_name}
        
        # 兜底判断：物体是否发生了位移
        init_pos = self.initial_object_positions.get(obj_name)
        if init_pos is not None:
            move_dist = np.linalg.norm(obj_pos - init_pos)
            threshold = 0.10
            self._log(f"[Verify Place] No target found, using movement: {move_dist:.4f}m, Threshold: >{threshold}m")
            if move_dist > threshold:
                self.grasped_object_name = None
                return True, None
        return True, None

    def _record_stage_end(self, stage: int, action_type: str, success: bool, error: Optional[Dict]):
        dur = time.time() - (self.stage_start_time or time.time())
        status = "SUCCESS" if success else f"FAILED - {error}"
        self._log(f"[Stage {stage} - {action_type}] {status} ({dur:.2f}s)")

    def _terminate_simulation(self, stage: int, error: Any):
        self.force_terminate = True
        self._log(f"!!! FORCED TERMINATION - Stage {stage} failed: {error}")
        self._log(f"ACTION_FAILED=1")
        sys.exit(1)

    def reset(self):
        self.frame_count = 0
        self.last_stage = -1
        self.current_action_type = None
        self.last_action_type = None
        self.initial_object_positions = {}
        self.force_terminate = False
        self.action_failed = 0
        self._log("--- Monitor Reset ---")
        if hasattr(self.controller, 'reset'): self.controller.reset()

    def __getattr__(self, name):
        return getattr(self.controller, name)
