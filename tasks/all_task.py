import numpy as np
from typing import Dict, Any
from pxr import Usd

from agent.action.plan_execution.state_contract import anchor_state_suffix
from agent.scene.anchor_resolver import matching_anchor_prims
from .base_task import BaseTask


_LEGACY_POSITION_ANCHORS = (
    'grisp_position',
    'place_position',
    'press_position',
    'pressz_position',
)


def _state_anchor_config(cfg):
    agent_cfg = getattr(cfg, 'agent', None)
    if agent_cfg is None:
        return False, {}
    anchors = getattr(agent_cfg, 'state_anchors', None)
    if anchors is not None:
        return True, anchors
    if hasattr(agent_cfg, 'get'):
        anchors = agent_cfg.get('state_anchors', None)
        if anchors is not None:
            return True, anchors
    return False, {}


class AllTask(BaseTask):
    
    def __init__(self, cfg, world, stage, robot):
        """
        初始化 XXXTask。
        
        Args:
            cfg: 配置对象，必须包含 task.obj_paths 配置
            world: 仿真世界实例
            stage: USD stage 实例
            robot: 机器人实例
        """
        super().__init__(cfg, world, stage, robot)
        
        # 从配置中提取所有物体路径
        self.object_paths = []
        
        for obj_config in self.obj_configs:
            # 处理 OmegaConf DictConfig 对象
            if hasattr(obj_config, 'get') and 'path' in obj_config:
                # OmegaConf DictConfig 或普通字典
                self.object_paths.append(obj_config['path'])
            elif isinstance(obj_config, dict) and 'path' in obj_config:
                # 普通字典
                self.object_paths.append(obj_config['path'])
            elif isinstance(obj_config, str):
                # 字符串路径
                self.object_paths.append(obj_config)
    
    def reset(self):
        """
        重置任务状态。
        初始化机器人位置，但不对任何物体位置进行随机化。
        所有物体保持在 USD 文件中定义的原始位置。
        """
        super().reset()
        self.robot.initialize()
        self.robot.post_reset()  # 重要：确保机器人正确重置到初始姿态

        for alignment in self.cfg.task.get('placement_alignments', []):
            source_position = self.object_utils.get_object_xform_position(
                object_path=alignment['object_path']
            )
            target_position = self.object_utils.get_object_xform_position(
                object_path=alignment['target_path']
            )
            aligned_position = np.asarray(target_position).copy()
            aligned_position[:2] = np.asarray(source_position)[:2]
            self.object_utils.set_object_position(
                object_path=alignment['target_path'],
                position=aligned_position,
            )
        
        # 不做任何物体位置随机化，保持 USD 原始位置
        # 所有物体将保持在场景文件中定义的位置
    
    def step(self):
        """
        执行一个仿真步骤。
        收集所有物体的状态信息并返回。
        
        Returns:
            dict: 包含所有物体状态信息的字典，如果未准备好则返回 None
        """
        self.frame_idx += 1
        
        # 前几帧预热，等待场景稳定
        if self.frame_idx < 5:
            return None
        
        # 检查是否超过最大步数
        if hasattr(self.cfg.task, 'max_steps') and self.frame_idx > self.cfg.task.max_steps:
            self.on_task_complete(True)
            self.reset_needed = True
        
        # 获取基础状态信息（机器人、相机等）
        joint_positions = self.robot.get_joint_positions()
        if joint_positions is None:
            return None
        
        camera_data, display_data = self.get_camera_data()
        
        # 构建基础状态字典
        state = {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'gripper_position': self.robot.get_gripper_position(),
        }
        has_anchor_config, anchors_by_instance = _state_anchor_config(self.cfg)
        
        # 收集所有物体的信息
        for obj_path in self.object_paths:
            # 从路径提取物体名称
            obj_name = obj_path.split('/')[-1]
            
            # 获取物体位置和大小
            try:
                position = self.object_utils.get_geometry_center(object_path=obj_path)
            except Exception:
                try:
                    position = self.object_utils.get_object_xform_position(object_path=obj_path)
                except Exception:
                    position = None
            
            try:
                size = self.object_utils.get_object_size(object_path=obj_path)
            except Exception:
                size = np.array([0.02, 0.02, 0.02])
            
            # 添加到状态字典，使用物体名称作为键
            if position is not None:
                state[f'{obj_name}_position'] = position
                if not has_anchor_config and obj_name == 'handle':
                    state['DryingBox_handle_position'] = position
            if size is not None:
                state[f'{obj_name}_size'] = size

            if has_anchor_config:
                try:
                    instance_prim = self.stage.GetPrimAtPath(obj_path)
                except Exception:
                    instance_prim = None
                for anchor in list(anchors_by_instance.get(obj_name, [])):
                    try:
                        matches = matching_anchor_prims(
                            Usd,
                            instance_prim,
                            obj_path,
                            anchor,
                        )
                        if len(matches) != 1:
                            continue
                        prim = matches[0]
                        anchor_path = prim.GetPath().pathString
                        anchor_name = prim.GetName()
                        type_name = str(prim.GetTypeName())
                        if type_name.endswith('RevoluteJoint'):
                            value = self.object_utils.get_revolute_joint_positions(
                                joint_path=anchor_path
                            )
                        else:
                            value = self.object_utils.get_object_xform_position(
                                object_path=anchor_path
                            )
                        if value is not None:
                            state[
                                f'{obj_name}_{anchor_state_suffix(anchor_name)}'
                            ] = value
                    except Exception:
                        continue
            else:
                for anchor in _LEGACY_POSITION_ANCHORS:
                    anchor_path = f'{obj_path}/{anchor}'
                    try:
                        prim = self.stage.GetPrimAtPath(anchor_path)
                        if not prim or not prim.IsValid():
                            continue
                        value = self.object_utils.get_object_xform_position(
                            object_path=anchor_path
                        )
                        if value is not None:
                            state[f'{obj_name}_{anchor}'] = value
                    except Exception:
                        continue

        if not has_anchor_config:
            try:
                value = self.object_utils.get_revolute_joint_positions(
                    joint_path='/World/DryingBox/RevoluteJoint'
                )
                if value is not None:
                    state['revolute_joint_position'] = value
                    state['DryingBox_revolute_joint_position'] = value
            except Exception:
                pass
        
        return state
