import numpy as np
from typing import Dict, Any, List, Optional
from .base_task import BaseTask
try:
    # 用于在引用层受限时强制写入世界位姿
    from pxr import Usd, UsdGeom, Gf  # type: ignore
except Exception:
    Usd = None
    UsdGeom = None
    Gf = None

class AllObjectsTask(BaseTask):
    """
    包含场景中所有物体的Task类。
    该Task会初始化并跟踪配置文件中定义的所有物体，
    在step中返回所有物体的位置、大小等信息。
    
    使用场景：
    - 需要获取场景中所有物体状态的任务
    - 多物体交互任务
    - 场景感知任务
    """
    
    def __init__(self, cfg, world, stage, robot):
        """
        初始化AllObjectsTask。
        
        Args:
            cfg: 配置对象，必须包含task.obj_paths配置
            world: 仿真世界实例
            stage: USD stage实例
            robot: 机器人实例
        """
        super().__init__(cfg, world, stage, robot)
        
        # 存储所有物体的路径和配置（基于BaseTask的obj_configs）
        self.object_paths = []
        self.object_configs = {}
        # 原始顶层路径 -> 包装器路径 映射
        self.path_redirect: Dict[str, str] = {}
        
        def _get_parent_path(path: str):
            parts = [p for p in path.split("/") if p]
            if len(parts) <= 1:
                return None
            return "/" + "/".join(parts[:-1])
        def _is_child_node(path: str) -> bool:
            last = path.split("/")[-1]
            return (
                last in ['mesh', 'Cylinder', 'button', 'button_2', 'button_3', 'plat']
                or last.startswith('mesh_')
            )
        def _normalize_to_top_xform(path: str) -> str:
            return _get_parent_path(path) if _is_child_node(path) and _get_parent_path(path) else path

        # 从BaseTask的obj_configs中提取物体路径和配置
        for obj_config in self.obj_configs:
                if isinstance(obj_config, dict):
                    obj_path = _normalize_to_top_xform(obj_config['path'])
                    self.object_paths.append(obj_path)
                    # 规范化后的路径也写回配置，保持读写一致
                    new_cfg = dict(obj_config)
                    new_cfg['path'] = obj_path
                    self.object_configs[obj_path] = new_cfg
                elif isinstance(obj_config, str):
                    # 如果只是字符串路径，使用默认配置，并规范化为顶层 Xform
                    obj_path = _normalize_to_top_xform(obj_config)
                    self.object_paths.append(obj_path)
                    self.object_configs[obj_path] = {
                        'path': obj_path,
                        'position_range': {
                            'x': [0.24, 0.30],
                            'y': [-0.05, 0.05],
                            'z': [0.85, 0.85]
                        }
                    }
        # 去重，保持顺序
        self.object_paths = list(dict.fromkeys(self.object_paths))
        
        # 一视同仁：不注入任何基于任务类型的额外路径，仅以 cfg.task.obj_paths 为准
        
    def reset(self):
        """
        重置任务状态。
        初始化机器人位置并随机化所有物体的位置（如果配置了position_range）。
        """
        super().reset()
        self.robot.initialize()
        
        # 自愈：确保初始化阶段创建了属性
        if not hasattr(self, 'object_paths') or self.object_paths is None:
            self.object_paths = []
        if not hasattr(self, 'object_configs') or self.object_configs is None:
            self.object_configs = {}
        if self.scene_randomizer is not None:
            return
        # 若对象列表为空，基于 BaseTask 的 obj_configs 重建一次
        if not self.object_paths and hasattr(self, 'obj_configs'):
            for obj_config in self.obj_configs:
                if isinstance(obj_config, dict):
                    obj_path = obj_config.get('path')
                    if obj_path:
                        self.object_paths.append(obj_path)
                        self.object_configs[obj_path] = obj_config
                elif isinstance(obj_config, str):
                    self.object_paths.append(obj_config)
                    self.object_configs[obj_config] = {
                        'path': obj_config,
                        'position_range': {
                            'x': [0.24, 0.30],
                            'y': [-0.05, 0.05],
                            'z': [0.85, 0.85]
                        }
                    }
        # 在 reset 阶段再次确保路径为顶层 Xform
        def _get_parent_path(path: str):
            parts = [p for p in path.split("/") if p]
            if len(parts) <= 1:
                return None
            return "/" + "/".join(parts[:-1])
        def _is_child_node(path: str) -> bool:
            last = path.split("/")[-1]
            return (
                last in ['mesh', 'Cylinder', 'button', 'button_2', 'button_3', 'plat']
                or last.startswith('mesh_')
            )
        def _normalize_to_top_xform(path: str) -> str:
            return _get_parent_path(path) if _is_child_node(path) and _get_parent_path(path) else path
        self.object_paths = [ _normalize_to_top_xform(p) for p in self.object_paths ]
        self.object_paths = list(dict.fromkeys(self.object_paths))
        self.object_configs = { _normalize_to_top_xform(k):
            (dict(v, path=_normalize_to_top_xform(v.get('path', k))) if isinstance(v, dict) else v)
            for k, v in self.object_configs.items()
        }
        
        # 为所有配置了position_range的物体随机化位置
        # 子节点（mesh/Cylinder/button/plat/mesh_*）随机化时改为作用在父 Xform 上
        def _get_xform_position(path: str) -> Optional[np.ndarray]:
            try:
                pos = self.object_utils.get_object_xform_position(object_path=path)
                return pos
            except Exception:
                return None

        # 还原到简单策略：不创建包装器，不做强制传送

        # 仅对关键可交互顶层对象设位；不搬动大环境
        def _is_interactable_top_name(name: str) -> bool:
            return (
                name.startswith('beaker')
                or name.startswith('conical_bottle')
                or name.startswith('graduated_cylinder')
                or name.startswith('target_plat')
                or name == 'target_beaker'
                or name == 'glass_rod'
                or name.startswith('DryingBox')
                or name == 'MuffleFurnace'
            )

        for obj_path in self.object_paths:
            obj_config = self.object_configs.get(obj_path, {})
            if 'position_range' not in obj_config:
                continue

            # 目标顶层路径与名称
            # 路径已在初始化与上方规范化为顶层 Xform
            target_path = obj_path
            target_name = target_path.split('/')[-1]

            # 跳过明显的大环境对象
            if target_name in ['GroundPlane'] or target_name.startswith('Cabinet') or target_name.startswith('DryingBox') or target_name.startswith('MuffleFurnace') or target_name in ['lab_015', 'table', 'lounge_booth_table', 'CylinderLight', 'ParticleSystem']:
                continue
            if not _is_interactable_top_name(target_name):
                continue

            try:
                position_range = obj_config['position_range']
                rx = float(np.random.uniform(position_range['x'][0], position_range['x'][1]))
                ry = float(np.random.uniform(position_range['y'][0], position_range['y'][1]))
                rz = float(np.random.uniform(position_range['z'][0], position_range['z'][1]))
                target_pos = np.array([rx, ry, rz])

                # 先常规写入并检测
                before = _get_xform_position(target_path)
                self.object_utils.set_object_position(object_path=target_path, position=target_pos)
                after = _get_xform_position(target_path)
                if after is None or (before is not None and np.allclose(after, before, atol=1e-4)):
                    # 常规写失败 -> 使用 SessionLayer Override 强制写入
                    try:
                        if Usd is not None and UsdGeom is not None and Gf is not None:
                            session_layer = self.stage.GetSessionLayer()
                            prev_target = self.stage.GetEditTarget()
                            self.stage.SetEditTarget(Usd.EditTarget(session_layer))
                            prim = self.stage.GetPrimAtPath(target_path)
                            if not prim or not prim.IsValid():
                                # 如果 prim 不存在，创建一个 Override
                                prim = self.stage.OverridePrim(target_path)
                            else:
                                # 即便存在，也创建 Override 以在 SessionLayer 写意见
                                self.stage.OverridePrim(target_path)
                            xformable = UsdGeom.Xformable(prim)
                            try:
                                xformable.ClearXformOpOrder()
                            except Exception:
                                pass
                            op = xformable.AddTranslateOp()
                            op.Set(Gf.Vec3d(float(rx), float(ry), float(rz)))
                            # 恢复原 EditTarget
                            self.stage.SetEditTarget(prev_target)
                            # 再次读取确认
                            after2 = _get_xform_position(target_path)
                            if after2 is None:
                                continue
                        else:
                            continue
                    except Exception as e2:
                        print(f"Warning: SessionLayer override failed for {target_path}: {e2}")
                        continue
            except Exception as e:
                print(f"Warning: Could not randomize position for object {obj_path}: {e}")
                continue

        # 不做路径重定向，保持与原任务一致
    
    def step(self) -> Optional[Dict[str, Any]]:
        """
        执行一个仿真步骤。
        收集所有物体的状态信息并返回。
        
        Returns:
            dict: 包含所有物体状态信息的字典，如果未准备好则返回None
        """
        self.frame_idx += 1
        # 首帧预热：前若干帧不产出状态，等待USD加载稳定
        warmup_frames = 30
        if hasattr(self.cfg, 'task') and hasattr(self.cfg.task, 'warmup_frames'):
            try:
                warmup_frames = int(self.cfg.task.warmup_frames)
            except Exception:
                warmup_frames = 30
        if self.frame_idx <= max(0, warmup_frames):
            return None
        
        # 检查帧限制，使用配置中的max_steps
        max_steps = None
        if hasattr(self.cfg, 'task') and hasattr(self.cfg.task, 'max_steps'):
            max_steps = self.cfg.task.max_steps
        
        if not self.check_frame_limits(max_steps=max_steps):
            return None
        
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
        
        # 收集所有物体的信息
        all_objects_info = {}
        all_objects_positions = {}
        all_objects_sizes = {}

        def _get_position_top_xform(object_path: str) -> Optional[np.ndarray]:
            # 仅使用顶层 Xform 位姿
            try:
                return self.object_utils.get_object_xform_position(object_path=object_path)
            except Exception:
                return None
        
        for obj_path in self.object_paths:
            obj_config = self.object_configs.get(obj_path, {})
            obj_name = self._extract_object_name(obj_path, obj_config)
                
            # 分别获取位置和大小，即使一个失败也设置另一个
            position = None
            size = None
            
            # 稳健位置获取：几何中心 -> 自身Xform -> 父节点Xform
            position = _get_position_top_xform(obj_path)
                
            try:
                size = self.object_utils.get_object_size(object_path=obj_path)
            except Exception as e:
                print(f"Warning: Could not get size for object {obj_path}: {e}")
                # 设置默认小尺寸，避免控制器报错
                size = np.array([0.02, 0.02, 0.02])
                
            # 存储物体信息（即使部分失败也存储）
                all_objects_info[obj_name] = {
                    'path': obj_path,
                    'position': position,
                    'size': size,
                    'name': obj_name
                }
                
                # 为了向后兼容，也使用带下划线的键名
                all_objects_positions[f'{obj_name}_position'] = position
                all_objects_sizes[f'{obj_name}_size'] = size
                
        
        # 将所有物体信息添加到状态字典（统一结构，无主物体/特例键）
        state.update({
            'all_objects': all_objects_info,  # 结构化信息
            'object_count': len(all_objects_info),  # 物体数量
        })
        state.update(all_objects_positions)  # 展开的位置信息
        state.update(all_objects_sizes)  # 展开的大小信息
        
        return state
    
    def get_all_object_paths(self) -> List[str]:
        """
        获取所有物体的路径列表。
        
        Returns:
            List[str]: 所有物体路径的列表
        """
        return self.object_paths.copy()
    
    def _extract_object_name(self, obj_path: str, obj_config: dict = None) -> str:
        """
        从物体路径或配置中提取合适的物体名称。
        优先使用配置中的name字段，否则从路径推断。
        对于子路径（如 /World/glass_rod/Cylinder），提取父名称（glass_rod）。
        
        Args:
            obj_path: 物体路径
            obj_config: 物体配置字典（可选）
            
        Returns:
            str: 提取的物体名称
        """
        # 首先检查配置中是否有明确的名称
        if obj_config and 'name' in obj_config:
            return obj_config['name']
        
        # 从路径提取名称
        path_parts = [p for p in obj_path.split("/") if p]
        
        # 如果路径包含 /mesh 或 /Cylinder 等子节点，使用父节点名称
        if len(path_parts) >= 2:
            last_part = path_parts[-1]
            parent_part = path_parts[-2]
            
            # 常见的子节点名称
            child_nodes = ['mesh', 'Cylinder', 'button', 'button_2', 'button_3', 'plat']
            if last_part in child_nodes or last_part.startswith('mesh_'):
                return parent_part
        
        # 否则使用路径的最后一部分
        return path_parts[-1] if path_parts else obj_path.split("/")[-1]
    
    def get_object_info(self, obj_path: str) -> Optional[Dict[str, Any]]:
        """
        获取指定物体的信息。
        
        Args:
            obj_path: 物体路径
            
        Returns:
            dict: 物体信息字典，如果物体不存在则返回None
        """
        obj_config = self.object_configs.get(obj_path, {})
        obj_name = self._extract_object_name(obj_path, obj_config)
        
        position = None
        size = None
        
        # 与 step 中一致的回退策略
        try:
            position = self.object_utils.get_object_xform_position(object_path=obj_path)
        except Exception:
            position = None
        
        try:
            size = self.object_utils.get_object_size(object_path=obj_path)
        except Exception as e:
            print(f"Warning: Could not get size for object {obj_path}: {e}")
            size = np.array([0.02, 0.02, 0.02])
        
        if position is None and size is None:
            return None
        
        return {
            'path': obj_path,
            'position': position,
            'size': size,
            'name': obj_name
        }
