"""场景提取器：从 USD 文件提取 /World 下直接子元素的位置和 bounding box 信息"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.isaacsim_runtime import prepare_isaacsim_argv

DEFAULT_SCENES_DIR = Path(__file__).resolve().parents[1] / "scenes"


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description="场景提取器：从 USD 文件提取 /World 下直接子元素的位置和 bounding box"
    )
    parser.add_argument(
        "usd_file",
        type=str,
        help="USD 文件名（在 scenes 目录下）",
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "scenes"),
        help="场景目录路径",
    )
    return parser.parse_known_args(argv)


# 初始化 Isaac Sim 环境以使用 pxr 库
# 注意：如果从 scene_initializer 调用，SimulationApp 应该已经初始化
# 这里使用延迟检查，避免在模块导入时立即初始化
simulation_app = None

def _get_or_create_simulation_app():
    """获取或创建 SimulationApp 实例（延迟初始化）"""
    global simulation_app
    if simulation_app is not None:
        return simulation_app
    
    # 检查 scene_initializer 模块是否已初始化（使用 sys.modules 避免循环导入）
    try:
        # 延迟导入，避免循环导入问题
        import importlib
        if 'agent.scene.scene_initializer' in sys.modules:
            si_module = sys.modules['agent.scene.scene_initializer']
            if hasattr(si_module, '_simulation_app') and si_module._simulation_app is not None:
                simulation_app = si_module._simulation_app
                return simulation_app
    except Exception as e:
        # 忽略错误，继续创建新实例
        pass
    
    # 如果没有找到已存在的实例，创建新的（仅在直接运行此模块时）
    try:
        prepare_isaacsim_argv()
        from isaacsim import SimulationApp
        simulation_config = {"headless": True}
        simulation_app = SimulationApp(simulation_config)
    except ImportError:
        print("警告：未找到 Isaac Sim，尝试直接导入 pxr 库...", file=sys.stderr)
        simulation_app = None
    
    return simulation_app

_cli_args = None

# 仅在直接运行此模块时初始化（通过 __main__ 检查）
if __name__ == "__main__":
    _cli_args, _kit_args = parse_cli_args()
    prepare_isaacsim_argv(_kit_args)
    _get_or_create_simulation_app()

# 延迟导入 pxr，确保 SimulationApp 已初始化
# 如果从 scene_initializer 调用，SimulationApp 应该已经初始化
try:
    from pxr import Usd, UsdGeom, Gf
except ImportError:
    print("错误：未找到 'pxr' (OpenUSD) 库。", file=sys.stderr)
    print("请确保在 Isaac Sim 环境中运行，或已正确安装 OpenUSD 及其 Python 绑定。", file=sys.stderr)
    # 注意：不在这里关闭 simulation_app，因为它可能由 scene_initializer 管理
    # 仅在直接运行此模块时才关闭
    if __name__ == "__main__" and simulation_app:
        simulation_app.close()
    if __name__ == "__main__":
        sys.exit(1)


def convert_to_list(data):
    """递归转换 Gf.* 类型以便 JSON 序列化"""
    if isinstance(data, (Gf.Vec3d, Gf.Vec3f, Gf.Vec3h)):
        return [data[0], data[1], data[2]]
    if isinstance(data, (Gf.Quatd, Gf.Quatf, Gf.Quath)):
        return [data.GetReal(), *data.GetImaginary()]
    if isinstance(data, dict):
        return {k: convert_to_list(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_list(i) for i in data]
    return data


def get_world_transform(prim, time=Usd.TimeCode.Default()):
    """获取 Prim 的世界坐标变换"""
    xformable = UsdGeom.Xformable(prim)
    world_matrix = xformable.ComputeLocalToWorldTransform(time)
    transform = Gf.Transform()
    transform.SetMatrix(world_matrix)
    return transform.GetTranslation()


def get_bounding_box(prim, time=Usd.TimeCode.Default()):
    """获取 Prim 的 bounding box（世界坐标）
    
    注意：ComputeWorldBound 会递归计算所有子节点和引用内容的 bounding box。
    对于包含大量子节点或引用的系统对象（如 table），这可能导致异常大的 bounding box。
    """
    bbox_cache = UsdGeom.BBoxCache(time, includedPurposes=[UsdGeom.Tokens.default_])
    
    try:
        # ComputeWorldBound 会递归计算所有子节点和引用内容的 bounding box
        # 对于系统对象（如 table），这可能包含非常大的几何体或单位不正确的引用
        world_bbox = bbox_cache.ComputeWorldBound(prim)
        if not world_bbox.GetRange().IsEmpty():
            range_obj = world_bbox.GetRange()
            min_point = range_obj.GetMin()
            max_point = range_obj.GetMax()
            
            world_center = (min_point + max_point) / 2.0
            world_size = max_point - min_point
            
            return {
                "min": [min_point[0], min_point[1], min_point[2]],
                "max": [max_point[0], max_point[1], max_point[2]],
                "center": [world_center[0], world_center[1], world_center[2]],
                "size": [world_size[0], world_size[1], world_size[2]]
            }
    except Exception:
        pass
    
    # 如果 ComputeWorldBound 失败，尝试使用 Boundable API
    try:
        boundable = UsdGeom.Boundable(prim)
        local_bounds = boundable.ComputeLocalBound(time, UsdGeom.Tokens.default_)
        if not local_bounds.GetRange().IsEmpty():
            # 将局部坐标转换为世界坐标
            xformable = UsdGeom.Xformable(prim)
            world_matrix = xformable.ComputeLocalToWorldTransform(time)
            
            local_min = local_bounds.GetRange().GetMin()
            local_max = local_bounds.GetRange().GetMax()
            
            # 转换到世界坐标
            world_min = world_matrix.Transform(local_min)
            world_max = world_matrix.Transform(local_max)
            
            world_center = (world_min + world_max) / 2.0
            world_size = world_max - world_min
            
            return {
                "min": [world_min[0], world_min[1], world_min[2]],
                "max": [world_max[0], world_max[1], world_max[2]],
                "center": [world_center[0], world_center[1], world_center[2]],
                "size": [world_size[0], world_size[1], world_size[2]]
            }
    except Exception:
        pass
    
    return {
        "min": [0, 0, 0],
        "max": [0, 0, 0],
        "center": [0, 0, 0],
        "size": [0, 0, 0]
    }


class SceneExtractor:
    """场景提取器类"""
    
    def __init__(self, scenes_dir: str = str(DEFAULT_SCENES_DIR)):
        self.scenes_dir = Path(scenes_dir)
        if not self.scenes_dir.exists():
            raise ValueError(f"场景目录不存在: {self.scenes_dir}")
    
    def extract_from_usd(self, usd_file_path: Path) -> Dict:
        """从 USD 文件提取场景信息（不进行单位转换）
        
        Args:
            usd_file_path: USD 文件路径
        """
        if not usd_file_path.exists():
            raise FileNotFoundError(f"USD 文件不存在: {usd_file_path}")
        
        try:
            # 使用 LoadAll 确保所有引用和 payload 都被加载
            stage = Usd.Stage.Open(str(usd_file_path), load=Usd.Stage.LoadAll)
        except Exception as e:
            raise RuntimeError(f"无法打开 USD 文件: {e}")
        
        if not stage:
            raise RuntimeError("无法加载 Stage")
        
        # 确保所有 Prim 都被加载
        stage.Load()
        
        # 强制加载所有引用
        for prim in stage.Traverse():
            if prim.HasAuthoredReferences():
                prim.Load()
        
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim:
            raise RuntimeError("未找到 /World Prim")
        
        default_time = Usd.TimeCode.Default()
        scene_data = {}
        
        excluded_types = {"Scope", "PhysicsScene", "CylinderLight", "DomeLight", "DistantLight", "RectLight", "DiskLight", "SphereLight"}
        excluded_names = {"Looks", "PhysicsScene", "FumeHood"}
        # 排除场景文件（通常以 lab_ 开头，或者是场景引用）
        excluded_patterns = {"lab_", "Lab_", "LAB_"}
        
        print(f"开始提取 /World 下的直接子元素...")
        
        for prim in world_prim.GetChildren():
            if not prim.IsActive():
                continue
            
            prim_path = prim.GetPath().pathString
            prim_name = prim.GetName()
            prim_type = prim.GetTypeName()
            
            # 过滤系统物体
            if prim_type in excluded_types or prim_name in excluded_names:
                print(f"  跳过系统物体: {prim_path} ({prim_type})")
                continue
            
            # 过滤场景文件（检查名称是否匹配场景文件模式，且通常是引用）
            is_scene_file = False
            for pattern in excluded_patterns:
                if prim_name.startswith(pattern):
                    is_scene_file = True
                    break
            
            # 如果名称看起来像场景文件，或者有引用且名称可疑，则跳过
            if is_scene_file or (prim.HasAuthoredReferences() and any(pattern in prim_name for pattern in excluded_patterns)):
                print(f"  跳过场景文件: {prim_path} ({prim_type})")
                continue
            
            print(f"  处理: {prim_path}")
            
            prim_data = {
                "prim_path": prim_path,
                "prim_name": prim_name,
                "prim_type": prim_type
            }
            
            # 获取位置（世界坐标，原始数据，不转换单位）
            try:
                position = get_world_transform(prim, default_time)
                prim_data["position"] = [position[0], position[1], position[2]]
            except Exception as e:
                print(f"    警告：无法获取位置: {e}")
                prim_data["position"] = [0, 0, 0]
            
            # 获取 bounding box（世界坐标，原始数据，不转换单位）
            try:
                bbox = get_bounding_box(prim, default_time)
                
                # 对于 ErlenmeyerFlask_Liquid 和 ErlenmeyerFlask_02 设备，bounding box 需要除以 100（单位修正）
                if prim_name.startswith("ErlenmeyerFlask_Liquid") or prim_name == "ErlenmeyerFlask_02":
                    bbox = {
                        "min": [v / 100.0 for v in bbox["min"]],
                        "max": [v / 100.0 for v in bbox["max"]],
                        "center": [v / 100.0 for v in bbox["center"]],
                        "size": [v / 100.0 for v in bbox["size"]]
                    }
                
                # 对于 ElectronicScale 设备，bounding box 需要除以 1000（单位修正）
                if "ElectronicScale" in prim_name:
                    bbox = {
                        "min": [v / 1000.0 for v in bbox["min"]],
                        "max": [v / 1000.0 for v in bbox["max"]],
                        "center": [v / 1000.0 for v in bbox["center"]],
                        "size": [v / 1000.0 for v in bbox["size"]]
                    }
                
                prim_data["bounding_box"] = bbox
            except Exception as e:
                print(f"    警告：无法获取 bounding box: {e}")
                prim_data["bounding_box"] = {
                    "min": [0, 0, 0],
                    "max": [0, 0, 0],
                    "center": [0, 0, 0],
                    "size": [0, 0, 0]
                }
            
            scene_data[prim_path] = prim_data
        
        return scene_data
    
    def extract(self, usd_filename: str) -> Optional[Path]:
        """提取场景信息并保存为 JSON
        
        Args:
            usd_filename: USD 文件名
        """
        usd_file_path = self.scenes_dir / usd_filename
        
        if not usd_file_path.exists():
            print(f"错误：USD 文件不存在: {usd_file_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f"正在提取场景: {usd_filename}")
        print(f"{'='*60}\n")
        
        try:
            scene_data = self.extract_from_usd(usd_file_path)
        except Exception as e:
            print(f"错误：提取失败: {e}")
            return None
        
        # 生成输出 JSON 文件名（与 USD 文件同名）
        json_filename = usd_file_path.stem + ".json"
        json_file_path = self.scenes_dir / json_filename
        
        # 转换为可序列化的格式
        json_data = convert_to_list(scene_data)
        
        # 保存 JSON 文件
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            print(f"\n✓ 成功保存到: {json_file_path}")
            print(f"  共提取 {len(scene_data)} 个物体")
            return json_file_path
        except Exception as e:
            print(f"✗ 保存 JSON 文件失败: {e}")
            return None


def extract_scene(
    usd_filename: str,
    scenes_dir: str = str(DEFAULT_SCENES_DIR)
) -> Optional[Path]:
    """便捷函数：提取场景信息（不进行单位转换）
    
    Args:
        usd_filename: USD 文件名
        scenes_dir: 场景目录
    """
    extractor = SceneExtractor(scenes_dir=scenes_dir)
    return extractor.extract(usd_filename)


if __name__ == "__main__":
    args = _cli_args
    
    result = extract_scene(args.usd_file, args.scenes_dir)
    
    if result:
        print(f"\n{'='*60}")
        print(f"提取完成！")
        print(f"输出文件: {result}")
        print(f"{'='*60}\n")
        if simulation_app:
            simulation_app.close()
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print(f"提取失败！")
        print(f"{'='*60}\n")
        if simulation_app:
            simulation_app.close()
        sys.exit(1)
