"""位置更新器：从 JSON 文件读取位置信息并应用到 USD 文件"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.isaacsim_runtime import prepare_isaacsim_argv

DEFAULT_SCENES_DIR = Path(__file__).resolve().parents[1] / "scenes"


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description="从 JSON 文件读取位置信息并应用到 USD 文件"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="JSON 文件名（在 scenes 目录下，如 example_protocol_scene.json）或完整路径",
    )
    parser.add_argument(
        "--usd-file",
        type=str,
        default=None,
        help="USD 文件名（在 scenes 目录下，如 example_protocol_scene.usd）或完整路径。如果未指定，从 JSON 文件名推断",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="输出 USD 文件名或完整路径（如果指定，将创建新文件而不是修改原文件）",
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "scenes"),
        help="场景文件目录",
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
    from pxr import Usd, UsdGeom, Gf, Sdf
except ImportError:
    print("错误：未找到 'pxr' (OpenUSD) 库。", file=sys.stderr)
    print("请确保在 Isaac Sim 环境中运行，或已正确安装 OpenUSD 及其 Python 绑定。", file=sys.stderr)
    # 注意：不在这里关闭 simulation_app，因为它可能由 scene_initializer 管理
    # 仅在直接运行此模块时才关闭
    if __name__ == "__main__" and simulation_app:
        simulation_app.close()
    if __name__ == "__main__":
        sys.exit(1)


class PositionUpdater:
    """位置更新器类：从 JSON 文件读取位置信息并应用到 USD 文件"""
    
    def __init__(self, scenes_dir: str = str(DEFAULT_SCENES_DIR)):
        """
        初始化位置更新器
        
        Args:
            scenes_dir: 场景文件目录
        """
        self.scenes_dir = Path(scenes_dir)
        if not self.scenes_dir.exists():
            raise ValueError(f"场景目录不存在: {self.scenes_dir}")
    
    def load_json_data(self, json_file_path: Path) -> Dict:
        """
        加载 JSON 文件数据
        
        Args:
            json_file_path: JSON 文件路径
        
        Returns:
            JSON 数据字典
        """
        if not json_file_path.exists():
            raise FileNotFoundError(f"JSON 文件不存在: {json_file_path}")
        
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            print(f"✓ 成功加载 JSON 文件: {len(json_data)} 个图元")
            return json_data
        except Exception as e:
            raise RuntimeError(f"无法读取 JSON 文件: {e}")
    
    def update_prim_position(
        self,
        prim: Usd.Prim,
        position: list,
        time: Usd.TimeCode = Usd.TimeCode.Default()
    ) -> bool:
        """
        更新 Prim 的位置
        
        Args:
            prim: USD Prim 对象
            position: 位置 [x, y, z]
            time: 时间码
        
        Returns:
            是否更新成功
        """
        if not prim or not prim.IsValid():
            return False
        
        if not prim.IsA(UsdGeom.Xformable):
            print(f"  警告：{prim.GetPath()} 不是 Xformable，跳过")
            return False
        
        try:
            xformable = UsdGeom.Xformable(prim)
            
            # 获取现有的 xform 操作
            existing_ops = xformable.GetOrderedXformOps()
            
            # 查找现有的 translate 操作
            translate_op = None
            precision = UsdGeom.XformOp.PrecisionDouble
            
            for op in existing_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    # 从现有操作推断精度
                    attr = prim.GetAttribute(op.GetOpName())
                    if attr:
                        type_name = attr.GetTypeName()
                        if type_name == Sdf.ValueTypeNames.Float3:
                            precision = UsdGeom.XformOp.PrecisionFloat
                    break
            
            # 如果没有现有的 translate 操作，创建一个新的
            if not translate_op:
                # 检测精度（检查是否有其他 xformOp 作为参考）
                if existing_ops:
                    # 尝试从第一个操作推断精度
                    first_op = existing_ops[0]
                    attr = prim.GetAttribute(first_op.GetOpName())
                    if attr:
                        type_name = attr.GetTypeName()
                        if type_name == Sdf.ValueTypeNames.Float3:
                            precision = UsdGeom.XformOp.PrecisionFloat
                
                translate_op = xformable.AddTranslateOp(precision=precision)
            
            # 设置位置值
            if precision == UsdGeom.XformOp.PrecisionFloat:
                translation = Gf.Vec3f(position)
            else:
                translation = Gf.Vec3d(position)
            
            if not translate_op.Set(translation, time):
                print(f"  错误：设置 {prim.GetPath()} 位姿失败")
                return False
            return True
            
        except Exception as e:
            print(f"  错误：更新 {prim.GetPath()} 位置失败: {e}")
            return False
    
    def apply_positions_to_usd(
        self,
        json_file_path: Path,
        usd_file_path: Path,
        output_usd_path: Optional[Path] = None,
        in_place: bool = True,
        required_prim_paths: Optional[Set[str]] = None,
    ) -> bool:
        """
        将 JSON 文件中的位置信息应用到 USD 文件
        
        Args:
            json_file_path: JSON 文件路径
            usd_file_path: 输入 USD 文件路径
            output_usd_path: 输出 USD 文件路径（如果为 None，则直接修改原文件）
            in_place: 是否直接修改原文件（默认 True，直接修改原文件）
            required_prim_paths: 必须成功更新的图元路径；None 保持旧调用行为
        
        Returns:
            是否成功
        """
        # 加载 JSON 数据
        json_data = self.load_json_data(json_file_path)
        required_paths = (
            None
            if required_prim_paths is None
            else set(required_prim_paths)
        )
        if required_paths is not None and any(
            not isinstance(path, str) or not path
            for path in required_paths
        ):
            raise ValueError("required_prim_paths must contain nonempty strings")
        
        # 打开 USD 文件
        if not usd_file_path.exists():
            raise FileNotFoundError(f"USD 文件不存在: {usd_file_path}")
        
        print(f"正在打开 USD 文件: {usd_file_path}...")
        try:
            stage = Usd.Stage.Open(str(usd_file_path), load=Usd.Stage.LoadAll)
        except Exception as e:
            raise RuntimeError(f"无法打开 USD 文件: {e}")
        
        if not stage:
            raise RuntimeError("无法加载 Stage")
        
        print("✓ USD Stage 加载成功")
        
        # 确定输出路径
        if in_place:
            output_path = usd_file_path
        elif output_usd_path:
            output_path = output_usd_path
        else:
            # 默认：直接修改原文件
            output_path = usd_file_path
        
        # 更新每个 prim 的位置
        default_time = Usd.TimeCode.Default()
        updated_count = 0
        skipped_count = 0
        error_count = 0
        updated_paths = set()
        
        print(f"\n开始更新位置信息...")
        for prim_path_str, data in json_data.items():
            if "position" not in data:
                skipped_count += 1
                continue
            
            position = data["position"]
            if not isinstance(position, list) or len(position) != 3:
                print(f"  警告：{prim_path_str} 的位置格式不正确，跳过")
                skipped_count += 1
                continue
            
            prim_path = Sdf.Path(prim_path_str)
            prim = stage.GetPrimAtPath(prim_path)
            
            if not prim or not prim.IsValid():
                print(f"  警告：未找到图元 {prim_path_str}，跳过")
                skipped_count += 1
                continue
            
            if self.update_prim_position(prim, position, default_time):
                print(f"  ✓ 更新 {prim_path_str}: position = {position}")
                updated_count += 1
                updated_paths.add(prim_path_str)
            else:
                error_count += 1

        if required_paths is not None:
            unresolved_paths = required_paths - updated_paths
            if unresolved_paths:
                print(
                    "  错误：以下必需图元未成功更新: "
                    + ", ".join(sorted(unresolved_paths))
                )
                return False
        
        # 保存 USD 文件
        print(f"\n正在保存 USD 文件: {output_path}...")
        try:
            exported = stage.GetRootLayer().Export(str(output_path))
        except Exception as e:
            raise RuntimeError(f"保存 USD 文件失败: {e}")
        if not exported:
            print(f"✗ 保存 USD 文件失败: Export 返回失败状态")
            return False
        print(f"✓ 成功保存 USD 文件")
        
        # 输出统计信息
        print(f"\n{'='*60}")
        print(f"更新完成:")
        print(f"  - 成功更新: {updated_count} 个图元")
        print(f"  - 跳过: {skipped_count} 个图元")
        print(f"  - 错误: {error_count} 个图元")
        print(f"  - 输出文件: {output_path}")
        print(f"{'='*60}\n")
        
        return True
    
    def update_from_json_filename(
        self,
        json_filename: str,
        usd_filename: Optional[str] = None,
        output_filename: Optional[str] = None,
        in_place: bool = True
    ) -> Optional[Path]:
        """
        从 JSON 文件名更新对应的 USD 文件
        
        Args:
            json_filename: JSON 文件名（如 "example_protocol_scene.json"）
            usd_filename: USD 文件名（如果为 None，则从 JSON 文件名推断）
            output_filename: 输出文件名（如果为 None 且 in_place=False，则自动生成）
            in_place: 是否直接修改原文件
        
        Returns:
            输出文件路径
        """
        json_file_path = self.scenes_dir / json_filename
        if not json_file_path.exists():
            raise FileNotFoundError(f"JSON 文件不存在: {json_file_path}")
        
        # 如果没有指定 USD 文件名，从 JSON 文件名推断
        if usd_filename is None:
            # 例如：example_protocol_scene.json -> example_protocol_scene.usd
            usd_filename = json_filename.replace(".json", ".usd")
        
        usd_file_path = self.scenes_dir / usd_filename
        if not usd_file_path.exists():
            raise FileNotFoundError(f"USD 文件不存在: {usd_file_path}")
        
        # 确定输出路径
        if in_place:
            output_path = usd_file_path
        elif output_filename:
            output_path = self.scenes_dir / output_filename
        else:
            # 默认：直接修改原文件
            output_path = usd_file_path
        
        self.apply_positions_to_usd(json_file_path, usd_file_path, output_path, in_place)
        return output_path if output_path else usd_file_path


def update_positions_from_json(
    json_file_path: str,
    usd_file_path: str,
    output_usd_path: Optional[str] = None,
    in_place: bool = True,
    scenes_dir: str = str(DEFAULT_SCENES_DIR)
) -> bool:
    """
    便捷函数：从 JSON 文件更新 USD 文件中的位置信息
    
    Args:
        json_file_path: JSON 文件路径
        usd_file_path: 输入 USD 文件路径
        output_usd_path: 输出 USD 文件路径（如果为 None 且 in_place=False，则自动生成）
        in_place: 是否直接修改原文件
        scenes_dir: 场景文件目录（用于相对路径解析）
    
    Returns:
        是否成功
    """
    updater = PositionUpdater(scenes_dir=scenes_dir)
    return updater.apply_positions_to_usd(
        Path(json_file_path),
        Path(usd_file_path),
        Path(output_usd_path) if output_usd_path else None,
        in_place
    )


if __name__ == "__main__":
    args = _cli_args
    
    # 判断是文件名还是完整路径
    json_path = Path(args.json_file)
    if not json_path.is_absolute():
        json_path = Path(args.scenes_dir) / args.json_file
    
    usd_path = None
    if args.usd_file:
        usd_path_obj = Path(args.usd_file)
        if not usd_path_obj.is_absolute():
            usd_path_obj = Path(args.scenes_dir) / args.usd_file
        usd_path = str(usd_path_obj)
    
    output_path = None
    if args.output_file:
        output_path_obj = Path(args.output_file)
        if not output_path_obj.is_absolute():
            output_path_obj = Path(args.scenes_dir) / args.output_file
        output_path = str(output_path_obj)
    
    # 如果指定了输出文件，则 in_place=False，否则 in_place=True（默认直接修改原文件）
    in_place = output_path is None
    
    try:
        updater = PositionUpdater(scenes_dir=args.scenes_dir)
        if usd_path:
            updater.apply_positions_to_usd(
                json_path,
                Path(usd_path),
                Path(output_path) if output_path else None,
                in_place
            )
        else:
            updater.update_from_json_filename(
                args.json_file,
                None,
                args.output_file,
                in_place
            )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if simulation_app:
            simulation_app.close()
