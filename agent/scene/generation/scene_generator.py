"""场景生成器：从器材列表生成 USD 场景文件"""

import os
import sys
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd, Sdf

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.isaacsim_runtime import prepare_isaacsim_argv


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description="场景生成器：从器材列表生成完整的 USD 场景文件"
    )
    parser.add_argument(
        "scene_info_file",
        type=str,
        help="scene_information 文件夹中的文件名（如 protocol1_scene.txt）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出文件名（默认使用 scene_info_file 的名称）",
    )
    parser.add_argument(
        "--instruments-dir",
        type=str,
        default=str(_project_root / "Instruments"),
        help="Instruments 文件夹路径",
    )
    parser.add_argument(
        "--base-usd",
        type=str,
        default=str(
            _project_root / "protocols/Level2_Protocol1/scene.usd"
        ),
        help="基础 USD 文件路径",
    )
    parser.add_argument(
        "--scene-info-dir",
        type=str,
        default=str(_project_root / "agent/protocol/scene_information"),
        help="scene_information 文件夹路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_project_root / "agent/scene/scenes"),
        help="输出场景文件夹路径",
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
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
except ImportError:
    print("错误：未找到 'pxr' (OpenUSD) 库。", file=sys.stderr)
    print("请确保在 Isaac Sim 环境中运行，或已正确安装 OpenUSD 及其 Python 绑定。", file=sys.stderr)
    # 注意：不在这里关闭 simulation_app，因为它可能由 scene_initializer 管理
    # 仅在直接运行此模块时才关闭
    if __name__ == "__main__" and simulation_app:
        simulation_app.close()
    if __name__ == "__main__":
        sys.exit(1)


class SceneGenerator:
    INTERACTIVE_ASSETS = {
        "BalaoVolumetrico", "Beaker", "Crucible", "DistillationFlask",
        "ErlenmeyerFlask", "FlatBottomFlask", "KippsApparatus",
        "RoundBottomFlask", "SuctionFlask", "TitrationFlasks",
        "VolumetricBottle", "GlassRod", "GlassFunnel", "SeparatoryFunnel",
        "Thermometer", "WashBottle", "Burette", "TestTube", "Pipette",
        "GrahamCondenser", "TubeRack", "TargetPlatform", "TargetPlat"
    }
    
    def __init__(
        self,
        instruments_dir: Optional[str] = None,
        base_usd_path: Optional[str] = None,
        scene_info_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        root = Path(__file__).resolve().parents[3]
        self.instruments_dir = Path(instruments_dir or root / "Instruments")
        self.base_usd_path = Path(
            base_usd_path or root / "protocols/Level2_Protocol1/scene.usd"
        )
        self.scene_info_dir = Path(
            scene_info_dir or root / "agent/protocol/scene_information"
        )
        self.output_dir = Path(output_dir or root / "agent/scene/scenes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.instruments_dir.exists():
            raise ValueError(f"Instruments 目录不存在: {self.instruments_dir}")
        if not self.base_usd_path.exists():
            raise ValueError(f"基础 USD 文件不存在: {self.base_usd_path}")
        if not self.scene_info_dir.exists():
            raise ValueError(f"scene_information 目录不存在: {self.scene_info_dir}")
    
    def read_scene_information(self, filename: str) -> Tuple[List[str], List[str]]:
        file_path = self.scene_info_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"场景信息文件不存在: {file_path}")
        
        original_list = []
        equipment_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    original_list.append(line)
                    equipment_name = self._remove_number_suffix(line)
                    equipment_list.append(equipment_name)
        
        print(f"从 {filename} 中读取到 {len(equipment_list)} 个器材")
        return original_list, equipment_list
    
    def _remove_number_suffix(self, name: str) -> str:
        import re
        if name.startswith("TargetPlatform") or name.startswith("TargetPlat"):
            return "TargetPlatform" if name.startswith("TargetPlatform") else "TargetPlat"
        # 处理 ErlenmeyerFlask_01/02 和 ErlenmeyerFlask_Solid/Liquid（保留完整名称）
        if (name.startswith("ErlenmeyerFlask_01") or name.startswith("ErlenmeyerFlask_02") or
            name.startswith("ErlenmeyerFlask_Solid") or name.startswith("ErlenmeyerFlask_Liquid")):
            return name
        match = re.match(r'^(.+?)(\d+)$', name)
        return match.group(1) if match else name
    
    def find_usd_file(self, equipment_name: str) -> Optional[Path]:
        # 处理 ErlenmeyerFlask_01 和 ErlenmeyerFlask_02（新格式，优先匹配）
        if equipment_name == "ErlenmeyerFlask_01" or equipment_name.startswith("ErlenmeyerFlask_01"):
            usd_path = self.instruments_dir / "InteractiveAssets" / "ErlenmeyerFlask" / "ErlenmeyerFlask_01" / "ErlenmeyerFlask_01.usd"
            return usd_path if usd_path.exists() else None
        
        if equipment_name == "ErlenmeyerFlask_02" or equipment_name.startswith("ErlenmeyerFlask_02"):
            usd_path = self.instruments_dir / "InteractiveAssets" / "ErlenmeyerFlask" / "ErlenmeyerFlask_02" / "ErlenmeyerFlask_02.usd"
            return usd_path if usd_path.exists() else None
        
        # 处理 ErlenmeyerFlask_Liquid 和 ErlenmeyerFlask_Solid（保留后缀，向后兼容）
        if equipment_name.startswith("ErlenmeyerFlask_Liquid"):
            usd_path = self.instruments_dir / "InteractiveAssets" / "ErlenmeyerFlask" / "ErlenmeyerFlask_02" / "ErlenmeyerFlask_02.usd"
            return usd_path if usd_path.exists() else None
        
        if equipment_name.startswith("ErlenmeyerFlask_Solid"):
            usd_path = self.instruments_dir / "InteractiveAssets" / "ErlenmeyerFlask" / "ErlenmeyerFlask_01" / "ErlenmeyerFlask_01.usd"
            return usd_path if usd_path.exists() else None
        
        # 处理 TargetPlatform 和 TargetPlat（包括带数字后缀的情况）
        if equipment_name.startswith("TargetPlatform") or equipment_name.startswith("TargetPlat"):
            target_path = self.instruments_dir / "PlacementAssets" / "DesktopLevel" / "TargetPlat"
            if target_path.exists():
                usd_file = self._find_usd_in_directory(target_path)
                if usd_file:
                    return usd_file
        
        # 先尝试使用原始名称查找
        interactive_path = self.instruments_dir / "InteractiveAssets" / equipment_name
        if interactive_path.exists():
            usd_file = self._find_usd_in_directory(interactive_path)
            if usd_file:
                return usd_file
        
        desktop_path = self.instruments_dir / "PlacementAssets" / "DesktopLevel" / equipment_name
        if desktop_path.exists():
            usd_file = self._find_usd_in_directory(desktop_path)
            if usd_file:
                return usd_file
        
        room_path = self.instruments_dir / "PlacementAssets" / "RoomLevel" / equipment_name
        if room_path.exists():
            usd_file = self._find_usd_in_directory(room_path)
            if usd_file:
                return usd_file
        
        # 如果原始名称找不到，尝试去除数字后缀后再查找
        base_name = self._remove_number_suffix(equipment_name)
        if base_name != equipment_name:
            # 再次尝试查找（但跳过已经处理过的特殊情况）
            if not (base_name.startswith("TargetPlatform") or base_name.startswith("TargetPlat") or 
                    base_name.startswith("ErlenmeyerFlask_01") or base_name.startswith("ErlenmeyerFlask_02") or
                    base_name.startswith("ErlenmeyerFlask_Liquid") or base_name.startswith("ErlenmeyerFlask_Solid")):
                interactive_path = self.instruments_dir / "InteractiveAssets" / base_name
                if interactive_path.exists():
                    usd_file = self._find_usd_in_directory(interactive_path)
                    if usd_file:
                        return usd_file
                
                desktop_path = self.instruments_dir / "PlacementAssets" / "DesktopLevel" / base_name
                if desktop_path.exists():
                    usd_file = self._find_usd_in_directory(desktop_path)
                    if usd_file:
                        return usd_file
                
                room_path = self.instruments_dir / "PlacementAssets" / "RoomLevel" / base_name
                if room_path.exists():
                    usd_file = self._find_usd_in_directory(room_path)
                    if usd_file:
                        return usd_file
        
        print(f"警告：未找到器材 {equipment_name} 的 USD 文件")
        return None
    
    def _find_usd_in_directory(self, directory: Path) -> Optional[Path]:
        """
        在目录中查找 USD 文件，按排序后的顺序返回第一个匹配的文件
        
        查找顺序（按优先级）：
        1. {目录名}.usd 或 {目录名}.usdc（如果存在）
        2. 目录中的文件（按名称排序）
        3. 一级子目录中的文件（子目录和文件都按名称排序）
        4. 递归遍历所有子目录（按路径排序）
        
        确保返回的是"第一个"文件，而不是随机的文件。
        """
        # 1. 优先查找与目录名相同的文件
        for ext in ['.usd', '.usdc']:
            usd_file = directory / f"{directory.name}{ext}"
            if usd_file.exists():
                return usd_file
        
        # 2. 查找目录中的文件（按名称排序）
        files_in_dir = [f for f in directory.iterdir() if f.is_file() and (f.suffix == '.usd' or f.suffix == '.usdc')]
        if files_in_dir:
            files_in_dir.sort(key=lambda p: p.name)  # 按文件名排序
            return files_in_dir[0]
        
        # 3. 查找一级子目录中的文件（子目录和文件都按名称排序）
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        subdirs.sort(key=lambda p: p.name)  # 按子目录名排序
        
        for subdir in subdirs:
            files_in_subdir = [f for f in subdir.iterdir() if f.is_file() and (f.suffix == '.usd' or f.suffix == '.usdc')]
            if files_in_subdir:
                files_in_subdir.sort(key=lambda p: p.name)  # 按文件名排序
                return files_in_subdir[0]
        
        # 4. 递归遍历所有子目录（按路径排序）
        all_usd_files = []
        for root, dirs, files in os.walk(directory):
            # 对目录列表排序，确保遍历顺序一致
            dirs.sort()
            for file in sorted(files):  # 对文件列表排序
                if file.endswith('.usd') or file.endswith('.usdc'):
                    all_usd_files.append(Path(root) / file)
        
        if all_usd_files:
            # 按完整路径排序，确保返回第一个
            all_usd_files.sort(key=lambda p: str(p))
            return all_usd_files[0]
        
        return None
    
    def _normalize_asset_unit(self, usd_file_path: Path, equipment_name: str, target_unit: float = 1.0) -> Path:
        """
        统一资产文件的单位设置，不改变几何体大小
        
        工作原理：
        1. USD 引用机制会自动根据单位元数据进行转换
        2. 如果资产文件单位是 0.01（厘米），场景单位是 1.0（米）
        3. USD 会自动将资产中的数值乘以 0.01 来转换为场景单位
        4. 此方法只修改单位元数据，不修改几何体数值
        5. USD 引用会根据新的单位元数据自动处理转换，保持几何体大小不变
        
        Args:
            usd_file_path: 原始 USD 文件路径
            equipment_name: 器材名称（用于日志）
            target_unit: 目标单位（米/单位），默认 1.0（米）
        
        Returns:
            如果单位已正确，返回原文件路径；如果需要修正，返回临时文件路径
        """
        try:
            ref_stage = Usd.Stage.Open(str(usd_file_path))
            if not ref_stage:
                return usd_file_path
            
            ref_meters_per_unit = UsdGeom.GetStageMetersPerUnit(ref_stage)
            if ref_meters_per_unit == 0:
                ref_meters_per_unit = 1.0
            
            # 如果单位已经正确，直接返回原文件
            if abs(ref_meters_per_unit - target_unit) < 1e-6:
                return usd_file_path
            
            # 单位不一致，创建临时文件并修正单位元数据
            # 注意：只修改单位元数据，不修改几何体数值
            # USD 引用机制会根据单位元数据自动转换，保持几何体大小不变
            import tempfile
            import shutil
            temp_dir = Path(tempfile.gettempdir()) / "usd_unit_normalize"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            temp_file = temp_dir / f"{usd_file_path.stem}_normalized{usd_file_path.suffix}"
            
            # 先复制原文件
            shutil.copy2(usd_file_path, temp_file)
            
            # 打开复制的文件并修改单位设置
            temp_stage = Usd.Stage.Open(str(temp_file))
            if temp_stage:
                # 只修改单位元数据，不改变几何体数值
                # USD 引用会根据新的单位元数据自动处理转换
                UsdGeom.SetStageMetersPerUnit(temp_stage, target_unit)
                temp_stage.GetRootLayer().Save()
                print(f"    [单位修正] {equipment_name}: {ref_meters_per_unit} -> {target_unit} 米/单位（几何体大小保持不变）")
                return temp_file
            else:
                return usd_file_path
            
        except Exception as e:
            print(f"    [警告] 无法修正 {usd_file_path.name} 的单位: {e}")
            return usd_file_path
    
    def add_reference_to_stage(
        self,
        stage: "Usd.Stage",
        usd_file_path: Path,
        prim_path: "Sdf.Path",
        equipment_name: str,
        position: Optional[Tuple[float, float, float]] = None,
        normalize_unit: bool = True
    ) -> bool:
        """
        添加 USD 引用到场景
        
        Args:
            stage: 目标场景 stage
            usd_file_path: 要引用的 USD 文件路径
            prim_path: 在场景中的 prim 路径
            equipment_name: 器材名称（用于日志）
            position: 可选的位置
            normalize_unit: 是否在导入前统一单位（默认 True）
        """
        try:
            # 在导入前统一单位（如果需要）
            actual_file_path = usd_file_path
            if normalize_unit:
                scene_unit = UsdGeom.GetStageMetersPerUnit(stage)
                if scene_unit == 0:
                    scene_unit = 1.0
                actual_file_path = self._normalize_asset_unit(usd_file_path, equipment_name, target_unit=scene_unit)
            
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                prim = stage.DefinePrim(prim_path, "Xform")
            
            abs_path = str(actual_file_path.resolve())
            prim.GetReferences().AddReference(abs_path)
            
            if position is not None:
                xformable = UsdGeom.Xformable(prim)
                translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(position))
                print(f"  ✓ 成功添加引用: {equipment_name} -> {prim_path} (位置: {position})")
            else:
                print(f"  ✓ 成功添加引用: {equipment_name} -> {prim_path} (默认位置: (0, 0, 0))")
            
            return True
        except Exception as e:
            print(f"  ✗ 添加引用失败 {equipment_name}: {e}")
            return False
    
    def add_physics_properties(
        self,
        stage: "Usd.Stage",
        prim_path: "Sdf.Path",
        is_interactive: bool
    ):
        """
        为场景中的物体添加 Colliders Preset。
        
        要求：
        - 对于可交互资产（is_interactive=True）：不额外添加外层碰撞壳，
          但会对其内部已存在的 CollisionAPI 做一次“修复”，
          例如将无效的 contactOffset/restOffset（如 -inf）重设为合理默认值，
          等价于在 UI 中对该资产执行一次 Colliders Preset 修复操作
        - 对于非交互资产（is_interactive=False）：为该 prim 及其子树添加碰撞体
        - 不在这里添加 RigidBodyAPI（刚体），刚体由上层或资产自身控制
        - 实现方式：遍历给定 prim 及其子节点，对所有 Xformable（或 Mesh）应用 CollisionAPI
        """
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        from pxr import Usd, UsdGeom, UsdPhysics  # 局部导入，避免顶部依赖顺序问题
        from math import isfinite

        # 可交互资产：完全不做任何修改（不修复、不新增碰撞体），交给原始资产配置和上层流程处理
        if is_interactive:
            return

        def _apply_collider(p):
            # 只对可变换的 prim（通常是 Xform/mesh 等）添加碰撞
            if p.IsA(UsdGeom.Xformable):
                if not p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(p)

        # 对当前 prim 及其所有后代节点应用“Colliders Preset”效果
        _apply_collider(prim)
        # 使用 Usd.PrimRange 来遍历整个子树，而不是对 Prim 调用不存在的 Traverse()
        for desc in Usd.PrimRange(prim):
            if desc == prim:
                continue  # 已经处理过根节点
            _apply_collider(desc)
    
    def is_interactive_asset(self, equipment_name: str) -> bool:
        # 精确匹配
        if equipment_name in self.INTERACTIVE_ASSETS:
            return True
        # 处理带后缀的情况：ErlenmeyerFlask_Solid1, ErlenmeyerFlask_Liquid1 等
        # 这些应该被视为可交互资产（因为它们本质上是 ErlenmeyerFlask）
        if equipment_name.startswith("ErlenmeyerFlask_Solid") or equipment_name.startswith("ErlenmeyerFlask_Liquid"):
            return True
        # 处理带数字后缀的情况：Beaker1, Beaker2 等
        import re
        base_name = re.match(r'^(.+?)(\d+)$', equipment_name)
        if base_name:
            base = base_name.group(1)
            if base in self.INTERACTIVE_ASSETS:
                return True
        return False
    
    def generate_scene(
        self,
        scene_info_filename: str,
        output_filename: Optional[str] = None
    ) -> Optional[Path]:
        print(f"\n{'='*60}")
        print(f"开始生成场景: {scene_info_filename}")
        print(f"{'='*60}\n")
        
        original_names, equipment_names = self.read_scene_information(scene_info_filename)
        if not equipment_names:
            print("错误：器材列表为空")
            return None

        equipment = [
            (
                original_name,
                self.find_usd_file(original_name),
                self.is_interactive_asset(equipment_name),
            )
            for original_name, equipment_name in zip(original_names, equipment_names)
        ]
        if output_filename is None:
            output_filename = f"{Path(scene_info_filename).stem}.usd"
        return self.generate_scene_from_assets(equipment, output_filename)

    def generate_scene_from_assets(
        self,
        equipment: list[tuple[str, Path, bool]],
        output_filename: str,
    ) -> Path | None:
        """Generate a scene from instance names and registry-resolved USD paths."""
        missing = [
            str(usd_file)
            for _, usd_file, _ in equipment
            if usd_file is None or not Path(usd_file).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"scene assets do not exist: {missing}")

        print(f"正在打开基础 USD 文件: {self.base_usd_path}...")
        try:
            stage = Usd.Stage.Open(str(self.base_usd_path))
        except NameError:
            print("错误：pxr (OpenUSD) 库未导入，无法打开 USD 文件")
            print("请确保在 Isaac Sim 环境中运行，或已正确安装 OpenUSD 及其 Python 绑定")
            return None
        except Exception as e:
            print(f"错误：无法打开基础 USD 文件: {e}")
            return None
        
        if not stage:
            print("错误：无法加载 Stage")
            return None

        instance_name_counts = Counter(name for name, _, _ in equipment)
        duplicate_names = sorted(
            name for name, count in instance_name_counts.items() if count > 1
        )
        occupied_paths = []
        for instance_name in sorted(instance_name_counts):
            prim_path = f"/World/{instance_name}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                occupied_paths.append(prim_path)

        conflicts = []
        if duplicate_names:
            conflicts.append(f"duplicate instance names: {duplicate_names}")
        if occupied_paths:
            conflicts.append(f"occupied prim paths: {occupied_paths}")
        if conflicts:
            raise ValueError(f"scene instance conflicts: {'; '.join(conflicts)}")
        
        # 获取并设置基础场景的单位（统一为米）
        base_meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        if base_meters_per_unit == 0:
            base_meters_per_unit = 1.0
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # 强制设置为米
        print(f"✓ 基础 USD 文件加载成功（单位: {base_meters_per_unit} -> 1.0 米/单位）\n")
        
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim:
            print("错误：未找到 /World Prim")
            return None
        
        print("正在添加器材引用...")
        added_count = 0
        unit_warnings = []
        
        for instance_name, usd_file, interactive in equipment:
            usd_file = Path(usd_file)

            # 检查被引用资产的单位设置
            try:
                ref_stage = Usd.Stage.Open(str(usd_file))
                if ref_stage:
                    ref_meters_per_unit = UsdGeom.GetStageMetersPerUnit(ref_stage)
                    if ref_meters_per_unit == 0:
                        ref_meters_per_unit = 1.0
                    
                    # 如果单位不一致，记录警告
                    if ref_meters_per_unit != 1.0:
                        unit_warnings.append(f"{instance_name}: {ref_meters_per_unit} 米/单位")
            except Exception:
                pass

            prim_path = Sdf.Path(f"/World/{instance_name}")

            if self.add_reference_to_stage(
                stage,
                usd_file,
                prim_path,
                instance_name,
                position=None,
            ):
                added_count += 1
        
        # 如果有单位不一致的警告，打印出来
        if unit_warnings:
            print(f"\n警告：以下资产的单位设置与场景不一致（场景已统一为米）:")
            for warning in unit_warnings:
                print(f"  - {warning}")
            print("  注意：USD 引用会自动处理单位转换，但建议统一资产文件的单位设置\n")
        
        print("\n正在添加物理属性...")
        stage.Load()
        
        for instance_name, _, interactive in equipment:
            prim_path = Sdf.Path(f"/World/{instance_name}")

            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                self.add_physics_properties(stage, prim_path, interactive)
                print(f"  ✓ 已为 {instance_name} 添加物理属性（碰撞: 是, 刚体: {'是' if interactive else '否'}）")

        print(f"\n✓ 成功添加 {added_count}/{len(equipment)} 个器材引用")
        
        output_path = self.output_dir / output_filename
        
        print(f"\n正在 flatten 场景并保存到: {output_path}...")
        try:
            # 使用 Flatten() 方法创建一个新的 flattened layer
            # 由于在导入前已经统一了单位，flatten 后的场景单位应该是一致的
            flattened_layer = stage.Flatten()
            
            # 确保 flatten 后的单位设置正确
            flattened_layer.Export(str(output_path))
            
            # 验证最终文件的单位设置
            final_stage = Usd.Stage.Open(str(output_path))
            if final_stage:
                final_unit = UsdGeom.GetStageMetersPerUnit(final_stage)
                if final_unit == 0:
                    final_unit = 1.0
                if abs(final_unit - 1.0) > 1e-6:
                    # 如果单位不一致，重新设置
                    UsdGeom.SetStageMetersPerUnit(final_stage, 1.0)
                    final_stage.GetRootLayer().Save()
                    print(f"  已统一最终场景单位: {final_unit} -> 1.0 米/单位")
            
            print(f"✓ 场景 flatten 并保存成功: {output_path}")
        except Exception as e:
            print(f"✗ 保存场景失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return output_path


def generate_scene(
    scene_info_filename: str,
    instruments_dir: Optional[str] = None,
    base_usd_path: Optional[str] = None,
    scene_info_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None
) -> Optional[Path]:
    root = Path(__file__).resolve().parents[3]
    generator = SceneGenerator(
        instruments_dir=instruments_dir or str(root / "Instruments"),
        base_usd_path=base_usd_path or str(
            root / "protocols/Level2_Protocol1/scene.usd"
        ),
        scene_info_dir=scene_info_dir or str(
            root / "agent/protocol/scene_information"
        ),
        output_dir=output_dir or str(root / "agent/scene/scenes")
    )
    return generator.generate_scene(scene_info_filename, output_filename)


if __name__ == "__main__":
    args = _cli_args
    
    result = generate_scene(
        scene_info_filename=args.scene_info_file,
        instruments_dir=args.instruments_dir,
        base_usd_path=args.base_usd,
        scene_info_dir=args.scene_info_dir,
        output_dir=args.output_dir,
        output_filename=args.output
    )
    
    if result:
        print(f"\n{'='*60}")
        print(f"场景生成完成！")
        print(f"输出文件: {result}")
        print(f"{'='*60}\n")
        if simulation_app:
            simulation_app.close()
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print(f"场景生成失败！")
        print(f"{'='*60}\n")
        if simulation_app:
            simulation_app.close()
        sys.exit(1)
