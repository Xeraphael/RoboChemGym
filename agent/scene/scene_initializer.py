"""
场景初始化器：完整的场景生成和优化流程

功能流程：
1. 从协议文本提取器材和动作信息
2. 根据器材信息生成USD场景
3. 提取场景信息到JSON
4. 优化物体位置
5. 更新USD场景文件
6. 生成YAML配置文件
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到Python路径，确保可以导入agent模块
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.isaacsim_runtime import prepare_isaacsim_argv

# 全局 SimulationApp 实例（由 SceneInitializer 管理）
# 不在模块级别初始化，而是在 SceneInitializer.__init__ 中初始化
# 这样可以避免在模块导入时就初始化，导致段错误
_simulation_app = None


@dataclass
class SceneInitializationResult:
    """场景初始化结果"""
    equipment_file_path: Optional[str] = None
    actions_file_path: Optional[str] = None
    scene_usd_path: Optional[str] = None
    scene_json_path: Optional[str] = None
    optimized_json_path: Optional[str] = None
    updated_usd_path: Optional[str] = None
    yaml_config_path: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None


class SceneInitializer:
    """场景初始化器：协调完整的场景生成和优化流程"""
    
    def __init__(
        self,
        protocol_dir: Optional[str] = None,
        scene_info_dir: Optional[str] = None,
        action_info_dir: Optional[str] = None,
        scenes_dir: Optional[str] = None,
        instruments_dir: Optional[str] = None,
        base_usd_path: Optional[str] = None,
        config_dir: Optional[str] = None,
        manage_simulation_app: bool = True
    ):
        """
        初始化场景初始化器
        
        Args:
            protocol_dir: 协议目录
            scene_info_dir: 场景信息目录（存储器材列表）
            action_info_dir: 动作信息目录（存储动作序列）
            scenes_dir: 场景文件目录（存储USD和JSON文件）
            instruments_dir: 器材资源目录
            base_usd_path: 基础USD场景文件路径
            config_dir: 配置文件目录（存储YAML配置文件）
            manage_simulation_app: 是否管理SimulationApp生命周期（默认True）
        """
        self.protocol_dir = Path(protocol_dir) if protocol_dir is not None else (
            _project_root / "agent/protocol"
        )
        self.scene_info_dir = Path(scene_info_dir) if scene_info_dir is not None else (
            _project_root / "agent/protocol/scene_information"
        )
        self.action_info_dir = Path(action_info_dir) if action_info_dir is not None else (
            _project_root / "agent/protocol/action_information"
        )
        self.scenes_dir = Path(scenes_dir) if scenes_dir is not None else (
            _project_root / "agent/scene/scenes"
        )
        self.instruments_dir = Path(instruments_dir) if instruments_dir is not None else (
            _project_root / "Instruments"
        )
        self.base_usd_path = Path(base_usd_path) if base_usd_path is not None else (
            _project_root / "protocols/Level2_Protocol1/scene.usd"
        )
        self.config_dir = Path(config_dir) if config_dir is not None else (
            _project_root / "config"
        )
        self.manage_simulation_app = manage_simulation_app
        
        # 确保目录存在
        self.scene_info_dir.mkdir(parents=True, exist_ok=True)
        self.action_info_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 在初始化时确保 SimulationApp 已初始化（在导入其他模块之前）
        # 这样当其他模块被导入时，它们可以检测到已存在的实例
        if self.manage_simulation_app:
            self._ensure_simulation_app()
    
    def _ensure_simulation_app(self):
        """确保 SimulationApp 已初始化（如果模块级别初始化失败，这里会重试）"""
        global _simulation_app
        if _simulation_app is None:
            try:
                prepare_isaacsim_argv()
                from isaacsim import SimulationApp
                _simulation_config = {"headless": True}
                _simulation_app = SimulationApp(_simulation_config)
                print("[SceneInitializer] SimulationApp initialized in _ensure_simulation_app")
            except ImportError:
                print("[SceneInitializer] Warning: Isaac Sim not found, some features may not work")
            except Exception as e:
                print(f"[SceneInitializer] Warning: Failed to initialize SimulationApp: {e}")
    
    def step1_extract_protocol(
        self,
        protocol_text: str,
        prompt_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        步骤1：从协议文本提取器材和动作信息
        
        Args:
            protocol_text: 协议文本
            prompt_path: 提示词文件路径
            api_key: API密钥
            base_url: API基础URL
        
        Returns:
            (equipment_file_path, actions_file_path)
        """
        print("\n" + "="*60)
        print("步骤 1: 提取协议信息")
        print("="*60)
        
        try:
            from agent.protocol.protocol_extractor import ProtocolExtractor
            
            extractor = ProtocolExtractor(
                prompt_path=prompt_path,
                api_key=api_key,
                base_url=base_url
            )
            
            equipment_text, actions_text, equipment_file, actions_file = extractor.extract(
                protocol_text,
                output_dir=str(self.protocol_dir)
            )
            
            if equipment_file and actions_file:
                print(f"✓ 器材信息已保存: {equipment_file}")
                print(f"✓ 动作信息已保存: {actions_file}")
                return equipment_file, actions_file
            else:
                print("✗ 提取失败：未生成文件")
                return None, None
                
        except Exception as e:
            print(f"✗ 提取协议信息失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def step2_generate_scene(
        self,
        equipment_file_path: str
    ) -> Optional[str]:
        """
        步骤2：根据器材信息生成USD场景
        
        Args:
            equipment_file_path: 器材信息文件路径
        
        Returns:
            生成的USD场景文件路径
        """
        print("\n" + "="*60)
        print("步骤 2: 生成场景")
        print("="*60)
        
        try:
            # 检查 pxr 库是否可用
            try:
                from pxr import Usd
            except ImportError:
                print("⚠ 警告：pxr (OpenUSD) 库未导入，无法生成场景")
                print("请确保在 Isaac Sim 环境中运行，或已正确安装 OpenUSD 及其 Python 绑定")
                print("场景生成步骤已跳过，但协议提取步骤已完成")
                return None
            
            from agent.scene.generation.scene_generator import SceneGenerator
            
            # 获取文件名（不含路径）
            equipment_filename = Path(equipment_file_path).name
            
            generator = SceneGenerator(
                instruments_dir=str(self.instruments_dir),
                base_usd_path=str(self.base_usd_path),
                scene_info_dir=str(self.scene_info_dir),
                output_dir=str(self.scenes_dir)
            )
            
            # 生成输出文件名（基于器材文件名）
            base_name = Path(equipment_filename).stem
            output_filename = f"{base_name}.usd"
            
            scene_path = generator.generate_scene(
                scene_info_filename=equipment_filename,
                output_filename=output_filename
            )
            
            if scene_path:
                print(f"✓ 场景已生成: {scene_path}")
                return str(scene_path)
            else:
                print("✗ 场景生成失败")
                return None
                
        except Exception as e:
            print(f"✗ 生成场景失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step3_extract_scene(
        self,
        scene_usd_path: str
    ) -> Optional[str]:
        """
        步骤3：从USD场景提取信息到JSON
        
        Args:
            scene_usd_path: USD场景文件路径
        
        Returns:
            生成的JSON文件路径
        """
        print("\n" + "="*60)
        print("步骤 3: 提取场景信息")
        print("="*60)
        
        try:
            from agent.scene.extractor.scene_extractor import SceneExtractor
            
            extractor = SceneExtractor(scenes_dir=str(self.scenes_dir))
            
            # 获取文件名（不含路径）
            usd_filename = Path(scene_usd_path).name
            
            json_path = extractor.extract(usd_filename)
            
            if json_path:
                print(f"✓ 场景信息已提取: {json_path}")
                return str(json_path)
            else:
                print("✗ 场景信息提取失败")
                return None
                
        except Exception as e:
            print(f"✗ 提取场景信息失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step4_optimize_positions(
        self,
        scene_json_path: str,
        z_height: Optional[float] = None,
        grid_resolution: Optional[float] = None,
        semantic_prompt: Optional[str] = None,
        actions_file: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        步骤4：优化物体位置
        
        Args:
            scene_json_path: 场景JSON文件路径
            z_height: 放置高度
            grid_resolution: 网格分辨率
            actions_file: 动作流程信息文件（用于语义优化参考）
            api_key: API密钥（未提供时由优化器使用环境变量）
            base_url: API基础URL
        
        Returns:
            优化后的JSON文件路径（与原文件相同，已更新）
        """
        print("\n" + "="*60)
        print("步骤 4: 优化物体位置")
        print("="*60)
        
        try:
            from agent.scene.optimization.position_optimizer import (
                PositionOptimizer,
                OptimizationMethod,
            )

            import yaml

            root = Path(__file__).resolve().parents[2]
            profiles = yaml.safe_load(
                (root / "agent/scene/layout_profiles.yaml").read_text(encoding="utf-8")
            )
            profile = profiles["profiles"]["lab_table_franka"]
            if z_height is not None:
                profile["surface_z"] = float(z_height)
            if grid_resolution is not None:
                profile["grid_resolution"] = float(grid_resolution)

            optimizer = PositionOptimizer.from_profile(Path(scene_json_path), profile)
            
            # 执行优化
            optimized_positions = optimizer.optimize(
                method=OptimizationMethod.MILP,
                grid_resolution=float(profile["grid_resolution"])
            )
            
            if optimized_positions is not None:
                # 保存优化结果（覆盖原文件）
                optimizer.save_optimized_positions(optimized_positions, None)
                print(f"✓ 位置优化完成: {scene_json_path}")
                
                # 应用语义约束
                semantic_constraint_text = semantic_prompt
                
                # 如果有提示词或者有动作流程文件，就应用语义约束
                if semantic_constraint_text or actions_file:
                    print("\n" + "="*60)
                    print("应用语义布局优化 (基于动作流程及提示词)...")
                    print("="*60)
                    if semantic_constraint_text:
                        print(f"[SceneInitializer] 语义约束内容: {semantic_constraint_text}")
                    if actions_file:
                        print(f"[SceneInitializer] 参考动作文件: {actions_file}")
                    
                    semantic_result = optimizer.apply_semantic_constraints(
                        semantic_prompt=semantic_constraint_text or "", # 允许为空字符串
                        actions_file=Path(actions_file) if actions_file else None,
                        json_file_path=Path(scene_json_path),
                        validate_constraints=True,
                        allow_collision=True,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    
                    if semantic_result is not None:
                        print(f"✓ 语义约束已应用: {scene_json_path}")
                    else:
                        print("✗ 语义约束应用失败")
                
                return scene_json_path
            else:
                print("✗ 位置优化失败：无法找到满足所有约束的解")
                return None
                
        except Exception as e:
            print(f"✗ 优化位置失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step5_update_scene(
        self,
        optimized_json_path: str,
        scene_usd_path: Optional[str] = None,
        in_place: bool = True
    ) -> Optional[str]:
        """
        步骤5：更新USD场景文件
        
        Args:
            optimized_json_path: 优化后的JSON文件路径
            scene_usd_path: USD场景文件路径（如果为None，从JSON文件名推断）
            in_place: 是否直接修改原文件
        
        Returns:
            更新后的USD文件路径
        """
        print("\n" + "="*60)
        print("步骤 5: 更新场景文件")
        print("="*60)
        
        try:
            from agent.scene.optimization.position_updater import PositionUpdater
            
            updater = PositionUpdater(scenes_dir=str(self.scenes_dir))
            
            # 获取JSON文件名
            json_filename = Path(optimized_json_path).name
            
            # 如果没有指定USD文件，从JSON文件名推断
            if scene_usd_path is None:
                usd_filename = json_filename.replace(".json", ".usd")
            else:
                usd_filename = Path(scene_usd_path).name
            
            updated_path = updater.update_from_json_filename(
                json_filename=json_filename,
                usd_filename=usd_filename,
                output_filename=None,
                in_place=in_place
            )
            
            if updated_path:
                print(f"✓ 场景已更新: {updated_path}")
                return str(updated_path)
            else:
                print("✗ 场景更新失败")
                return None
                
        except Exception as e:
            print(f"✗ 更新场景失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step6_generate_yaml(
        self,
        scene_json_path: str,
        scene_usd_path: str,
        template_yaml_path: Optional[str] = None,
        output_yaml_path: Optional[str] = None
    ) -> Optional[str]:
        """
        步骤6：从JSON场景文件生成YAML配置文件
        
        Args:
            scene_json_path: 场景JSON文件路径
            scene_usd_path: 场景USD文件路径（用于设置usd_path字段）
            template_yaml_path: 模板YAML文件路径（如果为None，使用Level2_Protocol1.yaml）
            output_yaml_path: 输出YAML文件路径（如果为None，根据JSON文件名生成）
        
        Returns:
            生成的YAML文件路径
        """
        print("\n" + "="*60)
        print("步骤 6: 生成YAML配置文件")
        print("="*60)
        
        try:
            # 尝试导入 ruamel.yaml，如果失败则使用标准 yaml
            try:
                from ruamel.yaml import YAML
                from ruamel.yaml.comments import CommentedMap, CommentedSeq
                use_ruamel = True
            except ImportError:
                import yaml
                use_ruamel = False
                print("[YAMLGenerator] 警告: ruamel.yaml 未安装，将使用标准 yaml 库（可能无法保留注释）")
            
            # 读取JSON文件
            json_path = Path(scene_json_path)
            if not json_path.exists():
                print(f"✗ JSON文件不存在: {json_path}")
                return None
            
            with open(json_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
            
            # 提取所有 /World/ 下的物体路径
            world_paths = []
            for prim_path in scene_data.keys():
                if prim_path.startswith("/World/"):
                    # 排除 /World 本身
                    if prim_path != "/World":
                        world_paths.append(prim_path)
            
            # 排序路径
            world_paths.sort()
            
            print(f"✓ 从JSON文件中提取了 {len(world_paths)} 个物体路径")
            
            # 确定模板文件路径
            if template_yaml_path is None:
                template_path = _project_root / "config" / "Level2_Protocol1.yaml"
            else:
                template_path = Path(template_yaml_path)
                if not template_path.is_absolute():
                    template_path = _project_root / "config" / template_path
            
            # 确定输出文件路径
            if output_yaml_path is None:
                # 根据JSON文件名生成YAML文件名
                # equipment_20251127_161552.json -> equipment_20251127_161552.yaml
                json_stem = json_path.stem
                output_path = self.config_dir / f"{json_stem}.yaml"
            else:
                output_path = Path(output_yaml_path)
                if not output_path.is_absolute():
                    output_path = self.config_dir / output_path
            
            # 读取模板YAML文件
            if template_path.exists():
                if use_ruamel:
                    yaml_loader = YAML()
                    yaml_loader.preserve_quotes = True
                    with open(template_path, 'r', encoding='utf-8') as f:
                        yaml_data = yaml_loader.load(f)
                else:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        yaml_data = yaml.safe_load(f)
                print(f"✓ 已加载模板YAML文件: {template_path}")
            else:
                # 如果模板不存在，创建默认配置
                print(f"⚠ 模板YAML文件不存在: {template_path}，将创建默认配置")
                if use_ruamel:
                    yaml_data = CommentedMap()
                else:
                    yaml_data = {}
                
                # 设置默认值
                yaml_data['name'] = json_path.stem
                yaml_data['task_type'] = "all"
                yaml_data['controller_type'] = ""
                yaml_data['mode'] = "collect"
                yaml_data['usd_path'] = ""
                yaml_data['max_episodes'] = 1000
                yaml_data['task'] = CommentedMap() if use_ruamel else {}
                yaml_data['cameras'] = []
                yaml_data['robot'] = {
                    'type': 'franka', 
                    'position': [-0.4, 0, 0.66]
                }
                yaml_data['collector'] = {'type': 'default', 'compression': 'gzip'}
                yaml_data['hydra'] = {'run': {'dir': 'outputs/${mode}/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}'}}
                yaml_data['multi_run'] = {'run_dir': 'outputs/${mode}/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}'}
            
            # 更新关键字段
            yaml_data['task_type'] = "all"
            yaml_data['controller_type'] = ""
            
            # 转换USD路径为相对路径（相对于项目根目录）
            usd_path_obj = Path(scene_usd_path)
            try:
                # 尝试转换为相对于项目根目录的路径
                relative_usd_path = usd_path_obj.relative_to(_project_root)
                yaml_data['usd_path'] = str(relative_usd_path)
            except ValueError:
                # 如果无法转换为相对路径，使用绝对路径
                yaml_data['usd_path'] = str(usd_path_obj)
                print(f"⚠ 警告：USD路径无法转换为相对路径，使用绝对路径: {yaml_data['usd_path']}")
            
            # 更新 obj_paths
            if 'task' not in yaml_data:
                yaml_data['task'] = CommentedMap() if use_ruamel else {}
            
            if use_ruamel:
                obj_paths = CommentedSeq()
                for path in world_paths:
                    obj_paths.append(CommentedMap([('path', path)]))
            else:
                obj_paths = [{'path': path} for path in world_paths]
            
            yaml_data['task']['obj_paths'] = obj_paths
            
            # 保存YAML文件
            if use_ruamel:
                yaml_dumper = YAML()
                yaml_dumper.default_flow_style = False
                yaml_dumper.preserve_quotes = True
                yaml_dumper.width = 4096
                
                # 设置块格式
                def set_block_style(data):
                    if isinstance(data, CommentedMap):
                        data.fa.set_block_style()
                        for value in data.values():
                            if isinstance(value, (CommentedMap, CommentedSeq)):
                                set_block_style(value)
                    elif isinstance(data, CommentedSeq):
                        if len(data) > 0 and isinstance(data[0], (dict, CommentedMap)):
                            data.fa.set_block_style()
                        for item in data:
                            if isinstance(item, (CommentedMap, CommentedSeq)):
                                set_block_style(item)
                
                set_block_style(yaml_data)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml_dumper.dump(yaml_data, f)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            print(f"✓ YAML配置文件已生成: {output_path}")
            print(f"  包含 {len(world_paths)} 个物体路径")
            
            return str(output_path)
            
        except Exception as e:
            print(f"✗ 生成YAML配置文件失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def initialize_scene(
        self,
        protocol_text: str,
        prompt_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        z_height: Optional[float] = None,
        grid_resolution: Optional[float] = None,
        skip_optimization: bool = False,
        skip_update: bool = False,
        semantic_prompt: Optional[str] = None
    ) -> SceneInitializationResult:
        """
        执行完整的场景初始化流程
        
        Args:
            protocol_text: 协议文本
            prompt_path: 提示词文件路径
            api_key: API密钥
            base_url: API基础URL
            z_height: 放置高度
            grid_resolution: 网格分辨率
            skip_optimization: 是否跳过优化步骤
            skip_update: 是否跳过更新步骤
            semantic_prompt: 语义约束提示词（可选）
        
        Returns:
            SceneInitializationResult 对象
        """
        result = SceneInitializationResult()
        
        try:
            # 步骤1：提取协议信息
            equipment_file, actions_file = self.step1_extract_protocol(
                protocol_text,
                prompt_path=prompt_path,
                api_key=api_key,
                base_url=base_url
            )
            
            if not equipment_file or not actions_file:
                result.error_message = "步骤1失败：协议信息提取失败"
                return result
            
            result.equipment_file_path = equipment_file
            result.actions_file_path = actions_file
            
            # 步骤2：生成场景
            scene_usd_path = self.step2_generate_scene(equipment_file)
            
            if not scene_usd_path:
                result.error_message = "步骤2失败：场景生成失败"
                return result
            
            result.scene_usd_path = scene_usd_path
            
            # 步骤3：提取场景信息
            scene_json_path = self.step3_extract_scene(scene_usd_path)
            
            if not scene_json_path:
                result.error_message = "步骤3失败：场景信息提取失败"
                return result
            
            result.scene_json_path = scene_json_path
            
            # 步骤4：优化位置（可选）
            if not skip_optimization:
                optimized_json_path = self.step4_optimize_positions(
                    scene_json_path,
                    z_height=z_height,
                    grid_resolution=grid_resolution,
                    semantic_prompt=semantic_prompt,
                    actions_file=actions_file,
                    api_key=api_key,
                    base_url=base_url,
                )
                
                if not optimized_json_path:
                    result.error_message = "步骤4失败：位置优化失败"
                    return result
                
                result.optimized_json_path = optimized_json_path
            else:
                result.optimized_json_path = scene_json_path  # 使用未优化的JSON
            
            # 步骤5：更新场景（可选）
            if not skip_update:
                updated_usd_path = self.step5_update_scene(
                    result.optimized_json_path,
                    scene_usd_path=scene_usd_path,
                    in_place=True
                )
                
                if not updated_usd_path:
                    result.error_message = "步骤5失败：场景更新失败"
                    return result
                
                result.updated_usd_path = updated_usd_path
            else:
                result.updated_usd_path = scene_usd_path  # 使用未更新的USD
            
            # 步骤6：生成YAML配置文件
            yaml_config_path = self.step6_generate_yaml(
                result.optimized_json_path,
                result.updated_usd_path if result.updated_usd_path else result.scene_usd_path
            )
            
            if yaml_config_path:
                result.yaml_config_path = yaml_config_path
            else:
                print("⚠ 警告：YAML配置文件生成失败，但继续执行")
            
            result.success = True
            print("\n" + "="*60)
            print("场景初始化完成！")
            print("="*60)
            print(f"器材信息: {result.equipment_file_path}")
            print(f"动作信息: {result.actions_file_path}")
            print(f"场景USD: {result.scene_usd_path}")
            print(f"场景JSON: {result.scene_json_path}")
            if result.optimized_json_path != result.scene_json_path:
                print(f"优化JSON: {result.optimized_json_path}")
            if result.updated_usd_path != result.scene_usd_path:
                print(f"更新USD: {result.updated_usd_path}")
            if result.yaml_config_path:
                print(f"YAML配置: {result.yaml_config_path}")
            print("="*60)
            
        except Exception as e:
            result.error_message = f"场景初始化过程中发生错误: {e}"
            import traceback
            traceback.print_exc()
        finally:
            # 注意：不在这里关闭 SimulationApp，因为其他模块可能还在使用
            # 如果需要在程序结束时关闭，可以在 main 函数中处理
            pass
        
        return result

    def initialize_plan(self, plan, artifacts, registry):
        from agent.scene.legacy_scene_backend import LegacySceneBackend
        from agent.scene.scene_compiler import SceneCompiler

        root = Path(__file__).resolve().parents[2]
        compiler = SceneCompiler(registry, LegacySceneBackend(root), root)
        return compiler.compile(plan, artifacts)
    
    def cleanup(self):
        """清理资源（可选，通常不需要手动调用）"""
        global _simulation_app
        if _simulation_app is not None and self.manage_simulation_app:
            try:
                _simulation_app.close()
                _simulation_app = None
                print("[SceneInitializer] SimulationApp closed")
            except Exception as e:
                print(f"[SceneInitializer] Warning: Failed to close SimulationApp: {e}")


def initialize_scene(
    protocol_text: str,
    protocol_dir: Optional[str] = None,
    scenes_dir: Optional[str] = None,
    config_dir: Optional[str] = None,
    prompt_path: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    z_height: Optional[float] = None,
    grid_resolution: Optional[float] = None,
    skip_optimization: bool = False,
    skip_update: bool = False,
    semantic_prompt: Optional[str] = None
) -> SceneInitializationResult:
    """
    便捷函数：执行完整的场景初始化流程
    
    Args:
        protocol_text: 协议文本
        protocol_dir: 协议目录
        scenes_dir: 场景文件目录
        config_dir: 配置文件目录（存储YAML配置文件）
        prompt_path: 提示词文件路径
        api_key: API密钥
        base_url: API基础URL
        z_height: 放置高度
        grid_resolution: 网格分辨率
        skip_optimization: 是否跳过优化步骤
        skip_update: 是否跳过更新步骤
        semantic_prompt: 语义约束提示词（可选）
    
    Returns:
        SceneInitializationResult 对象
    """
    initializer = SceneInitializer(
        protocol_dir=protocol_dir,
        scenes_dir=scenes_dir,
        config_dir=config_dir
    )
    
    return initializer.initialize_scene(
        protocol_text=protocol_text,
        prompt_path=prompt_path,
        api_key=api_key,
        base_url=base_url,
        z_height=z_height,
        grid_resolution=grid_resolution,
        skip_optimization=skip_optimization,
        skip_update=skip_update,
        semantic_prompt=semantic_prompt
    )


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="场景初始化器：完整的场景生成和优化流程"
    )
    
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        help=(
            "协议文件名（如 protocol1.txt），默认从 "
            f"{_project_root / 'agent/protocol/protocols'} 目录读取，也可提供完整路径"
        )
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="提示词文件路径（默认使用 protocol_extract_prompt.txt）"
    )
    
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default=str(_project_root / "agent/protocol"),
        help="协议目录"
    )
    
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default=str(_project_root / "agent/scene/scenes"),
        help="场景文件目录"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API密钥（默认从环境变量读取）"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API基础URL"
    )
    
    parser.add_argument(
        "--z-height",
        type=float,
        default=None,
        help="放置高度（默认从布局配置读取）"
    )
    
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=None,
        help="网格分辨率（默认从布局配置读取）"
    )
    
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="跳过位置优化步骤"
    )
    
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="跳过场景更新步骤"
    )
    
    parser.add_argument(
        "--semantic-prompt",
        type=str,
        default=None,
        help="语义约束提示词（可选）"
    )
    
    args = parser.parse_args()
    
    # 读取协议文本
    # 默认从项目内的 agent/protocol/protocols/ 目录读取
    default_protocols_dir = _project_root / "agent/protocol/protocols"
    protocol_path = Path(args.protocol)
    
    # 如果路径不存在，尝试在默认目录和常见目录中查找
    if not protocol_path.exists():
        # 优先在默认 protocols 目录中查找
        possible_paths = [
            default_protocols_dir / args.protocol,  # 默认目录
            _project_root / "agent" / "protocol" / "protocols" / args.protocol,
            Path.cwd() / "agent" / "protocol" / "protocols" / args.protocol,
            Path.cwd() / args.protocol,
        ]
        
        found = False
        for possible_path in possible_paths:
            if possible_path.exists():
                protocol_path = possible_path
                print(f"[SceneInitializer] Found protocol file at: {protocol_path}")
                found = True
                break
        
        if not found:
            # 如果仍然找不到，检查是否可能是文本内容
            if len(args.protocol) < 50 and '\n' not in args.protocol:
                # 看起来像文件名，但文件不存在
                print(f"[SceneInitializer] Error: Protocol file not found: {args.protocol}")
                print(f"[SceneInitializer] Searched in:")
                for possible_path in possible_paths:
                    print(f"  - {possible_path}")
                sys.exit(1)
    
    if protocol_path.exists():
        with open(protocol_path, 'r', encoding='utf-8') as f:
            protocol_text = f.read()
        print(f"[SceneInitializer] Loaded protocol from file: {protocol_path}")
        print(f"[SceneInitializer] Protocol text length: {len(protocol_text)} chars")
        print(f"[SceneInitializer] Protocol text preview: {protocol_text[:200]}...")
    else:
        # 可能是直接的文本内容
        protocol_text = args.protocol
        print(f"[SceneInitializer] Using protocol text directly (length: {len(protocol_text)} chars)")
    
    if not protocol_text or not protocol_text.strip():
        print("[SceneInitializer] Error: Protocol text is empty!")
        sys.exit(1)
    
    # 执行初始化
    result = initialize_scene(
        protocol_text=protocol_text,
        protocol_dir=args.protocol_dir,
        scenes_dir=args.scenes_dir,
        prompt_path=args.prompt,
        api_key=args.api_key,
        base_url=args.base_url,
        z_height=args.z_height,
        grid_resolution=args.grid_resolution,
        skip_optimization=args.skip_optimization,
        skip_update=args.skip_update,
        semantic_prompt=args.semantic_prompt
    )
    
    # 清理 SimulationApp（如果是在命令行模式下）
    try:
        if _simulation_app is not None:
            _simulation_app.close()
            print("[SceneInitializer] SimulationApp closed")
    except Exception as e:
        print(f"[SceneInitializer] Warning: Failed to close SimulationApp: {e}")
    
    if result.success:
        sys.exit(0)
    else:
        print(f"\n✗ 场景初始化失败: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
