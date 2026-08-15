"""
Agent主程序：完整的场景初始化、代码生成和优化迭代流程

功能流程：
1. 场景初始化（scene_initializer）
2. 代码生成（code_generator）
3. 循环执行：
   - 调用action_orchestrator执行动作
   - 如果达到迭代上限仍未成功，调用continuous_optimizater优化场景
   - 使用position_updater更新USD场景
   - 重复执行直到成功
"""

import os
import sys
import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime
import multiprocessing

# 设置多进程启动模式为 spawn
try:
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

def _mp_initialize_scene(kwargs, result_file):
    """子进程：执行场景初始化"""
    try:
        # 清理 sys.argv 防止 SimulationApp 误解析父进程参数
        sys.argv = [sys.argv[0]]
        prepare_isaacsim_argv()
        
        from agent.scene.scene_initializer import SceneInitializer
        initializer = SceneInitializer(
            protocol_dir=kwargs['protocol_dir'],
            scenes_dir=kwargs['scenes_dir'],
            config_dir=kwargs['config_dir'],
            manage_simulation_app=True
        )
        result = initializer.initialize_scene(
            protocol_text=kwargs['protocol_text'],
            api_key=kwargs['api_key'],
            base_url=kwargs['base_url']
        )
        
        output = {"success": False, "data": None}
        if result.success:
            res_dict = {
                'scene_json_path': result.scene_json_path,
                'scene_usd_path': result.updated_usd_path,
                'actions_file_path': result.actions_file_path,
                'yaml_config_path': result.yaml_config_path,
                'equipment_file_path': result.equipment_file_path
            }
            output = {"success": True, "data": res_dict}
        else:
            output = {"success": False, "data": result.error_message}
        
        with open(result_file, 'w') as f:
            json.dump(output, f)
            
    except Exception as e:
        import traceback
        with open(result_file, 'w') as f:
            json.dump({"success": False, "data": f"{str(e)}\n{traceback.format_exc()}"}, f)

def _mp_optimize_scene(kwargs, result_file):
    """子进程：执行场景优化"""
    try:
        sys.argv = [sys.argv[0]]
        prepare_isaacsim_argv()
        from agent.scene.optimization.continuous_optimizater import ContinuousOptimizer
        optimizer = ContinuousOptimizer(
            scenes_dir=kwargs['scenes_dir'],
            api_key=kwargs['api_key'],
            base_url=kwargs['base_url']
        )
        
        if kwargs['log_file_path']:
            result = optimizer.optimize_scene(
                log_file_path=kwargs['log_file_path'],
                scene_json_path=kwargs['scene_json_path'],
                backup=True
            )
        else:
            result = optimizer.optimize_scene(
                scene_json_path=kwargs['scene_json_path'],
                timestamp=kwargs['timestamp'],
                backup=True
            )
        with open(result_file, 'w') as f:
            json.dump({"success": True, "data": result}, f)
    except Exception as e:
        import traceback
        with open(result_file, 'w') as f:
            json.dump({"success": False, "data": f"{str(e)}\n{traceback.format_exc()}"}, f)

def _mp_update_usd(kwargs, result_file):
    """子进程：执行 USD 更新"""
    try:
        sys.argv = [sys.argv[0]]
        prepare_isaacsim_argv()
        from agent.scene.optimization.position_updater import PositionUpdater
        updater = PositionUpdater(scenes_dir=kwargs['scenes_dir'])
        json_filename = Path(kwargs['scene_json_path']).name
        updated_path = updater.update_from_json_filename(
            json_filename=json_filename,
            in_place=True
        )
        with open(result_file, 'w') as f:
            json.dump({"success": True, "data": str(updated_path) if updated_path else None}, f)
    except Exception as e:
        import traceback
        with open(result_file, 'w') as f:
            json.dump({"success": False, "data": f"{str(e)}\n{traceback.format_exc()}"}, f)

# 添加项目根目录到Python路径
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.isaacsim_runtime import prepare_isaacsim_argv


class AgentMain:
    """Agent主控制器：协调完整的初始化、生成和优化流程"""
    
    def __init__(
        self,
        protocol_text: Optional[str] = None,
        protocol_file: Optional[str] = None,
        protocol_dir: Optional[str] = None,
        scenes_dir: Optional[str] = None,
        config_dir: Optional[str] = None,
        controllers_dir: Optional[str] = None,
        max_action_iterations: int = 5,
        max_scene_optimizations: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化Agent主控制器
        
        Args:
            protocol_text: 协议文本（如果为None，从protocol_file读取）
            protocol_file: 协议文件路径
            protocol_dir: 协议目录
            scenes_dir: 场景文件目录
            config_dir: 配置文件目录
            controllers_dir: 控制器目录
            max_action_iterations: action_orchestrator的最大迭代次数
            max_scene_optimizations: 场景优化的最大次数
            api_key: OpenAI API密钥
            base_url: API基础URL
        """
        self.protocol_text = protocol_text
        self.protocol_file = Path(protocol_file) if protocol_file else None
        self.protocol_dir = (
            Path(protocol_dir)
            if protocol_dir is not None
            else _project_root / "agent" / "protocol"
        )
        self.scenes_dir = (
            Path(scenes_dir)
            if scenes_dir is not None
            else _project_root / "agent" / "scene" / "scenes"
        )
        self.config_dir = (
            Path(config_dir)
            if config_dir is not None
            else _project_root / "config"
        )
        self.controllers_dir = (
            Path(controllers_dir)
            if controllers_dir is not None
            else _project_root / "controllers"
        )
        self.max_action_iterations = max_action_iterations
        self.max_scene_optimizations = max_scene_optimizations
        self.api_key = api_key
        self.base_url = base_url
        
        # 确保目录存在
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.controllers_dir.mkdir(parents=True, exist_ok=True)
    
    def step1_initialize_scene(self) -> Optional[dict]:
        """
        步骤1：初始化场景 (多进程模式以释放 GPU)
        
        Returns:
            场景初始化结果字典，包含各种文件路径，如果失败则返回None
        """
        print("\n" + "="*80)
        print("步骤 1: 场景初始化")
        print("="*80)
        
        try:
            # 读取协议文本
            if self.protocol_text is None:
                if self.protocol_file is None:
                    raise ValueError("必须提供protocol_text或protocol_file")
                
                if not self.protocol_file.is_absolute():
                    # 尝试在protocol目录下查找
                    protocol_path = self.protocol_dir / "protocols" / self.protocol_file
                    if not protocol_path.exists():
                        protocol_path = self.protocol_file
                else:
                    protocol_path = self.protocol_file
                
                if not protocol_path.exists():
                    raise FileNotFoundError(f"协议文件不存在: {protocol_path}")
                
                with open(protocol_path, 'r', encoding='utf-8') as f:
                    self.protocol_text = f.read()
            
            if not self.protocol_text or not self.protocol_text.strip():
                raise ValueError("协议文本为空")

            # 准备多进程调用，使用临时文件传递结果
            kwargs = {
                'protocol_dir': str(self.protocol_dir),
                'scenes_dir': str(self.scenes_dir),
                'config_dir': str(self.config_dir),
                'protocol_text': self.protocol_text,
                'api_key': self.api_key,
                'base_url': self.base_url
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
                result_file = tf.name
            
            try:
                process = multiprocessing.Process(target=_mp_initialize_scene, args=(kwargs, result_file))
                process.start()
                process.join() # 等待进程结束
                
                # 从文件读取结果
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        output = json.load(f)
                    success = output.get("success", False)
                    result = output.get("data")
                else:
                    success = False
                    result = "子进程未生成结果文件"
            finally:
                if os.path.exists(result_file):
                    os.unlink(result_file)
            
            if success:
                print(f"✓ 场景初始化成功")
                print(f"  - 场景JSON: {result['scene_json_path']}")
                print(f"  - 场景USD: {result['scene_usd_path']}")
                print(f"  - 动作信息: {result['actions_file_path']}")
                print(f"  - YAML配置: {result['yaml_config_path']}")
                return result
            else:
                print(f"✗ 场景初始化失败: {result}")
                return None
                
        except Exception as e:
            print(f"✗ 场景初始化出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step2_generate_controller(self, actions_file_path: str) -> Optional[Tuple[str, str, str]]:
        """
        步骤2：生成控制器代码
        
        Args:
            actions_file_path: 动作信息文件路径
        
        Returns:
            (控制器文件路径, 类名, 注册名称)，如果失败则返回None
        """
        print("\n" + "="*80)
        print("步骤 2: 生成控制器代码")
        print("="*80)
        
        try:
            from agent.action.generation.code_generator import generate_controller_code
            
            # 生成控制器代码
            controller_path, class_name, register_name = generate_controller_code(
                action_info_path=actions_file_path,
                controllers_dir=str(self.controllers_dir),
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            print(f"✓ 控制器代码生成成功")
            print(f"  - 控制器文件: {controller_path}")
            print(f"  - 类名: {class_name}")
            print(f"  - 注册名称: {register_name}")
            
            return (str(controller_path), class_name, register_name)
            
        except Exception as e:
            print(f"✗ 控制器代码生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step3_optimize_scene(
        self,
        scene_json_path: str,
        log_file_path: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        步骤3：优化场景 (多进程模式以释放 GPU)
        """
        print("\n" + "="*80)
        print("步骤 3: 优化场景")
        print("="*80)
        
        try:
            kwargs = {
                'scenes_dir': str(self.scenes_dir),
                'api_key': self.api_key,
                'base_url': self.base_url,
                'log_file_path': log_file_path,
                'scene_json_path': scene_json_path,
                'timestamp': timestamp
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
                result_file = tf.name
            
            try:
                process = multiprocessing.Process(target=_mp_optimize_scene, args=(kwargs, result_file))
                process.start()
                process.join()
                
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        output = json.load(f)
                    success = output.get("success", False)
                    result = output.get("data")
                else:
                    success = False
                    result = "子进程未生成结果文件"
            finally:
                if os.path.exists(result_file):
                    os.unlink(result_file)
            
            if success and result:
                print(f"✓ 场景优化成功")
                return True
            else:
                print(f"✗ 场景优化失败: {result}")
                return False
        except Exception as e:
            print(f"✗ 场景优化出错: {e}")
            return False

    def step4_update_usd(self, scene_json_path: str) -> bool:
        """
        步骤4：更新USD场景文件 (多进程模式以释放 GPU)
        """
        print("\n" + "="*80)
        print("步骤 4: 更新USD场景文件")
        print("="*80)
        
        try:
            kwargs = {
                'scenes_dir': str(self.scenes_dir),
                'scene_json_path': scene_json_path
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
                result_file = tf.name
            
            try:
                process = multiprocessing.Process(target=_mp_update_usd, args=(kwargs, result_file))
                process.start()
                process.join()
                
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        output = json.load(f)
                    success = output.get("success", False)
                    updated_path = output.get("data")
                else:
                    success = False
                    updated_path = "子进程未生成结果文件"
            finally:
                if os.path.exists(result_file):
                    os.unlink(result_file)
            
            if success and updated_path:
                print(f"✓ USD场景更新成功: {updated_path}")
                return True
            else:
                print(f"✗ USD场景更新失败: {updated_path}")
                return False
        except Exception as e:
            print(f"✗ USD场景更新出错: {e}")
            return False
    
    def step5_run_action_orchestrator(
        self,
        controller_file: str,
        config_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        步骤5：运行动作编排器
        
        Args:
            controller_file: 控制器文件路径
            config_name: 配置文件名（不含.yaml）
        
        Returns:
            (是否成功, 合并后的日志文件路径)
        """
        print("\n" + "="*80)
        print("步骤 5: 运行动作编排器")
        print("="*80)
        
        try:
            from agent.action.action_orchestrator import AgentOrchestrator
            
            # 创建编排器
            orchestrator = AgentOrchestrator(
                controller_file=controller_file,
                config_name=config_name,
                config_dir=str(self.config_dir),
                max_iterations=self.max_action_iterations,
                project_root=str(_project_root)
            )
            
            # 运行
            success = orchestrator.run()
            
            # 获取合并后的日志文件路径
            merged_log_path = None
            if hasattr(orchestrator, 'run_dir'):
                # 查找合并后的日志文件
                merged_logs = list(orchestrator.run_dir.glob("*_all_iterations_*.txt"))
                if merged_logs:
                    merged_log_path = str(merged_logs[0])
                else:
                    # 如果没有合并日志，尝试查找最新的迭代日志
                    iterations_dir = orchestrator.run_dir / "iterations"
                    if iterations_dir.exists():
                        # 查找所有迭代目录中的execution_log.txt
                        log_files = list(iterations_dir.glob("*/execution_log.txt"))
                        if log_files:
                            # 返回最新的日志文件
                            merged_log_path = str(max(log_files, key=lambda p: p.stat().st_mtime))
            
            if success:
                print(f"✓ 动作执行成功")
            else:
                print(f"✗ 动作执行失败（达到最大迭代次数）")
                if merged_log_path:
                    print(f"  日志文件: {merged_log_path}")
            
            return success, merged_log_path
            
        except Exception as e:
            print(f"✗ 动作编排器运行出错: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def step6_rate_trajectory(
        self,
        config_name: str,
        log_path: Optional[str] = None
    ) -> Optional[Dict]:
        """
        步骤6：轨迹评分
        
        Args:
            config_name: 配置文件名（不含.yaml）
            log_path: 日志文件路径（用于查找轨迹文件）
        
        Returns:
            评分结果字典，如果失败则返回None
        """
        print("\n" + "="*80)
        print("步骤 6: 轨迹评分")
        print("="*80)
        
        try:
            from agent.action.rating.trajectory_analyzer import TrajectoryAnalyzer
            from agent.action.rating.trajectory_recorder import TrajectoryRecorder
            
            # 查找轨迹文件
            trajectory_file = None
            if log_path:
                log_dir = Path(log_path).parent
                # 查找轨迹文件（可能在迭代目录中）
                trajectory_files = list(log_dir.glob("**/trajectory.json"))
                if trajectory_files:
                    trajectory_file = trajectory_files[-1]  # 使用最新的
            
            if trajectory_file and trajectory_file.exists():
                print(f"[Rating] 找到轨迹文件: {trajectory_file}")
                
                # 加载轨迹
                recorder = TrajectoryRecorder()
                recorder.load(str(trajectory_file))
                positions, _ = recorder.get_trajectory()
                
                if len(positions) < 2:
                    print(f"[Rating] ⚠️ 轨迹数据不足（{len(positions)} 个点），无法评分")
                    return None
                
                # 分析轨迹
                analyzer = TrajectoryAnalyzer(
                    collision_threshold=0.05,
                    stall_threshold=0.001,
                    stall_duration=30
                )
                
                result = analyzer.analyze(positions)
                
                # 输出评分结果
                print(f"\n[Rating] 轨迹分析结果:")
                print(result.analysis_summary)
                
                # 保存评分结果
                rating_file = trajectory_file.parent / "trajectory_rating.json"
                rating_data = {
                    'smoothness_score': result.smoothness_score,
                    'collision_count': result.collision_count,
                    'collision_frames': result.collision_frames,
                    'stall_count': result.stall_count,
                    'stall_frames': result.stall_frames,
                    'max_velocity': result.max_velocity,
                    'mean_velocity': result.mean_velocity,
                    'total_distance': result.total_distance,
                    'trajectory_length': len(positions),
                    'analysis_summary': result.analysis_summary
                }
                
                import json
                with open(rating_file, 'w', encoding='utf-8') as f:
                    json.dump(rating_data, f, indent=2)
                
                print(f"[Rating] ✓ 评分结果已保存: {rating_file}")
                
                return rating_data
            else:
                print(f"[Rating] ⚠️ 未找到轨迹文件，跳过评分")
                if log_path:
                    print(f"[Rating]   日志路径: {log_path}")
                return None
                
        except ImportError as e:
            print(f"[Rating] ⚠️ 无法导入轨迹评分模块: {e}")
            print(f"[Rating]   跳过评分步骤")
            return None
        except Exception as e:
            print(f"[Rating] ✗ 轨迹评分出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_timestamp_from_path(self, file_path: str) -> Optional[str]:
        """
        从文件路径中提取时间戳
        
        Args:
            file_path: 文件路径
        
        Returns:
            时间戳字符串（格式：YYYYMMDD_HHMMSS），如果未找到则返回None
        """
        import re
        filename = Path(file_path).name
        # 匹配格式：YYYYMMDD_HHMMSS
        pattern = r'(\d{8}_\d{6})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None
    
    def run(self) -> bool:
        """
        运行完整的流程
        
        Returns:
            是否成功完成
        """
        print("\n" + "="*80)
        print("Agent Main - 开始完整流程")
        print("="*80)
        
        # 步骤1：初始化场景
        scene_result = self.step1_initialize_scene()
        if not scene_result:
            print("\n✗ 场景初始化失败，流程终止")
            return False
        
        scene_json_path = scene_result['scene_json_path']
        scene_usd_path = scene_result['scene_usd_path']
        actions_file_path = scene_result['actions_file_path']
        yaml_config_path = scene_result['yaml_config_path']
        
        # 提取时间戳（用于后续匹配日志文件）
        timestamp = self.extract_timestamp_from_path(scene_json_path)
        
        # 步骤2：生成控制器代码
        controller_result = self.step2_generate_controller(actions_file_path)
        if not controller_result:
            print("\n✗ 控制器代码生成失败，流程终止")
            return False
        
        controller_file, class_name, register_name = controller_result
        
        # 从YAML配置文件名提取config_name（不含.yaml）
        config_name = Path(yaml_config_path).stem if yaml_config_path else None
        if not config_name:
            # 如果无法从YAML获取，尝试从controller文件名推断
            config_name = Path(controller_file).stem.replace('_controller', '')
        
        # 主循环：执行动作编排和场景优化
        scene_optimization_count = 0
        
        while scene_optimization_count < self.max_scene_optimizations:
            print(f"\n{'='*80}")
            print(f"场景优化轮次: {scene_optimization_count + 1}/{self.max_scene_optimizations}")
            print(f"{'='*80}")
            
            # 步骤5：运行动作编排器
            success, merged_log_path = self.step5_run_action_orchestrator(
                controller_file=controller_file,
                config_name=config_name
            )
            
            # 如果成功，进行轨迹评分
            if success:
                print(f"\n{'='*80}")
                print(f"✓✓✓ 任务成功完成！")
                print(f"{'='*80}")
                
                # 步骤6：轨迹评分
                self.step6_rate_trajectory(config_name=config_name, log_path=merged_log_path)
                
                return True
            
            # 如果失败且未达到场景优化上限，直接修改物体位置
            if scene_optimization_count < self.max_scene_optimizations - 1:
                print(f"\n动作执行失败，开始修改物体位置...")
                
                # 步骤3：优化场景（使用合并后的日志文件）
                if not self.step3_optimize_scene(
                    scene_json_path=scene_json_path,
                    log_file_path=merged_log_path,
                    timestamp=timestamp
                ):
                    print(f"✗ 场景优化失败，继续尝试...")
                    scene_optimization_count += 1
                    continue
                
                # 步骤4：更新USD场景（修改物体位置）
                if not self.step4_update_usd(scene_json_path):
                    print(f"✗ USD场景更新失败，继续尝试...")
                    scene_optimization_count += 1
                    continue
                
                scene_optimization_count += 1
                print(f"\n物体位置修改完成，准备重新执行动作...")
            else:
                # 达到场景优化上限，退出循环
                break
        
        # 所有尝试都失败
        print(f"\n{'='*80}")
        print(f"✗✗✗ 任务失败：已达到最大场景优化次数 ({self.max_scene_optimizations})")
        print(f"{'='*80}")
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent validated planning and execution pipeline",
    )
    parser.add_argument("--protocol-text", default=None)
    parser.add_argument("--protocol", "--protocol-file", dest="protocol_file", default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Re-execute an existing run directory without calling the LLM",
    )
    parser.add_argument("--protocol-dir", default=None)
    parser.add_argument("--scenes-dir", default=None)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--controllers-dir", default="controllers")
    parser.add_argument(
        "--execution-backend",
        choices=("plan_executor", "legacy_codegen"),
        default="plan_executor",
    )
    parser.add_argument(
        "--allow-unsafe-codegen",
        action="store_true",
        help="explicitly allow the legacy backend to execute LLM-generated Python",
    )
    parser.add_argument("--max-plan-attempts", type=int, default=3)
    parser.add_argument("--max-action-iterations", type=int, default=5)
    parser.add_argument("--max-scene-optimizations", type=int, default=3)
    parser.add_argument("--run-root", default="outputs/action_agent")
    parser.add_argument("--simulation-timeout", type=int, default=600)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def read_protocol_input(args) -> str:
    if args.protocol_text:
        return args.protocol_text
    if not args.protocol_file:
        raise ValueError("--protocol-text or --protocol-file is required")
    path = Path(args.protocol_file)
    if not path.is_absolute():
        if path.is_file():
            return path.read_text(encoding="utf-8")
        protocol_dir = getattr(args, "protocol_dir", None)
        if protocol_dir is None:
            protocol_dir = _project_root / "agent" / "protocol"
        else:
            protocol_dir = Path(protocol_dir)
            if not protocol_dir.is_absolute():
                protocol_dir = _project_root / protocol_dir
        path = protocol_dir / "protocols" / path
    return path.read_text(encoding="utf-8")


def resume_run(args):
    from agent.action.plan_orchestrator import SubprocessSimulationRunner
    from agent.planning.models import AgentPlan
    from agent.planning.protocol_planner import ProtocolPlanningService
    from agent.planning.registry import CapabilityRegistry
    from agent.planning.validator import PlanValidator, ValidationReport
    from agent.runtime.run_artifacts import RunArtifacts
    from agent.scene.isaac_scene_worker import IsaacSubprocessSceneBackend
    from agent.scene.scene_compiler import SceneCompiler

    artifacts = RunArtifacts.open(args.resume)
    plan = AgentPlan.model_validate_json(
        artifacts.plan_path.read_text(encoding="utf-8")
    )
    ProtocolPlanningService._normalize_instance_names(plan)
    stored_validation = ValidationReport.model_validate_json(
        artifacts.validation_path.read_text(encoding="utf-8")
    )
    validator = PlanValidator(
        CapabilityRegistry.load_default(_project_root)
    )
    current_validation = validator.validate(plan)
    if not current_validation.valid:
        raise ValueError(
            "resume plan no longer matches the current capability registry"
        )
    if current_validation.plan_fingerprint != stored_validation.plan_fingerprint:
        artifacts.write_plan(plan)
        artifacts.write_json(artifacts.validation_path, current_validation)
    elif current_validation != stored_validation:
        raise ValueError(
            "resume validation report no longer matches the current registry"
        )
    compiler = SceneCompiler(
        CapabilityRegistry.load_default(_project_root),
        IsaacSubprocessSceneBackend(
            _project_root,
            python_executable=sys.executable,
            timeout=args.simulation_timeout,
        ),
        _project_root,
    )
    compiled = compiler.compile(plan, artifacts)

    raw_report = SubprocessSimulationRunner(
        _project_root,
        python_executable=sys.executable,
        timeout=args.simulation_timeout,
        headless=args.headless,
    ).run(compiled.config_path)
    report = dict(raw_report)
    report.update(
        {
            "status": "completed" if report["execution_success"] else "execution_failed",
            "resumed": True,
            "run_dir": str(artifacts.run_dir),
        }
    )
    if not report["execution_success"]:
        report["failure_type"] = "action_execution_failed"
        report["error_code"] = next(
            (
                step.get("verification", {}).get("code")
                for step in reversed(report.get("steps", []))
                if step.get("step_id") == report.get("failed_step")
                and step.get("verification", {}).get("code")
            ),
            "ACTION_EXECUTION_FAILED",
        )
    artifacts.write_json(artifacts.execution_report_path, report)
    return report


def build_plan_pipeline(args):
    from agent.action.optimization.plan_parameter_optimizer import (
        PlanParameterOptimizer,
    )
    from agent.action.plan_orchestrator import (
        PlanOrchestrator,
        SubprocessSimulationRunner,
    )
    from agent.plan_pipeline import ActionAgentPipeline, LegacyCodegenBackend
    from agent.planning.llm_client import OpenAIChatClient
    from agent.planning.protocol_planner import ProtocolPlanningService
    from agent.planning.registry import CapabilityRegistry
    from agent.planning.validator import PlanValidator
    from agent.scene.isaac_scene_worker import IsaacSubprocessSceneBackend
    from agent.scene.optimization.continuous_optimizater import ContinuousOptimizer
    from agent.scene.scene_compiler import SceneCompiler

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    model = os.getenv("ACTION_AGENT_MODEL")
    if not model:
        raise ValueError("ACTION_AGENT_MODEL is required")
    base_url = os.getenv("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAIChatClient(**client_kwargs)

    registry = CapabilityRegistry.load_default(_project_root)
    validator = PlanValidator(registry)
    planner = ProtocolPlanningService(
        client,
        validator,
        _project_root,
        model=model,
    )
    scene_backend = IsaacSubprocessSceneBackend(
        _project_root,
        python_executable=sys.executable,
        timeout=args.simulation_timeout,
    )
    compiler = SceneCompiler(registry, scene_backend, _project_root)
    runner = SubprocessSimulationRunner(
        _project_root,
        python_executable=sys.executable,
        timeout=args.simulation_timeout,
        headless=args.headless,
    )
    parameter_optimizer = PlanParameterOptimizer(
        client,
        registry,
        model=model,
    )

    def scene_optimizer_factory(artifacts, scene_compile_result):
        return ContinuousOptimizer(
            client=client.client,
            scenes_dir=str(artifacts.run_dir),
            scene_json_path=scene_compile_result.scene_json_path,
            scene_usd_path=scene_compile_result.usd_path,
            layout_profile=compiler.layout_profile,
            model=model,
            position_updater_factory=scene_backend.position_updater_factory,
        )

    orchestrator = PlanOrchestrator(
        runner,
        parameter_optimizer,
        scene_optimizer_factory,
        registry=registry,
        validator=validator,
        scene_preflight=compiler.preflight,
        max_parameter_iterations=args.max_action_iterations,
        max_scene_iterations=args.max_scene_optimizations,
    )

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = _project_root / run_root
    controllers_dir = Path(args.controllers_dir)
    if not controllers_dir.is_absolute():
        controllers_dir = _project_root / controllers_dir
    return ActionAgentPipeline(
        planner,
        compiler,
        orchestrator,
        LegacyCodegenBackend(
            controllers_dir,
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_iterations=args.max_action_iterations,
            python_executable=sys.executable,
            project_root=_project_root,
            headless=args.headless,
        ),
        run_root,
    )


def run_from_args(args):
    if args.resume is not None:
        return resume_run(args)
    protocol_text = read_protocol_input(args)
    pipeline = build_plan_pipeline(args)
    return pipeline.run(
        protocol_text,
        execution_backend=args.execution_backend,
        max_plan_attempts=args.max_plan_attempts,
    )


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.execution_backend == "legacy_codegen" and not args.allow_unsafe_codegen:
        parser.error(
            "legacy_codegen executes generated Python; add --allow-unsafe-codegen "
            "only in an isolated environment"
        )
    new_run = bool(args.protocol_text or args.protocol_file)
    if new_run == (args.resume is not None):
        parser.error(
            "provide exactly one of --protocol/--protocol-text or --resume"
        )
    return run_from_args(args)


if __name__ == '__main__':
    result = main()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("execution_success") else 1)
