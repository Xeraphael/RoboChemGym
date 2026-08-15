"""
Agent Orchestrator - 总控制器

负责协调代码优化迭代流程：
1. 读取初始生成的controller代码
2. 在仿真环境中执行并监控
3. 收集日志信息
4. 调用优化器生成改进代码
5. 迭代以上过程直到成功或达到最大迭代次数

使用方式：
1. 作为模块调用：
   from agent.action.action_orchestrator import AgentOrchestrator
   orchestrator = AgentOrchestrator(...)
   success = orchestrator.run()

2. 命令行调用：
   python agent/action/action_orchestrator.py --controller ... --config-name ...
"""

import os
import sys
import argparse
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List


class AgentOrchestrator:
    """总控制器：协调优化迭代流程"""
    
    def __init__(
        self,
        controller_file: str,
        config_name: str,
        max_iterations: int = 5,
        log_dir: str = "agent/logs",
        python_executable: Optional[str] = None,
        project_root: Optional[str] = None,
        run_dir: Optional[str] = None,
        config_dir: Optional[str] = None,
        headless: bool = False,
    ):
        """
        初始化总控制器
        
        Args:
            controller_file: controller文件路径（如 "controllers/protocol1_controller.py"）
            config_name: 配置文件名（不含.yaml，如 "level1_pick"）
            max_iterations: 最大迭代次数
            log_dir: 日志目录（已废弃，仅保留兼容性）
            python_executable: Python解释器路径（IsaacSim的python），None则使用当前
            project_root: 项目根目录（如果为None，自动检测）
            run_dir: 运行输出目录（如果为None，自动生成）
        """
        self.controller_file = Path(controller_file)
        self.config_name = config_name
        self.max_iterations = max_iterations
        self.log_dir = Path(log_dir)  # 保留但不再使用
        
        # 设置Python解释器（通常使用当前环境即可）
        self.python_executable = python_executable or sys.executable
        
        # 设置项目根目录
        if project_root:
            self.project_root = Path(project_root)
        else:
            # 自动检测：从当前文件位置向上查找
            # __file__ 是 agent/action/action_orchestrator.py
            # parent.parent.parent 是项目根目录
            self.project_root = Path(__file__).parent.parent.parent

        if config_dir is None:
            self.config_dir = self.project_root / "config"
        else:
            config_path = Path(config_dir)
            self.config_dir = (
                config_path
                if config_path.is_absolute()
                else self.project_root / config_path
            )
        self.headless = bool(headless)
        
        # 验证controller文件存在
        if not self.controller_file.exists():
            # 尝试相对于项目根目录查找
            controller_abs = self.project_root / self.controller_file
            if controller_abs.exists():
                self.controller_file = controller_abs
            else:
                raise FileNotFoundError(f"Controller file not found: {controller_file}")
        
        # 确保 controller_file 是绝对路径
        if not self.controller_file.is_absolute():
            self.controller_file = self.project_root / self.controller_file
        
        # --- 新的目录结构设计 ---
        # outputs/optimization/{date}/{time}_{name}/
        if run_dir:
            self.run_dir = Path(run_dir)
        else:
            now = datetime.now()
            date_str = now.strftime("%Y.%m.%d")
            time_str = now.strftime("%H.%M.%S")
            controller_name = self.controller_file.stem
            
            # 根输出目录
            self.run_dir = self.project_root / "outputs" / "optimization" / date_str / f"{time_str}_{controller_name}"
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 代码历史目录 (替代原来的 agent/backups)
        self.history_dir = self.run_dir / "source_history"
        self.history_dir.mkdir(exist_ok=True)
        
        # 2. 迭代数据目录 (日志、分析)
        self.iterations_dir = self.run_dir / "iterations"
        self.iterations_dir.mkdir(exist_ok=True)
        
        print(f"[Orchestrator] Initialized for controller: {self.controller_file}")
        print(f"[Orchestrator] Config: {config_name}")
        print(f"[Orchestrator] Max iterations: {max_iterations}")
        print(f"[Orchestrator] Run directory: {self.run_dir}")
        print(f"[Orchestrator] Project root: {self.project_root}")
        
    def backup_controller(self, iteration: int) -> Path:
        """
        备份当前controller代码到历史目录
        
        Args:
            iteration: 迭代次数
        
        Returns:
            备份文件路径
        """
        # 保存为 iter_{N}_before.py
        backup_path = self.history_dir / f"iter_{iteration}_before.py"
        shutil.copy2(self.controller_file, backup_path)
        print(f"[Orchestrator] Saved source snapshot: {backup_path}")
        return backup_path
    
    def execute_simulation(self, iteration: int) -> Tuple[bool, Path]:
        """
        执行仿真并监控
        
        Args:
            iteration: 当前迭代次数
            
        Returns:
            (是否成功, 日志文件路径)
        """
        print(f"\n[Orchestrator] === Iteration {iteration}: Executing Simulation ===")
        
        # 创建迭代日志目录: iterations/{N}/
        iter_dir = self.iterations_dir / str(iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        log_file = iter_dir / "execution_log.txt"
        
        # 设置环境变量，告诉仿真环境启用监控模式
        env = os.environ.copy()
        env["AGENT_MONITOR_MODE"] = "true"
        env["AGENT_LOG_FILE"] = str(log_file.absolute())
        env["AGENT_ITERATION"] = str(iteration)
        
        # 构建命令 - 直接运行 main.py
        cmd = [
            self.python_executable,
            "main.py",
            "--config-dir",
            str(self.config_dir),
            "--config-name",
            self.config_name,
            "--no-video",
        ]
        if self.headless:
            cmd.append("--headless")
        
        print(f"[Orchestrator] Python: {self.python_executable}")
        print(f"[Orchestrator] Working directory: {self.project_root}")
        print(f"[Orchestrator] Running command: {' '.join(cmd)}")
        print(f"[Orchestrator] Log file: {log_file}")
        
        try:
            # 运行仿真（在项目根目录执行）
            result = subprocess.run(
                cmd,
                env=env,
                cwd=str(self.project_root),  # 设置工作目录为项目根目录
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            # 仅在失败时保存stderr以便排查
            if result.returncode != 0 or not log_file.exists():
                (iter_dir / "stderr.txt").write_text(result.stderr)
                if result.stdout:
                    (iter_dir / "stdout.txt").write_text(result.stdout)
                
                # 如果日志不存在，打印错误输出以便即时调试
                if not log_file.exists():
                    print(f"[Orchestrator] ✗ Error: Log file was not created. Simulation output:")
                    print("-" * 40)
                    print(result.stderr if result.stderr else "(No stderr output)")
                    print("-" * 40)
            
            # 检查进程返回码
            process_success = result.returncode == 0
            
            # 【关键修复】检查日志文件中的实际执行结果
            action_failed = None
            task_completed = False
            
            if log_file.exists():
                try:
                    log_content = log_file.read_text(encoding='utf-8')
                    # 检查是否有 ACTION_FAILED 标记
                    for line in log_content.split('\n'):
                        if 'ACTION_FAILED=' in line:
                            # 提取失败标志
                            try:
                                action_failed = int(line.split('ACTION_FAILED=')[1].strip())
                            except (ValueError, IndexError):
                                pass
                        if 'Task Complete:' in line:
                            task_completed = True
                except Exception as e:
                    print(f"[Orchestrator] Warning: Failed to read log file: {e}")
            else:
                print(f"[Orchestrator] Warning: Log file does not exist: {log_file}")
            
            # 综合判断：进程成功 + 日志文件存在 + 任务完成 + 无动作失败
            if process_success and log_file.exists() and task_completed:
                if action_failed == 0:
                    success = True
                    print(f"[Orchestrator] ✓ Simulation completed successfully")
                    print(f"[Orchestrator]   - Process exit code: {result.returncode}")
                    print(f"[Orchestrator]   - Task completed: {task_completed}")
                    print(f"[Orchestrator]   - Action failed flag: {action_failed}")
                elif action_failed == 1:
                    success = False
                    print(f"[Orchestrator] ✗ Simulation failed: Action failed flag = 1")
                    print(f"[Orchestrator]   - Process exit code: {result.returncode}")
                    print(f"[Orchestrator]   - Task completed: {task_completed}")
                    print(f"[Orchestrator]   - Action failed flag: {action_failed}")
                else:
                    # 日志文件存在但无法确定失败状态，保守判断为失败
                    success = False
                    print(f"[Orchestrator] ✗ Simulation status unclear: Could not determine action failed flag")
                    print(f"[Orchestrator]   - Process exit code: {result.returncode}")
                    print(f"[Orchestrator]   - Log file exists: {log_file.exists()}")
                    print(f"[Orchestrator]   - Task completed: {task_completed}")
                    print(f"[Orchestrator]   - Action failed flag: {action_failed}")
            elif process_success and not log_file.exists():
                # 进程成功但日志文件不存在，可能是监控未启用或提前退出
                success = False
                print(f"[Orchestrator] ✗ Simulation failed: Log file not created")
                print(f"[Orchestrator]   - Process exit code: {result.returncode}")
                print(f"[Orchestrator]   - Log file exists: {log_file.exists()}")
                print(f"[Orchestrator]   - Possible causes: Controller import error, early exit, or monitor not enabled")
            else:
                # 进程失败
                success = False
                print(f"[Orchestrator] ✗ Simulation failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"[Orchestrator] Error output: {result.stderr[:500]}")
            
            return success, log_file
            
        except subprocess.TimeoutExpired:
            print(f"[Orchestrator] ✗ Simulation timeout after 600 seconds")
            return False, log_file
        except Exception as e:
            print(f"[Orchestrator] ✗ Error running simulation: {e}")
            return False, log_file
    
    def call_optimizer(self, iteration: int, log_files: List[Path]) -> Optional[str]:
        """
        调用参数优化器
        
        Args:
            iteration: 当前迭代次数
            log_files: 所有迭代的日志文件列表
            
        Returns:
            优化后的代码字符串，如果优化失败则返回None
        """
        print(f"\n[Orchestrator] === Iteration {iteration}: Calling Optimizer ===")
        
        try:
            # 动态导入优化器
            # 添加 agent/action 目录到路径（如果还没有）
            agent_action_path = Path(__file__).parent
            if str(agent_action_path) not in sys.path:
                sys.path.insert(0, str(agent_action_path))
            from optimization.parameter_optimizer import ParameterOptimizer
            
            # 读取当前代码
            original_code = self.controller_file.read_text(encoding='utf-8')
            
            # 创建优化器
            optimizer = ParameterOptimizer()
            
            # 执行优化
            analysis_text, optimized_code = optimizer.optimize(
                original_code=original_code,
                log_files=log_files,
                iteration=iteration
            )
            
            if optimized_code:
                print(f"[Orchestrator] ✓ Optimization completed")
                
                iter_dir = self.iterations_dir / str(iteration)
                
                # 保存分析文本
                if analysis_text:
                    analysis_path = iter_dir / "optimization_analysis.txt"
                    analysis_path.write_text(analysis_text, encoding='utf-8')
                    print(f"[Orchestrator] Saved analysis to: {analysis_path}")
                else:
                    print(f"[Orchestrator] Warning: No analysis text extracted")
                
                # 不再保存 optimized_code.py，因为会直接更新到 controller 文件
                
                return optimized_code
            else:
                print(f"[Orchestrator] ✗ Optimization failed")
                return None
                
        except Exception as e:
            print(f"[Orchestrator] ✗ Error calling optimizer: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def update_controller(self, new_code: str, iteration: int) -> bool:
        """
        更新controller文件并保存优化后的代码到历史记录
        
        Args:
            new_code: 新的代码内容
            iteration: 当前迭代次数
            
        Returns:
            是否更新成功
        """
        try:
            # 直接写入新代码到源文件
            self.controller_file.write_text(new_code, encoding='utf-8')
            print(f"[Orchestrator] ✓ Updated controller file: {self.controller_file}")
            
            # 保存优化后的代码到历史记录
            optimized_path = self.history_dir / f"iter_{iteration}_optimized.py"
            optimized_path.write_text(new_code, encoding='utf-8')
            print(f"[Orchestrator] ✓ Saved optimized code: {optimized_path}")
            
            return True
        except Exception as e:
            print(f"[Orchestrator] ✗ Error updating controller: {e}")
            return False
    
    def run_iteration(
        self,
        iteration: int,
        log_files: List[Path]
    ) -> Tuple[bool, Optional[str]]:
        """
        运行单次迭代
        
        Args:
            iteration: 迭代次数
            log_files: 已收集的日志文件列表（会被更新）
        
        Returns:
            (是否成功, 优化后的代码（如果失败且有优化）)
        """
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}/{self.max_iterations}")
        print(f"{'='*80}")
        
        # 1. 备份当前controller
        self.backup_controller(iteration)
        
        # 2. 执行仿真并监控
        print(f"\n[Orchestrator] Step 2: Executing simulation...")
        success, log_file = self.execute_simulation(iteration)
        log_files.append(log_file)
        print(f"[Orchestrator] Step 2 completed: success={success}, log_file={log_file}")
        
        # 3. 判断是否成功
        has_any_failure = not success
        
        # 调试信息
        print(f"\n[Orchestrator] Step 3: Failure Detection")
        print(f"  success (from simulation): {success}")
        print(f"  → has_any_failure: {has_any_failure}")
        
        # 如果成功，返回
        if not has_any_failure:
            print(f"[Orchestrator] Step 3: No failures detected, task SUCCESS")
            return True, None
        
        # 4. 失败处理：调用优化器或结束
        print(f"\n[Orchestrator] Step 4: Failure detected, checking iteration limit...")
        print(f"  Current iteration: {iteration}")
        print(f"  Max iterations: {self.max_iterations}")
        
        if iteration < self.max_iterations:
            print(f"\n{'='*80}")
            print(f"[Orchestrator] ✗ Task FAILED in iteration {iteration}")
            print(f"{'='*80}")
            print(f"[Orchestrator] Failure reason:")
            print(f"  - Simulation returned failure (success={success})")
            print(f"\n[Orchestrator] → Calling optimizer to improve code...")
            print(f"[Orchestrator] Remaining iterations: {self.max_iterations - iteration}")
            
            try:
                print(f"[Orchestrator] Step 4.1: Calling optimizer...")
                optimized_code = self.call_optimizer(iteration, log_files)
                print(f"[Orchestrator] Step 4.1 completed: optimized_code={'exists' if optimized_code else 'None'}")
            except Exception as e:
                print(f"[Orchestrator] ✗ Exception calling optimizer: {e}")
                import traceback
                traceback.print_exc()
                optimized_code = None
            
            return False, optimized_code
        
        # 到达最大迭代次数
        print(f"\n[Orchestrator] Reached maximum iterations ({self.max_iterations})")
        return False, None
        
        return False, None
    
    def save_result(self, success: bool, iterations: int) -> Path:
        """
        保存最终结果
        
        Args:
            success: 是否成功
            iterations: 完成的迭代次数
        
        Returns:
            结果文件路径
        """
        final_result = {
            "success": success,
            "iterations": iterations,
            "final_controller": str(self.controller_file),
            "log_directory": str(self.run_dir)
        }
        result_file = self.run_dir / "final_result.json"
        result_file.write_text(json.dumps(final_result, indent=2), encoding='utf-8')
        
        # 保存最终代码副本
        shutil.copy2(self.controller_file, self.run_dir / "final_controller.py")
        
        return result_file
    
    def merge_all_logs(self, log_files: List[Path]) -> Optional[Path]:
        """
        合并所有迭代的日志文件到一个文件中
        
        Args:
            log_files: 所有迭代的日志文件列表
        
        Returns:
            合并后的日志文件路径，如果失败则返回None
        """
        try:
            # 创建日志目录
            logs_dir = self.project_root / "agent" / "scene" / "optimization" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名：使用controller名称和时间戳
            controller_name = self.controller_file.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merged_log_filename = f"{controller_name}_all_iterations_{timestamp}.txt"
            merged_log_path = logs_dir / merged_log_filename
            
            # 合并所有日志文件
            with open(merged_log_path, 'w', encoding='utf-8') as merged_file:
                merged_file.write("=" * 80 + "\n")
                merged_file.write(f"合并的迭代日志 - {controller_name}\n")
                merged_file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                merged_file.write(f"迭代次数: {len(log_files)}\n")
                merged_file.write("=" * 80 + "\n\n")
                
                for i, log_file in enumerate(log_files, 1):
                    if log_file.exists():
                        merged_file.write("\n" + "=" * 80 + "\n")
                        merged_file.write(f"迭代 {i} - {log_file.name}\n")
                        merged_file.write(f"文件路径: {log_file}\n")
                        merged_file.write("=" * 80 + "\n\n")
                        
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                merged_file.write(content)
                                if not content.endswith('\n'):
                                    merged_file.write('\n')
                        except Exception as e:
                            merged_file.write(f"[错误] 无法读取日志文件: {e}\n")
                        
                        merged_file.write("\n")
                    else:
                        merged_file.write(f"\n[警告] 日志文件不存在: {log_file}\n\n")
            
            print(f"[Orchestrator] ✓ 所有迭代日志已合并到: {merged_log_path}")
            return merged_log_path
            
        except Exception as e:
            print(f"[Orchestrator] ✗ 合并日志文件时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self) -> bool:
        """
        运行完整的优化迭代流程
        
        Returns:
            是否成功完成
        """
        print("\n" + "="*80)
        print("Agent Orchestrator - Starting Optimization Iterations")
        print("="*80)
        
        log_files: List[Path] = []
        
        for iteration in range(1, self.max_iterations + 1):
            # 运行单次迭代
            success, optimized_code = self.run_iteration(iteration, log_files)
            
            # 如果成功，保存结果并返回
            if success:
                print(f"\n{'='*80}")
                print(f"✓✓✓ SUCCESS! Task completed successfully in iteration {iteration}")
                print(f"{'='*80}")
                
                self.save_result(True, iteration)
                
                print(f"\nFinal controller: {self.controller_file}")
                print(f"Logs saved to: {self.run_dir}")
                return True
            
            # 如果失败但有优化代码，更新controller并继续下一轮
            if optimized_code:
                if self.update_controller(optimized_code, iteration):
                    print(f"[Orchestrator] ✓ Controller updated, proceeding to next iteration")
                    continue
                else:
                    print(f"[Orchestrator] ✗ Failed to update controller, stopping")
                    break
            
            # 失败且没有优化代码，提前停止（除非刚好是最后一轮）
            if iteration < self.max_iterations:
                print(f"[Orchestrator] ✗ Optimization failed with no new code, stopping")
                break
        
        # 所有迭代都失败
        print(f"\n{'='*80}")
        print(f"✗✗✗ FAILED: Could not complete task within {self.max_iterations} iterations")
        print(f"{'='*80}")
        
        self.save_result(False, self.max_iterations)
        
        # 合并所有迭代的日志文件
        print(f"\n[Orchestrator] Merging all iteration logs...")
        merged_log_path = self.merge_all_logs(log_files)
        if merged_log_path:
            print(f"[Orchestrator] Merged log saved to: {merged_log_path}")
        
        print(f"\nLogs saved to: {self.run_dir}")
        return False


def run_optimization_iterations(
    controller_file: str,
    config_name: str,
    max_iterations: int = 5,
    python_executable: Optional[str] = None,
    project_root: Optional[str] = None,
    run_dir: Optional[str] = None,
    config_dir: Optional[str] = None,
    headless: bool = False,
) -> bool:
    """
    便捷函数：运行优化迭代流程
    
    Args:
        controller_file: controller文件路径
        config_name: 配置文件名（不含.yaml）
        max_iterations: 最大迭代次数
        python_executable: Python解释器路径
        project_root: 项目根目录
        run_dir: 运行输出目录
    
    Returns:
        是否成功完成
    """
    orchestrator = AgentOrchestrator(
        controller_file=controller_file,
        config_name=config_name,
        max_iterations=max_iterations,
        python_executable=python_executable,
        project_root=project_root,
        run_dir=run_dir,
        config_dir=config_dir,
        headless=headless,
    )
    
    return orchestrator.run()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Agent Orchestrator - 总控制器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python agent/action/action_orchestrator.py \\
      --controller controllers/protocol1_controller.py \\
      --config-name protocol1
  
  # 指定最大迭代次数
  python agent/action/action_orchestrator.py \\
      --controller controllers/protocol1_controller.py \\
      --config-name protocol1 \\
      --max-iterations 10
  
  # 指定Python解释器（Isaac Sim环境）
  python agent/action/action_orchestrator.py \\
      --controller controllers/protocol1_controller.py \\
      --config-name protocol1 \\
      --python /path/to/isaac-sim/python
        """
    )
    
    parser.add_argument(
        "--controller",
        type=str,
        required=True,
        help="Controller文件路径（如 controllers/protocol1_controller.py）"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="配置文件名（不含.yaml，如 level1_pick）"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="最大迭代次数（默认：5）"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="agent/logs",
        help="日志目录（默认：agent/logs，已废弃，仅保留兼容性）"
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python解释器路径（默认：使用当前环境的python）"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="项目根目录（默认：自动检测）"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="运行输出目录（默认：outputs/optimization/{date}/{time}_{controller_name}）"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="配置目录（默认：项目根目录/config）",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否使用无界面仿真",
    )
    
    args = parser.parse_args()
    
    # 创建并运行orchestrator
    orchestrator = AgentOrchestrator(
        controller_file=args.controller,
        config_name=args.config_name,
        max_iterations=args.max_iterations,
        log_dir=args.log_dir,
        python_executable=args.python,
        project_root=args.project_root,
        run_dir=args.run_dir,
        config_dir=args.config_dir,
        headless=args.headless,
    )
    
    success = orchestrator.run()
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
