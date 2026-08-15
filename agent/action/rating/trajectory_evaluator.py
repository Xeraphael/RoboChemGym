"""
轨迹评估器：执行控制器并记录、分析轨迹
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import hydra
from omegaconf import OmegaConf

from .trajectory_recorder import TrajectoryRecorder
from .trajectory_analyzer import TrajectoryAnalyzer, TrajectoryAnalysisResult

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.isaacsim_runtime import prepare_isaacsim_argv


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description='评估控制器轨迹')
    parser.add_argument('--config-name', type=str, required=True, help='配置文件名（不含.yaml）')
    parser.add_argument('--frame-interval', type=int, default=1, help='记录间隔（每N帧记录一次）')
    parser.add_argument('--collision-threshold', type=float, default=0.05, help='碰撞检测阈值（米）')
    parser.add_argument('--stall-threshold', type=float, default=0.001, help='卡顿检测阈值（米）')
    parser.add_argument('--stall-duration', type=int, default=30, help='卡顿持续时间（帧数）')
    parser.add_argument('--max-steps', type=int, default=1000, help='最大步数')
    parser.add_argument('--headless', action='store_true', help='无头模式运行')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'gpu'], help='物理引擎后端')
    parser.add_argument('--save-trajectory', type=str, default=None, help='保存轨迹的文件路径')
    return parser.parse_known_args(argv)


class TrajectoryEvaluator:
    """轨迹评估器：执行控制器并评估轨迹"""
    
    def __init__(
        self,
        config_name: str,
        frame_interval: int = 1,
        collision_threshold: float = 0.05,
        stall_threshold: float = 0.001,
        stall_duration: int = 30,
        headless: bool = True,
        backend: str = 'numpy'
    ):
        """
        初始化轨迹评估器
        
        Args:
            config_name: 配置文件名（不含.yaml）
            frame_interval: 记录间隔（每N帧记录一次）
            collision_threshold: 碰撞检测阈值（米）
            stall_threshold: 卡顿检测阈值（米）
            stall_duration: 卡顿持续时间（帧数）
            headless: 是否无头模式运行
            backend: 物理引擎后端（'numpy' 或 'gpu'）
        """
        self.config_name = config_name
        self.frame_interval = frame_interval
        self.headless = headless
        self.backend = backend
        
        self.recorder = TrajectoryRecorder(frame_interval=frame_interval)
        self.analyzer = TrajectoryAnalyzer(
            collision_threshold=collision_threshold,
            stall_threshold=stall_threshold,
            stall_duration=stall_duration
        )
        
        self.simulation_app = None
        self.world = None
        self.robot = None
        self.task = None
        self.controller = None
    
    def _initialize_simulation(self):
        """初始化仿真环境"""
        prepare_isaacsim_argv()
        try:
            from isaacsim import SimulationApp
        except ImportError:
            if 'ISAACSIM_PATH' not in os.environ:
                raise ImportError("无法导入 isaacsim，请确保已正确安装 Isaac Sim")
            from isaacsim import SimulationApp

        simulation_config = {"headless": self.headless}
        self.simulation_app = SimulationApp(simulation_config)

        import omni.physx
        import omni.usd
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage

        from factories.controller_factory import create_controller
        from factories.robot_factory import create_robot
        from factories.task_factory import create_task
        from utils.object_utils import ObjectUtils
        
        hydra.initialize(config_path="config", job_name=self.config_name)
        cfg = hydra.compose(config_name=self.config_name)
        
        if self.backend == 'gpu':
            self.world = World(stage_units_in_meters=1, device="cpu")
            physx_interface = omni.physx.get_physx_interface()
            physx_interface.overwrite_gpu_setting(1)
        else:
            self.world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
        
        self.robot = create_robot(
            cfg.robot.type,
            position=np.array(cfg.robot.position)
        )
        
        stage = omni.usd.get_context().get_stage()
        add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")
        
        ObjectUtils.get_instance(stage)
        
        self.task = create_task(
            cfg.task_type,
            cfg=cfg,
            world=self.world,
            stage=stage,
            robot=self.robot,
        )
        
        self.controller = create_controller(
            cfg.controller_type,
            cfg=cfg,
            robot=self.robot,
        )
        
        self.task.reset()
        self.controller.reset()
    
    def run(self, max_steps: int = 1000) -> TrajectoryAnalysisResult:
        """
        运行控制器并记录轨迹
        
        Args:
            max_steps: 最大步数
        
        Returns:
            TrajectoryAnalysisResult: 分析结果
        """
        if self.simulation_app is None:
            self._initialize_simulation()
        
        self.recorder.reset()
        step_count = 0
        
        while self.simulation_app.is_running() and step_count < max_steps:
            self.world.step(render=not self.headless)
            
            if self.world.is_stopped():
                self.controller.reset_needed = True
            
            if self.world.is_playing():
                if self.controller.need_reset() or self.task.need_reset():
                    self.controller.reset()
                    self.task.reset()
                    continue
                
                state = self.task.step()
                if state is None:
                    continue
                
                # 记录夹爪位置
                if 'gripper_position' in state:
                    self.recorder.record(state['gripper_position'])
                
                # 执行控制器
                action, done, is_success = self.controller.step(state)
                if action is not None:
                    self.robot.get_articulation_controller().apply_action(action)
                
                if done:
                    self.task.on_task_complete(is_success)
                    break
                
                step_count += 1
        
        # 分析轨迹
        positions, _ = self.recorder.get_trajectory()
        result = self.analyzer.analyze(positions)
        
        return result
    
    def evaluate(
        self,
        max_steps: int = 1000,
        save_trajectory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        评估轨迹（完整流程）
        
        Args:
            max_steps: 最大步数
            save_trajectory: 保存轨迹的文件路径（可选）
        
        Returns:
            评估结果字典
        """
        result = self.run(max_steps=max_steps)
        
        if save_trajectory:
            self.recorder.save(save_trajectory)
        
        return {
            'smoothness_score': result.smoothness_score,
            'collision_count': result.collision_count,
            'collision_frames': result.collision_frames,
            'stall_count': result.stall_count,
            'stall_frames': result.stall_frames,
            'max_velocity': result.max_velocity,
            'mean_velocity': result.mean_velocity,
            'total_distance': result.total_distance,
            'analysis_summary': result.analysis_summary,
            'trajectory_length': len(self.recorder.positions)
        }
    
    def close(self):
        """关闭仿真环境"""
        if self.simulation_app is not None:
            self.simulation_app.close()
            self.simulation_app = None


def evaluate_trajectory(
    config_name: str,
    frame_interval: int = 1,
    collision_threshold: float = 0.05,
    stall_threshold: float = 0.001,
    stall_duration: int = 30,
    max_steps: int = 1000,
    headless: bool = True,
    backend: str = 'numpy',
    save_trajectory: Optional[str] = None
) -> Dict[str, Any]:
    """
    评估控制器轨迹（便捷函数）
    
    Args:
        config_name: 配置文件名（不含.yaml）
        frame_interval: 记录间隔
        collision_threshold: 碰撞检测阈值（米）
        stall_threshold: 卡顿检测阈值（米）
        stall_duration: 卡顿持续时间（帧数）
        max_steps: 最大步数
        headless: 是否无头模式
        backend: 物理引擎后端
        save_trajectory: 保存轨迹的文件路径（可选）
    
    Returns:
        评估结果字典
    """
    evaluator = TrajectoryEvaluator(
        config_name=config_name,
        frame_interval=frame_interval,
        collision_threshold=collision_threshold,
        stall_threshold=stall_threshold,
        stall_duration=stall_duration,
        headless=headless,
        backend=backend
    )
    
    try:
        result = evaluator.evaluate(max_steps=max_steps, save_trajectory=save_trajectory)
        return result
    finally:
        evaluator.close()


if __name__ == '__main__':
    args, kit_args = parse_cli_args()
    prepare_isaacsim_argv(kit_args)
    
    result = evaluate_trajectory(
        config_name=args.config_name,
        frame_interval=args.frame_interval,
        collision_threshold=args.collision_threshold,
        stall_threshold=args.stall_threshold,
        stall_duration=args.stall_duration,
        max_steps=args.max_steps,
        headless=args.headless,
        backend=args.backend,
        save_trajectory=args.save_trajectory
    )
    
    print("\n" + "="*80)
    print("轨迹评估结果")
    print("="*80)
    print(result['analysis_summary'])
    print("="*80)
