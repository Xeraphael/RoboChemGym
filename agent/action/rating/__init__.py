"""
轨迹评估模块：记录和分析控制器执行时的末端夹爪轨迹

功能：
1. 记录末端夹爪位置轨迹
2. 检测碰撞（位置快速改变）
3. 检测卡顿（位置长期不变）
4. 计算轨迹平滑系数
"""

from .trajectory_recorder import TrajectoryRecorder
from .trajectory_analyzer import TrajectoryAnalyzer
from .trajectory_evaluator import TrajectoryEvaluator

__all__ = ['TrajectoryRecorder', 'TrajectoryAnalyzer', 'TrajectoryEvaluator']

