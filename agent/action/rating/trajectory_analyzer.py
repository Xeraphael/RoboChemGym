"""
轨迹分析器：分析末端夹爪轨迹，检测碰撞、卡顿和平滑度
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryAnalysisResult:
    """轨迹分析结果"""
    smoothness_score: float  # 平滑系数（0-1，越高越平滑）
    collision_count: int  # 碰撞次数
    collision_frames: List[int]  # 发生碰撞的帧索引
    stall_count: int  # 卡顿次数
    stall_frames: List[int]  # 发生卡顿的帧索引
    max_velocity: float  # 最大速度
    mean_velocity: float  # 平均速度
    total_distance: float  # 总移动距离
    analysis_summary: str  # 分析摘要


class TrajectoryAnalyzer:
    """分析末端夹爪轨迹"""
    
    def __init__(
        self,
        collision_threshold: float = 0.05,
        stall_threshold: float = 0.001,
        stall_duration: int = 30
    ):
        """
        初始化轨迹分析器
        
        Args:
            collision_threshold: 碰撞检测阈值（米），相邻帧位置变化超过此值视为碰撞
            stall_threshold: 卡顿检测阈值（米），位置变化小于此值视为静止
            stall_duration: 卡顿持续时间（帧数），连续静止超过此帧数视为卡顿
        """
        self.collision_threshold = collision_threshold
        self.stall_threshold = stall_threshold
        self.stall_duration = stall_duration
    
    def analyze(self, positions: List[np.ndarray]) -> TrajectoryAnalysisResult:
        """
        分析轨迹
        
        Args:
            positions: 位置列表，每个元素是3D numpy array
        
        Returns:
            TrajectoryAnalysisResult: 分析结果
        """
        if len(positions) < 2:
            return TrajectoryAnalysisResult(
                smoothness_score=1.0,
                collision_count=0,
                collision_frames=[],
                stall_count=0,
                stall_frames=[],
                max_velocity=0.0,
                mean_velocity=0.0,
                total_distance=0.0,
                analysis_summary="轨迹数据不足，无法分析"
            )
        
        positions_array = np.array(positions)
        
        # 计算速度（相邻帧位置差）
        velocities = np.diff(positions_array, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # 检测碰撞（速度突然增大）
        collision_frames = []
        for i, speed in enumerate(speeds):
            if speed > self.collision_threshold:
                collision_frames.append(i)
        
        # 检测卡顿（速度长期很小）
        stall_frames = []
        stall_start = None
        for i, speed in enumerate(speeds):
            if speed < self.stall_threshold:
                if stall_start is None:
                    stall_start = i
            else:
                if stall_start is not None:
                    stall_duration = i - stall_start
                    if stall_duration >= self.stall_duration:
                        stall_frames.extend(range(stall_start, i))
                    stall_start = None
        
        # 处理轨迹末尾的卡顿
        if stall_start is not None:
            stall_duration = len(speeds) - stall_start
            if stall_duration >= self.stall_duration:
                stall_frames.extend(range(stall_start, len(speeds)))
        
        # 计算平滑系数（基于加速度的变化）
        if len(speeds) > 1:
            accelerations = np.diff(speeds)
            acceleration_variance = np.var(accelerations)
            # 归一化到0-1，方差越小越平滑
            max_acceleration_variance = np.max(np.abs(accelerations)) ** 2 if len(accelerations) > 0 else 1.0
            if max_acceleration_variance > 0:
                smoothness_score = max(0.0, 1.0 - acceleration_variance / max_acceleration_variance)
            else:
                smoothness_score = 1.0
        else:
            smoothness_score = 1.0
        
        # 计算统计信息
        max_velocity = float(np.max(speeds)) if len(speeds) > 0 else 0.0
        mean_velocity = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
        total_distance = float(np.sum(speeds))
        
        # 生成摘要
        collision_count = len(set(collision_frames))
        stall_count = len(set(stall_frames))
        
        summary_parts = []
        summary_parts.append(f"轨迹总长度: {len(positions)} 帧")
        summary_parts.append(f"总移动距离: {total_distance:.3f} m")
        summary_parts.append(f"平均速度: {mean_velocity:.4f} m/frame")
        summary_parts.append(f"最大速度: {max_velocity:.4f} m/frame")
        summary_parts.append(f"平滑系数: {smoothness_score:.3f}")
        
        if collision_count > 0:
            summary_parts.append(f"⚠️ 检测到 {collision_count} 次碰撞（阈值: {self.collision_threshold}m）")
        else:
            summary_parts.append("✓ 未检测到碰撞")
        
        if stall_count > 0:
            summary_parts.append(f"⚠️ 检测到 {stall_count} 帧卡顿（阈值: {self.stall_threshold}m, 持续时间: {self.stall_duration}帧）")
        else:
            summary_parts.append("✓ 未检测到卡顿")
        
        analysis_summary = "\n".join(summary_parts)
        
        return TrajectoryAnalysisResult(
            smoothness_score=smoothness_score,
            collision_count=collision_count,
            collision_frames=collision_frames,
            stall_count=stall_count,
            stall_frames=stall_frames,
            max_velocity=max_velocity,
            mean_velocity=mean_velocity,
            total_distance=total_distance,
            analysis_summary=analysis_summary
        )

