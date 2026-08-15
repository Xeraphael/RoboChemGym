"""
轨迹记录器：记录末端夹爪位置轨迹
"""

import numpy as np
from typing import List, Optional
from pathlib import Path
import json
from datetime import datetime


class TrajectoryRecorder:
    """记录末端夹爪位置轨迹"""
    
    def __init__(self, frame_interval: int = 1):
        """
        初始化轨迹记录器
        
        Args:
            frame_interval: 记录间隔（每N帧记录一次，1表示每帧都记录）
        """
        self.frame_interval = frame_interval
        self.positions: List[np.ndarray] = []
        self.timestamps: List[float] = []
        self.frame_count = 0
    
    def reset(self):
        """重置记录器"""
        self.positions.clear()
        self.timestamps.clear()
        self.frame_count = 0
    
    def record(self, gripper_position: np.ndarray, timestamp: Optional[float] = None):
        """
        记录一次夹爪位置
        
        Args:
            gripper_position: 夹爪位置 (3D numpy array)
            timestamp: 时间戳（可选，如果不提供则使用帧计数）
        """
        if self.frame_count % self.frame_interval == 0:
            self.positions.append(gripper_position.copy())
            if timestamp is not None:
                self.timestamps.append(timestamp)
            else:
                self.timestamps.append(float(self.frame_count))
        self.frame_count += 1
    
    def get_trajectory(self) -> tuple:
        """
        获取记录的轨迹
        
        Returns:
            (positions, timestamps): 位置列表和时间戳列表
        """
        return self.positions.copy(), self.timestamps.copy()
    
    def get_trajectory_array(self) -> np.ndarray:
        """
        获取轨迹的numpy数组形式
        
        Returns:
            Nx3 numpy array，每行是一个3D位置
        """
        if not self.positions:
            return np.array([]).reshape(0, 3)
        return np.array(self.positions)
    
    def save(self, filepath: str):
        """
        保存轨迹到文件
        
        Args:
            filepath: 保存路径（JSON格式）
        """
        data = {
            'positions': [pos.tolist() for pos in self.positions],
            'timestamps': self.timestamps,
            'frame_interval': self.frame_interval,
            'total_frames': self.frame_count,
            'recorded_frames': len(self.positions)
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """
        从文件加载轨迹
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.positions = [np.array(pos) for pos in data['positions']]
        self.timestamps = data['timestamps']
        self.frame_interval = data.get('frame_interval', 1)
        self.frame_count = data.get('total_frames', len(self.positions))

