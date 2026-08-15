"""
场景生成模块

提供场景生成功能，从器材列表生成完整的 USD 场景文件
"""

from .scene_generator import SceneGenerator, generate_scene

__all__ = ['SceneGenerator', 'generate_scene']

