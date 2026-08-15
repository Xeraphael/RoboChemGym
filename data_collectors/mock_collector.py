from __future__ import annotations

from typing import List

import numpy as np


class MockCollector:
    """No-op collector retaining the production collector's public lifecycle."""

    def __init__(
        self,
        camera_configs: List[dict],
        save_dir="output",
        max_episodes=10,
        max_workers=4,
        compression=None,
        **kwargs,
    ):
        del camera_configs, save_dir, max_workers, compression, kwargs
        self.max_episodes = int(max_episodes)
        self.episode_count = 0

    def cache_step(self, camera_images=None, joint_angles=None, language_instruction=None):
        del camera_images, joint_angles, language_instruction

    def record_step(self, **kwargs):
        del kwargs

    def start_episode(self, metadata=None):
        del metadata
        return f"mock-{self.episode_count:04d}"

    def write_cached_data(self, final_joint_positions=None):
        del final_joint_positions
        self.episode_count += 1

    def commit_episode(self, report):
        del report
        self.episode_count += 1

    def fail_episode(self, report):
        del report
        self.episode_count += 1

    def clear_cache(self, *args, **kwargs):
        del args, kwargs

    def close(self):
        return None
