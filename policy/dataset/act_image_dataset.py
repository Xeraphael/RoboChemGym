from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch

from policy.common.normalize_util import get_image_range_normalizer
from policy.dataset.base_dataset import BaseImageDataset
from policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class ACTImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta,
        dataset_path: str,
        seed: int = 42,
        horizon: int | None = None,
        n_obs_steps: int | None = 1,
        val_ratio: float = 0.0,
        in_memory: bool = False,
        split: str = "train",
        validation_split: str = "validation",
        sample_stride: int = 1,
    ):
        del seed
        if horizon is None or horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_obs_steps != 1:
            raise ValueError("ACT dataset schema v1 supports n_obs_steps=1")
        if val_ratio not in (0, 0.0):
            raise ValueError("val_ratio is replaced by persisted splits.json")
        self.dataset_path = Path(dataset_path)
        self.shape_meta = shape_meta
        self.horizon = int(horizon)
        self.in_memory = bool(in_memory)
        self.split = split
        self.validation_split = validation_split
        if isinstance(sample_stride, bool) or int(sample_stride) <= 0:
            raise ValueError("sample_stride must be a positive integer")
        self.sample_stride = int(sample_stride)
        self.schema = self._read_json("schema.json")
        self.manifest = self._read_json("manifest.json")
        self.splits = self._read_json("splits.json")
        if self.schema.get("schema_version") != self.manifest.get("schema_version"):
            raise ValueError("dataset schema and manifest versions differ")
        if split not in self.splits["splits"]:
            raise ValueError(f"unknown persisted split: {split}")
        self.camera_names = list(self.schema["camera_order"])
        requested_ids = set(self.splits["splits"][split])
        entries = {
            entry["episode_id"]: entry
            for entry in self.manifest["episodes"]
            if entry.get("status") == "completed" and entry.get("success") is True
        }
        self.entries = [entries[value] for value in sorted(requested_ids & entries.keys())]
        if not self.entries:
            raise ValueError(f"dataset split has no successful episodes: {split}")
        self.episode_data = {}
        self.sequences = []
        for entry in self.entries:
            episode_id = entry["episode_id"]
            path = self.dataset_path / entry["hdf5_path"]
            if not path.is_file():
                raise ValueError(f"manifest HDF5 does not exist: {path}")
            with h5py.File(path, "r") as episode:
                self._validate_episode(episode, entry)
                if self.in_memory:
                    self.episode_data[episode_id] = {
                        key: episode[key][:]
                        for key in (*self.camera_names, "agent_pose", "actions")
                    }
            self.sequences.extend(
                (episode_id, index)
                for index in range(0, entry["length"], self.sample_stride)
            )
        self.entry_by_id = {entry["episode_id"]: entry for entry in self.entries}

    def _read_json(self, name: str):
        path = self.dataset_path / name
        if not path.is_file():
            raise ValueError(f"dataset artifact does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _validate_episode(self, episode, entry):
        required = (*self.camera_names, "agent_pose", "actions", "timestamps")
        missing = [name for name in required if name not in episode]
        if missing:
            raise ValueError("episode fields are missing: " + ", ".join(missing))
        if any(len(episode[name]) != entry["length"] for name in required):
            raise ValueError(f"episode lengths do not match manifest: {entry['episode_id']}")
        if episode["agent_pose"].shape[1:] != (8,) or episode["actions"].shape[1:] != (8,):
            raise ValueError("episode pose/action shape is incompatible")

    def __len__(self) -> int:
        return len(self.sequences)

    def _load_fields(self, episode_id: str, start: int, stop: int | None = None):
        if self.in_memory:
            episode = self.episode_data[episode_id]
            return {key: value[start:stop] for key, value in episode.items()}
        path = self.dataset_path / self.entry_by_id[episode_id]["hdf5_path"]
        with h5py.File(path, "r") as episode:
            return {
                key: episode[key][start:stop]
                for key in (*self.camera_names, "agent_pose", "actions")
            }

    def _load_sample(self, episode_id: str, start: int):
        stop = start + self.horizon
        if self.in_memory:
            episode = self.episode_data[episode_id]
            return {
                "images": {name: episode[name][start] for name in self.camera_names},
                "agent_pose": episode["agent_pose"][start],
                "actions": episode["actions"][start:stop],
            }
        path = self.dataset_path / self.entry_by_id[episode_id]["hdf5_path"]
        with h5py.File(path, "r") as episode:
            return {
                "images": {name: episode[name][start] for name in self.camera_names},
                "agent_pose": episode["agent_pose"][start],
                "actions": episode["actions"][start:stop],
            }

    def get_all_actions(self) -> torch.Tensor:
        values = []
        for entry in self.entries:
            if self.in_memory:
                actions = self.episode_data[entry["episode_id"]]["actions"]
            else:
                with h5py.File(self.dataset_path / entry["hdf5_path"], "r") as episode:
                    actions = episode["actions"][:]
            values.append(torch.from_numpy(np.asarray(actions, dtype=np.float32)))
        return torch.cat(values, dim=0)

    def get_normalizer(self, mode="limits", **kwargs):
        del mode, kwargs
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.get_all_actions().numpy()
        )
        poses = []
        for entry in self.entries:
            if self.in_memory:
                value = self.episode_data[entry["episode_id"]]["agent_pose"]
            else:
                with h5py.File(self.dataset_path / entry["hdf5_path"], "r") as episode:
                    value = episode["agent_pose"][:]
            poses.append(np.asarray(value, dtype=np.float32))
        normalizer["agent_pose"] = SingleFieldLinearNormalizer.create_fit(
            np.concatenate(poses, axis=0)
        )
        for camera_name in self.camera_names:
            normalizer[camera_name] = get_image_range_normalizer()
        return normalizer

    def get_validation_dataset(self):
        return type(self)(
            shape_meta=self.shape_meta,
            dataset_path=str(self.dataset_path),
            horizon=self.horizon,
            n_obs_steps=1,
            in_memory=self.in_memory,
            split=self.validation_split,
            validation_split=self.validation_split,
            sample_stride=self.sample_stride,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_id, start = self.sequences[index]
        sample = self._load_sample(episode_id, start)
        actions = np.asarray(sample["actions"], dtype=np.float32)
        action_length = len(actions)
        padded_actions = np.zeros((self.horizon, 8), dtype=np.float32)
        padded_actions[:action_length] = actions
        is_pad = np.ones(self.horizon, dtype=bool)
        is_pad[:action_length] = False
        obs = {
            camera_name: torch.from_numpy(
                np.asarray(sample["images"][camera_name], dtype=np.uint8)
            ).float()
            / 255.0
            for camera_name in self.camera_names
        }
        obs["agent_pose"] = torch.from_numpy(
            np.asarray(sample["agent_pose"], dtype=np.float32)
        )
        return {
            "obs": obs,
            "action": torch.from_numpy(padded_actions),
            "is_pad": torch.from_numpy(is_pad),
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "obs": {
                key: torch.stack([item["obs"][key] for item in batch])
                for key in batch[0]["obs"]
            },
            "action": torch.stack([item["action"] for item in batch]),
            "is_pad": torch.stack([item["is_pad"] for item in batch]),
        }
