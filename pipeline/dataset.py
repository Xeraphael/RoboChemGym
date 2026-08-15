from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DATASET_SCHEMA_VERSION = "1.0"
SPLIT_VERSION = "1.0"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_dataset_schema(camera_configs: Sequence[Any]) -> dict[str, Any]:
    cameras = []
    features: dict[str, Any] = {}
    camera_order = []
    for raw in camera_configs:
        config = dict(raw)
        width, height = map(int, config["resolution"])
        image_types = str(config["image_type"]).split("+")
        for image_type in image_types:
            key = f"{config['name']}_{image_type}"
            channels = 3 if image_type == "rgb" else 1
            features[key] = {
                "shape": [channels, height, width],
                "dtype": "uint8",
                "units": "pixel",
                "encoding": image_type,
            }
            if image_type == "rgb":
                camera_order.append(key)
        cameras.append(
            {
                "name": str(config["name"]),
                "prim_path": str(config["prim_path"]),
                "image_type": str(config["image_type"]),
                "resolution": [width, height],
                "focal_length": config.get("focal_length"),
            }
        )
    features.update(
        {
            "agent_pose": {"shape": [8], "dtype": "float32", "units": "radians"},
            "actions": {"shape": [8], "dtype": "float32", "units": "radians"},
            "timestamps": {"shape": [], "dtype": "float64", "units": "seconds"},
        }
    )
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "features": features,
        "camera_order": camera_order,
        "cameras": cameras,
        "action_convention": "absolute_joint_position",
        "gripper_convention": "first_finger_position_second_mirrored",
        "terminal_status": "successful_plan_execution_only",
    }


def ensure_schema(dataset_dir: Path, camera_configs: Sequence[Any]) -> dict[str, Any]:
    path = dataset_dir / "schema.json"
    expected = build_dataset_schema(camera_configs)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("configured observation schema differs from dataset schema")
        return existing
    atomic_json(path, expected)
    return expected


def ensure_splits(
    dataset_dir: Path,
    episode_ids: Sequence[str],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, Any]:
    path = dataset_dir / "splits.json"
    episode_ids = list(episode_ids)
    if path.is_file():
        splits = json.loads(path.read_text(encoding="utf-8"))
        if splits.get("split_version") != SPLIT_VERSION:
            raise ValueError("dataset split version is incompatible")
        if splits.get("seed") != seed or splits.get("episode_ids") != episode_ids:
            raise ValueError("persisted dataset split inputs differ from configuration")
        return splits
    if len(episode_ids) < 3:
        splits = {
            "split_version": SPLIT_VERSION,
            "seed": seed,
            "ratios": list(ratios),
            "episode_ids": episode_ids,
            "splits": {"train": episode_ids, "validation": [], "test": []},
        }
        atomic_json(path, splits)
        return splits
    if not np.isclose(sum(ratios), 1.0) or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must be positive and sum to one")
    shuffled = list(episode_ids)
    np.random.default_rng(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * ratios[1]))
    test_count = max(1, round(len(shuffled) * ratios[2]))
    train_count = len(shuffled) - validation_count - test_count
    if train_count <= 0:
        raise ValueError("split ratios leave no training episodes")
    membership = {
        "train": sorted(shuffled[:train_count]),
        "validation": sorted(shuffled[train_count : train_count + validation_count]),
        "test": sorted(shuffled[train_count + validation_count :]),
    }
    splits = {
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "ratios": list(ratios),
        "episode_ids": episode_ids,
        "splits": membership,
    }
    atomic_json(path, splits)
    return splits


def split_for_episode(splits: dict[str, Any], episode_id: str) -> str:
    matches = [name for name, ids in splits["splits"].items() if episode_id in ids]
    if len(matches) != 1:
        raise ValueError(f"episode has invalid split membership: {episode_id}")
    return matches[0]


def build_dataset_identity(
    manifest: dict[str, Any], splits: dict[str, Any]
) -> dict[str, Any]:
    for field in ("config_id", "protocol_id", "schema_version"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"dataset manifest has no valid {field}")
    episodes = []
    for entry in manifest.get("episodes", []):
        if entry.get("status") != "completed" or entry.get("success") is not True:
            continue
        episode_id = str(entry["episode_id"])
        episodes.append(
            {
                "episode_id": episode_id,
                "length": int(entry["length"]),
                "split": split_for_episode(splits, episode_id),
            }
        )
    if not episodes:
        raise ValueError("dataset has no successful completed episodes")
    return {
        "config_id": manifest.get("config_id"),
        "protocol_id": manifest.get("protocol_id"),
        "schema_version": manifest.get("schema_version"),
        "split_version": splits.get("split_version"),
        "split_seed": splits.get("seed"),
        "split_membership": {
            name: list(values) for name, values in sorted(splits["splits"].items())
        },
        "episodes": sorted(episodes, key=lambda value: value["episode_id"]),
    }
