from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

import h5py
import numpy as np

from pipeline.dataset import atomic_json, ensure_schema, ensure_splits, split_for_episode

SCHEMA_VERSION = "1.0"


def resolve_action_target(action: Any, joint_positions: np.ndarray) -> np.ndarray:
    current = np.asarray(joint_positions, dtype=np.float32)
    if current.shape != (9,):
        raise ValueError(f"expected 9 Franka joints, got {current.shape}")
    values = getattr(action, "joint_positions", action)
    if values is None:
        values = [None] * 9
    values = list(values)
    if len(values) not in (7, 9):
        raise ValueError(f"expected 7 or 9 action targets, got {len(values)}")
    resolved = current.copy()
    for index, value in enumerate(values):
        if value is not None:
            resolved[index] = float(value)
    return np.concatenate((resolved[:7], resolved[7:8])).astype(np.float32)


class DataCollector:
    def __init__(
        self,
        camera_configs: List[dict],
        save_dir="output",
        max_episodes=10,
        max_workers=4,
        compression=None,
        protocol_id: str = "unknown",
        config_id: str = "unknown",
        split_seed: int = 20260813,
    ):
        del max_workers
        self.save_dir = str(save_dir)
        self.max_episodes = int(max_episodes)
        self.compression = compression
        self.protocol_id = protocol_id
        self.config_id = config_id
        self.session_dir = Path(save_dir) / "dataset"
        self.episodes_dir = self.session_dir / "episodes"
        self.reports_dir = self.session_dir / "reports"
        self.manifest_path = self.session_dir / "manifest.json"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.camera_configs = list(camera_configs)
        self.camera_keys = []
        for config in camera_configs:
            image_types = str(config["image_type"]).split("+")
            self.camera_keys.extend(
                f"{config['name']}_{image_type}" for image_type in image_types
            )
        self.schema = ensure_schema(self.session_dir, self.camera_configs)
        episode_ids = [
            f"{self.protocol_id}-{index:04d}" for index in range(self.max_episodes)
        ]
        self.splits = ensure_splits(
            self.session_dir, episode_ids, seed=int(split_seed)
        )
        self.manifest = self._load_manifest()
        self._recover_committed_episodes()
        self.episode_count = len(self.manifest["episodes"])
        indices = [entry["episode_index"] for entry in self.manifest["episodes"]]
        self._next_episode_index = max(indices, default=-1) + 1
        self._active: dict[str, Any] | None = None
        self.clear_cache()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": self.protocol_id,
                "config_id": self.config_id,
                "episodes": [],
            }
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("dataset manifest schema version is incompatible")
        return manifest

    def _recover_committed_episodes(self) -> None:
        known = {entry["episode_id"] for entry in self.manifest["episodes"]}
        recovered = []
        for episode_path in sorted(self.episodes_dir.glob("episode_*.h5")):
            with h5py.File(episode_path, "r") as h5_file:
                metadata = json.loads(h5_file.attrs["metadata_json"])
                report = json.loads(h5_file.attrs["report_json"])
                episode_id = metadata["episode_id"]
                if episode_id in known:
                    continue
                lengths = {
                    len(h5_file[name])
                    for name in (*self.camera_keys, "agent_pose", "actions", "timestamps")
                }
                if len(lengths) != 1 or not report.get("execution_success"):
                    raise ValueError(f"cannot recover invalid episode: {episode_path}")
                episode_name = episode_path.stem
                report_path = self.reports_dir / f"{episode_name}.json"
                atomic_json(report_path, {**metadata, "execution": report})
                recovered.append(
                    {
                        **metadata,
                        "status": "completed",
                        "success": True,
                        "split": split_for_episode(self.splits, episode_id),
                        "length": lengths.pop(),
                        "hdf5_path": str(episode_path.relative_to(self.session_dir)),
                        "report_path": str(report_path.relative_to(self.session_dir)),
                    }
                )
                known.add(episode_id)
        if recovered:
            self.manifest["episodes"].extend(recovered)
            self.manifest["episodes"].sort(key=lambda item: item["episode_index"])
            atomic_json(self.manifest_path, self.manifest)

    def start_episode(self, metadata: Optional[dict[str, Any]] = None) -> str:
        if self._active is not None:
            raise RuntimeError("an episode is already active")
        index = self._next_episode_index
        if index >= self.max_episodes:
            raise RuntimeError("configured episode limit has been reached")
        episode_id = f"{self.protocol_id}-{index:04d}"
        self._active = {
            "episode_id": episode_id,
            "episode_index": index,
            "metadata": metadata or {},
        }
        self.clear_cache(keep_active=True)
        return episode_id

    def record_step(
        self,
        *,
        camera_images: dict[str, np.ndarray],
        joint_positions: np.ndarray,
        action: Any,
        timestamp: float,
        language_instruction: Optional[str] = None,
    ) -> None:
        if self._active is None:
            self.start_episode()
        missing = sorted(set(self.camera_keys) - set(camera_images))
        if missing:
            raise ValueError("missing configured camera data: " + ", ".join(missing))
        for camera_name in self.camera_keys:
            self.temp_cameras[camera_name].append(
                np.asarray(camera_images[camera_name], dtype=np.uint8)
            )
        joints = np.asarray(joint_positions, dtype=np.float32)
        self.temp_agent_pose.append(
            np.concatenate((joints[:7], joints[7:8])).astype(np.float32)
        )
        self.temp_actions.append(resolve_action_target(action, joints))
        self.temp_timestamps.append(float(timestamp))
        if language_instruction is not None:
            self.temp_language_instruction = language_instruction

    def commit_episode(self, report: dict[str, Any]) -> Path:
        active = self._require_active()
        arrays = self._validated_arrays()
        episode_name = f"episode_{active['episode_index']:04d}"
        final_path = self.episodes_dir / f"{episode_name}.h5"
        partial_path = final_path.with_suffix(".h5.partial")
        if final_path.exists():
            raise FileExistsError(f"completed episode already exists: {final_path}")
        metadata = dict(active["metadata"])
        metadata.update(
            {
                "episode_id": active["episode_id"],
                "episode_index": active["episode_index"],
                "schema_version": SCHEMA_VERSION,
            }
        )
        try:
            with h5py.File(partial_path, "w") as h5_file:
                for camera_name, image_data in arrays["cameras"].items():
                    options = {
                        "data": image_data,
                        "dtype": "uint8",
                        "chunks": (min(64, len(image_data)),) + image_data.shape[1:],
                    }
                    if self.compression == "gzip":
                        options.update(compression="gzip", compression_opts=5)
                    h5_file.create_dataset(camera_name, **options)
                h5_file.create_dataset("agent_pose", data=arrays["agent_pose"])
                h5_file.create_dataset("actions", data=arrays["actions"])
                h5_file.create_dataset("timestamps", data=arrays["timestamps"])
                if self.temp_language_instruction is not None:
                    h5_file.create_dataset(
                        "language_instruction",
                        data=self.temp_language_instruction,
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                h5_file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
                h5_file.attrs["report_json"] = json.dumps(report, sort_keys=True)
                h5_file.flush()
            with partial_path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(partial_path, final_path)
            report_path = self.reports_dir / f"{episode_name}.json"
            atomic_json(report_path, {**metadata, "execution": report})
            self._append_manifest(
                {
                    **metadata,
                    "status": "completed",
                    "success": True,
                    "split": split_for_episode(self.splits, active["episode_id"]),
                    "length": len(arrays["actions"]),
                    "hdf5_path": str(final_path.relative_to(self.session_dir)),
                    "report_path": str(report_path.relative_to(self.session_dir)),
                }
            )
            self._finish_episode()
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
        return final_path

    def fail_episode(self, report: dict[str, Any]) -> Path:
        active = self._require_active()
        episode_name = f"episode_{active['episode_index']:04d}"
        report_path = self.reports_dir / f"{episode_name}.json"
        metadata = {
            **active["metadata"],
            "episode_id": active["episode_id"],
            "episode_index": active["episode_index"],
            "schema_version": SCHEMA_VERSION,
            "failure_code": self._failure_code(report),
        }
        try:
            atomic_json(report_path, {**metadata, "execution": report})
            self._append_manifest(
                {
                    **metadata,
                    "status": "failed",
                    "success": False,
                    "split": split_for_episode(self.splits, active["episode_id"]),
                    "length": len(self.temp_actions),
                    "hdf5_path": None,
                    "report_path": str(report_path.relative_to(self.session_dir)),
                }
            )
            self._finish_episode()
        except Exception:
            raise
        return report_path

    @staticmethod
    def _failure_code(report: dict[str, Any]) -> str:
        if report.get("failure_code"):
            return str(report["failure_code"])
        for step in reversed(report.get("steps", [])):
            verification = step.get("verification", {})
            if not step.get("success", False):
                return str(verification.get("code", "STEP_FAILED"))
        return str(report.get("failed_step") or "EXECUTION_FAILED")

    def _append_manifest(self, entry: dict[str, Any]) -> None:
        if any(
            item["episode_id"] == entry["episode_id"]
            for item in self.manifest["episodes"]
        ):
            raise ValueError(f"episode already exists in manifest: {entry['episode_id']}")
        updated = {**self.manifest, "episodes": [*self.manifest["episodes"], entry]}
        atomic_json(self.manifest_path, updated)
        self.manifest = updated

    def _validated_arrays(self) -> dict[str, Any]:
        if not self.temp_actions:
            raise ValueError("cannot commit an episode without applied actions")
        cameras = {
            name: np.stack(images).astype(np.uint8, copy=False)
            for name, images in self.temp_cameras.items()
        }
        agent_pose = np.stack(self.temp_agent_pose).astype(np.float32, copy=False)
        actions = np.stack(self.temp_actions).astype(np.float32, copy=False)
        timestamps = np.asarray(self.temp_timestamps, dtype=np.float64)
        lengths = {len(agent_pose), len(actions), len(timestamps)} | {
            len(images) for images in cameras.values()
        }
        if len(lengths) != 1:
            raise ValueError(f"episode feature lengths differ: {sorted(lengths)}")
        if agent_pose.shape[1:] != (8,) or actions.shape[1:] != (8,):
            raise ValueError("agent_pose and actions must have shape [T,8]")
        if any(images.ndim != 4 or images.shape[1] not in (1, 3) for images in cameras.values()):
            raise ValueError("camera arrays must have shape [T,C,H,W]")
        if not np.all(np.isfinite(agent_pose)) or not np.all(np.isfinite(actions)):
            raise ValueError("pose/action arrays must be finite")
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
            raise ValueError("timestamps must be finite and non-decreasing")
        return {
            "cameras": cameras,
            "agent_pose": agent_pose,
            "actions": actions,
            "timestamps": timestamps,
        }

    def _require_active(self) -> dict[str, Any]:
        if self._active is None:
            raise RuntimeError("no episode is active")
        return self._active

    def _finish_episode(self) -> None:
        self.episode_count += 1
        self._next_episode_index += 1
        self._active = None
        self.clear_cache()

    # Compatibility for legacy controllers. New code must use record_step.
    def cache_step(
        self,
        camera_images: dict,
        joint_angles: np.ndarray,
        language_instruction: Optional[str] = None,
    ) -> None:
        if self._active is None:
            self.start_episode()
        for camera_name, image in camera_images.items():
            self.temp_cameras[camera_name].append(np.asarray(image))
        self.temp_agent_pose.append(np.asarray(joint_angles))
        if language_instruction is not None:
            self.temp_language_instruction = language_instruction

    def write_cached_data(self, final_joint_positions=None):
        if not self.temp_actions:
            poses = [np.asarray(value) for value in self.temp_agent_pose]
            self.temp_actions = poses[1:] + [np.asarray(final_joint_positions)]
            self.temp_timestamps = list(range(len(poses)))
        return self.commit_episode({"execution_success": True, "steps": []})

    def clear_cache(self, keep_active: bool = False):
        self.temp_cameras = {name: [] for name in self.camera_keys}
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_timestamps = []
        self.temp_language_instruction = None
        if not keep_active and self._active is not None:
            self._active = None

    def close(self, merge=False):
        if merge:
            raise ValueError("merged HDF5 output is not part of dataset schema v1")
