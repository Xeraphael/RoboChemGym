from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np


_ATTEMPT_FILE_PATTERN = re.compile(
    r"^attempt_(\d+)(?:\.partial|_(?:success|failed|aborted))\.mp4$"
)


class _VideoWriter(Protocol):
    def isOpened(self) -> bool: ...

    def write(self, frame: np.ndarray) -> Any: ...

    def release(self) -> Any: ...


WriterFactory = Callable[[str, int, float, tuple[int, int]], _VideoWriter]


@dataclass(frozen=True)
class AttemptVideoConfig:
    enabled: bool = False
    every_n_episodes: int = 1
    frame_stride: int = 4
    source_fps: float = 60.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if isinstance(self.frame_stride, bool) or not isinstance(
            self.frame_stride, Integral
        ):
            raise TypeError("frame_stride must be an integer")
        if isinstance(self.every_n_episodes, bool) or not isinstance(
            self.every_n_episodes, Integral
        ):
            raise TypeError("every_n_episodes must be an integer")
        if self.every_n_episodes <= 0:
            raise ValueError("every_n_episodes must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if isinstance(self.source_fps, bool) or not isinstance(self.source_fps, Real):
            raise TypeError("source_fps must be a number")
        if not math.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("source_fps must be positive and finite")

        object.__setattr__(self, "frame_stride", int(self.frame_stride))
        object.__setattr__(self, "every_n_episodes", int(self.every_n_episodes))
        object.__setattr__(self, "source_fps", float(self.source_fps))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "AttemptVideoConfig":
        if value is None:
            return cls(enabled=False)
        if not isinstance(value, Mapping):
            raise TypeError("video configuration must be a mapping or None")
        return cls(**dict(value))

    @property
    def output_fps(self) -> float:
        return self.source_fps / self.frame_stride


def _has_rgb_image_type(image_type: Any) -> bool:
    if not isinstance(image_type, str):
        return False
    return "rgb" in {part.strip().lower() for part in image_type.split("+")}


def build_rgb_mosaic(
    camera_data: Mapping[str, np.ndarray],
    camera_configs: Sequence[Mapping[str, Any]],
    *,
    draw_labels: bool = True,
) -> np.ndarray | None:
    frames: list[tuple[str, np.ndarray]] = []
    for camera_config in camera_configs:
        if not _has_rgb_image_type(camera_config.get("image_type")):
            continue

        camera_name = camera_config.get("name")
        if not isinstance(camera_name, str):
            continue
        frame = camera_data.get(f"{camera_name}_rgb")
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[0] != 3
            or frame.shape[1] <= 0
            or frame.shape[2] <= 0
        ):
            continue

        bgr_frame = np.ascontiguousarray(frame.transpose(1, 2, 0)[..., ::-1])
        frames.append((camera_name, bgr_frame))

    if not frames:
        return None

    target_height = min(frame.shape[0] for _, frame in frames)
    resized_frames: list[np.ndarray] = []
    for camera_name, frame in frames:
        if frame.shape[0] != target_height:
            target_width = max(
                1,
                int(round(frame.shape[1] * target_height / frame.shape[0])),
            )
            frame = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )

        if draw_labels:
            baseline = max(1, min(16, target_height - 1))
            cv2.putText(
                frame,
                camera_name,
                (4, baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        resized_frames.append(frame)

    return np.concatenate(resized_frames, axis=1)


def _opencv_writer_factory(
    path: str,
    fourcc: int,
    fps: float,
    frame_size: tuple[int, int],
) -> _VideoWriter:
    return cv2.VideoWriter(path, fourcc, fps, frame_size)


class AttemptVideoRecorder:
    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        camera_configs: Sequence[Mapping[str, Any]],
        config: AttemptVideoConfig,
        writer_factory: WriterFactory | None = None,
    ) -> None:
        if not isinstance(config, AttemptVideoConfig):
            raise TypeError("config must be an AttemptVideoConfig")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_configs = tuple(camera_configs)
        self.config = config
        self.writer_factory = (
            writer_factory if writer_factory is not None else _opencv_writer_factory
        )

        self._active = False
        self._attempt_index: int | None = None
        self._partial_path: Path | None = None
        self._writer: _VideoWriter | None = None
        self._frame_count = 0
        self._source_frame_count = 0
        self._record_current_attempt = True
        self._next_attempt_index = self._scan_next_attempt_index()

    def _scan_next_attempt_index(self) -> int:
        indices = []
        for path in self.output_dir.glob("attempt_*.mp4"):
            match = _ATTEMPT_FILE_PATTERN.fullmatch(path.name)
            if match is not None:
                indices.append(int(match.group(1)))
        return max(indices, default=-1) + 1

    def start_attempt(self) -> int:
        if self._active:
            raise RuntimeError("an attempt is already active")

        attempt_index = max(
            self._next_attempt_index,
            self._scan_next_attempt_index(),
        )
        self._next_attempt_index = attempt_index + 1
        self._active = True
        self._attempt_index = attempt_index
        self._partial_path = self.output_dir / f"attempt_{attempt_index:04d}.partial.mp4"
        self._writer = None
        self._frame_count = 0
        self._source_frame_count = 0
        self._record_current_attempt = (
            attempt_index % self.config.every_n_episodes == 0
        )
        return attempt_index

    def capture(self, camera_data: Mapping[str, np.ndarray]) -> bool:
        if not self._active:
            raise RuntimeError("no attempt is active")
        mosaic = build_rgb_mosaic(camera_data, self.camera_configs)
        if mosaic is None:
            return False

        source_frame_index = self._source_frame_count
        self._source_frame_count += 1
        if source_frame_index % self.config.frame_stride != 0:
            return False

        try:
            if self._writer is None:
                self._open_writer(mosaic)
            self._writer.write(mosaic)
        except Exception:
            self._invalidate_active_attempt()
            raise

        self._frame_count += 1
        return True

    def _open_writer(self, frame: np.ndarray) -> None:
        if self._partial_path is None:
            raise RuntimeError("attempt partial path is unavailable")

        writer = self.writer_factory(
            str(self._partial_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.config.output_fps,
            (frame.shape[1], frame.shape[0]),
        )
        self._writer = writer
        if writer is None or not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {self._partial_path}")

    def finish(self, *, success: bool) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("no attempt is active")
        if type(success) is not bool:
            raise TypeError("success must be a boolean")
        status = "success" if success else "failed"
        return self._finalize(status=status, success=success)

    def abort(self) -> dict[str, Any] | None:
        if not self._active:
            return None
        return self._finalize(status="aborted", success=None)

    def close(self) -> dict[str, Any] | None:
        return self.abort()

    def _finalize(
        self,
        *,
        status: str,
        success: bool | None,
    ) -> dict[str, Any]:
        attempt_index = self._attempt_index
        partial_path = self._partial_path
        writer = self._writer
        frame_count = self._frame_count
        source_frame_count = self._source_frame_count
        self._clear_active_attempt()

        if attempt_index is None or partial_path is None:
            raise RuntimeError("active attempt state is incomplete")

        if writer is not None:
            writer.release()

        video_path = None
        keep_video = success is not True or self._record_current_attempt
        if frame_count > 0 and keep_video:
            final_path = self.output_dir / f"attempt_{attempt_index:04d}_{status}.mp4"
            os.replace(partial_path, final_path)
            video_path = final_path.name
        else:
            partial_path.unlink(missing_ok=True)

        record = {
            "attempt_index": attempt_index,
            "status": status,
            "success": success,
            "frame_count": frame_count,
            "source_frame_count": source_frame_count,
            "frame_stride": self.config.frame_stride,
            "fps": self.config.output_fps,
            "video_path": video_path,
        }
        self._append_manifest(record)
        return record

    def _append_manifest(self, record: Mapping[str, Any]) -> None:
        manifest_path = self.output_dir / "attempts.jsonl"
        with manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(dict(record)) + "\n")

    def _invalidate_active_attempt(self) -> None:
        writer = self._writer
        self._writer = None
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        finally:
            self._clear_active_attempt()

    def _clear_active_attempt(self) -> None:
        self._active = False
        self._attempt_index = None
        self._partial_path = None
        self._writer = None
        self._frame_count = 0
        self._source_frame_count = 0
