import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest

from data_collectors.attempt_video_recorder import (
    AttemptVideoConfig,
    AttemptVideoRecorder,
    build_rgb_mosaic,
)


def _solid_rgb(rgb, *, height=4, width=5):
    return np.full(
        (3, height, width),
        np.asarray(rgb, dtype=np.uint8)[:, None, None],
        dtype=np.uint8,
    )


def test_video_config_defaults_to_disabled_and_is_frozen():
    config = AttemptVideoConfig.from_mapping(None)

    assert config == AttemptVideoConfig(enabled=False)
    assert config.frame_stride == 4
    assert config.source_fps == 60.0
    assert config.output_fps == 15.0
    with pytest.raises(FrozenInstanceError):
        config.enabled = True


def test_video_config_accepts_valid_mapping_and_computes_output_fps():
    config = AttemptVideoConfig.from_mapping(
        {"enabled": True, "frame_stride": 3, "source_fps": 72}
    )

    assert config == AttemptVideoConfig(
        enabled=True,
        frame_stride=3,
        source_fps=72.0,
    )
    assert config.output_fps == 24.0


@pytest.mark.parametrize("enabled", [0, 1, "true", None, np.bool_(True)])
def test_video_config_rejects_non_boolean_enabled(enabled):
    with pytest.raises((TypeError, ValueError), match="enabled"):
        AttemptVideoConfig.from_mapping({"enabled": enabled})


@pytest.mark.parametrize("frame_stride", [True, False, 0, -1, 1.5, "2", None])
def test_video_config_rejects_invalid_frame_stride(frame_stride):
    with pytest.raises((TypeError, ValueError), match="frame_stride"):
        AttemptVideoConfig.from_mapping({"frame_stride": frame_stride})


@pytest.mark.parametrize(
    "source_fps",
    [True, False, 0, -0.5, float("inf"), float("-inf"), float("nan"), "60", None],
)
def test_video_config_rejects_invalid_source_fps(source_fps):
    with pytest.raises((TypeError, ValueError), match="source_fps"):
        AttemptVideoConfig.from_mapping({"source_fps": source_fps})


def test_build_rgb_mosaic_uses_config_order_and_converts_rgb_to_bgr():
    cameras = [
        {"name": "right", "image_type": "rgb"},
        {"name": "left", "image_type": "rgb"},
    ]
    frames = {
        "left_rgb": _solid_rgb([10, 20, 30]),
        "right_rgb": _solid_rgb([40, 50, 60]),
    }

    mosaic = build_rgb_mosaic(frames, cameras, draw_labels=False)

    assert mosaic.shape == (4, 10, 3)
    np.testing.assert_array_equal(mosaic[0, 0], [60, 50, 40])
    np.testing.assert_array_equal(mosaic[0, 5], [30, 20, 10])


def test_build_rgb_mosaic_selects_rgb_from_combined_image_types():
    cameras = [
        {"name": "depth_only", "image_type": "depth"},
        {"name": "combined", "image_type": "depth+rgb"},
    ]
    frames = {
        "depth_only_rgb": _solid_rgb([1, 2, 3]),
        "combined_rgb": _solid_rgb([4, 5, 6]),
    }

    mosaic = build_rgb_mosaic(frames, cameras, draw_labels=False)

    assert mosaic.shape == (4, 5, 3)
    np.testing.assert_array_equal(mosaic[0, 0], [6, 5, 4])


@pytest.mark.parametrize(
    "invalid_frame",
    [
        np.zeros((3, 4, 5), dtype=np.float32),
        np.zeros((4, 5, 3), dtype=np.uint8),
        np.zeros((1, 4, 5), dtype=np.uint8),
        np.zeros((3, 0, 5), dtype=np.uint8),
        [[[]]],
    ],
)
def test_build_rgb_mosaic_ignores_invalid_frames(invalid_frame):
    cameras = [{"name": "front", "image_type": "rgb"}]

    assert (
        build_rgb_mosaic(
            {"front_rgb": invalid_frame},
            cameras,
            draw_labels=False,
        )
        is None
    )


def test_build_rgb_mosaic_returns_none_without_an_eligible_stream():
    cameras = [
        {"name": "missing", "image_type": "rgb"},
        {"name": "depth", "image_type": "depth"},
    ]

    assert build_rgb_mosaic({}, cameras) is None


def test_build_rgb_mosaic_resizes_to_smallest_height_preserving_aspect_ratio():
    cameras = [
        {"name": "large", "image_type": "rgb"},
        {"name": "small", "image_type": "rgb"},
    ]
    frames = {
        "large_rgb": _solid_rgb([10, 20, 30], height=8, width=12),
        "small_rgb": _solid_rgb([40, 50, 60], height=4, width=5),
    }

    mosaic = build_rgb_mosaic(frames, cameras, draw_labels=False)

    assert mosaic.shape == (4, 11, 3)
    np.testing.assert_array_equal(mosaic[0, 0], [30, 20, 10])
    np.testing.assert_array_equal(mosaic[0, 6], [60, 50, 40])


def test_build_rgb_mosaic_draws_each_camera_label(monkeypatch):
    calls = []

    def fake_put_text(image, text, origin, font, scale, color, thickness, line_type):
        calls.append((text, origin, font, scale, color, thickness, line_type))
        image[0, 0] = [1, 2, 3]
        return image

    monkeypatch.setattr(cv2, "putText", fake_put_text)
    cameras = [
        {"name": "front", "image_type": "rgb"},
        {"name": "wrist", "image_type": "rgb"},
    ]

    mosaic = build_rgb_mosaic(
        {
            "front_rgb": _solid_rgb([10, 20, 30]),
            "wrist_rgb": _solid_rgb([40, 50, 60]),
        },
        cameras,
    )

    assert [call[0] for call in calls] == ["front", "wrist"]
    np.testing.assert_array_equal(mosaic[0, 0], [1, 2, 3])
    np.testing.assert_array_equal(mosaic[0, 5], [1, 2, 3])


class FakeWriter:
    def __init__(self, *, opened=True, write_error=None):
        self.opened = opened
        self.write_error = write_error
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        if self.write_error is not None:
            raise self.write_error
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FakeWriterFactory:
    def __init__(self, writer=None):
        self.writer = writer or FakeWriter()
        self.calls = []

    def __call__(self, path, fourcc, fps, frame_size):
        path = Path(path)
        path.touch()
        self.calls.append(
            {
                "path": path,
                "fourcc": fourcc,
                "fps": fps,
                "frame_size": frame_size,
            }
        )
        return self.writer


def _front_frame(value, *, dtype=np.uint8):
    return {"front_rgb": np.full((3, 4, 5), value, dtype=dtype)}


def _enabled_recorder(tmp_path, *, writer_factory=None, frame_stride=2):
    return AttemptVideoRecorder(
        output_dir=tmp_path / "videos",
        camera_configs=[{"name": "front", "image_type": "rgb"}],
        config=AttemptVideoConfig(
            enabled=True,
            frame_stride=frame_stride,
            source_fps=60,
        ),
        writer_factory=writer_factory,
    )


def _read_manifest(output_dir):
    manifest_path = output_dir / "attempts.jsonl"
    return [json.loads(line) for line in manifest_path.read_text().splitlines()]


def test_recorder_samples_source_frame_zero_and_every_stride(tmp_path):
    factory = FakeWriterFactory()
    recorder = _enabled_recorder(tmp_path, writer_factory=factory, frame_stride=2)

    assert recorder.start_attempt() == 0
    capture_results = [recorder.capture(_front_frame(value)) for value in range(5)]
    record = recorder.finish(success=False)

    assert capture_results == [True, False, True, False, True]
    assert [int(frame[0, 0, 0]) for frame in factory.writer.frames] == [0, 2, 4]
    assert record == {
        "attempt_index": 0,
        "status": "failed",
        "success": False,
        "frame_count": 3,
        "source_frame_count": 5,
        "frame_stride": 2,
        "fps": 30.0,
        "video_path": "attempt_0000_failed.mp4",
    }
    assert factory.calls == [
        {
            "path": tmp_path / "videos" / "attempt_0000.partial.mp4",
            "fourcc": cv2.VideoWriter_fourcc(*"mp4v"),
            "fps": 30.0,
            "frame_size": (5, 4),
        }
    ]
    assert factory.writer.released
    assert (tmp_path / "videos" / "attempt_0000_failed.mp4").is_file()
    assert not (tmp_path / "videos" / "attempt_0000.partial.mp4").exists()


def test_recorder_counts_only_eligible_source_frames_for_sampling(tmp_path):
    factory = FakeWriterFactory()
    recorder = _enabled_recorder(tmp_path, writer_factory=factory, frame_stride=2)
    recorder.start_attempt()

    assert recorder.capture(_front_frame(10)) is True
    assert recorder.capture(_front_frame(99, dtype=np.float32)) is False
    assert recorder.capture({}) is False
    assert recorder.capture(_front_frame(11)) is False
    assert recorder.capture(_front_frame(12)) is True
    record = recorder.finish(success=True)

    assert [int(frame[0, 0, 0]) for frame in factory.writer.frames] == [10, 12]
    assert record["source_frame_count"] == 3
    assert record["frame_count"] == 2


@pytest.mark.parametrize(
    ("success", "status"),
    [(True, "success"), (False, "failed")],
)
def test_finish_finalizes_the_requested_status(tmp_path, success, status):
    factory = FakeWriterFactory()
    recorder = _enabled_recorder(tmp_path, writer_factory=factory)
    recorder.start_attempt()
    recorder.capture(_front_frame(1))

    record = recorder.finish(success=success)

    assert record["status"] == status
    assert record["success"] is success
    assert record["video_path"] == f"attempt_0000_{status}.mp4"
    assert (tmp_path / "videos" / record["video_path"]).is_file()
    assert _read_manifest(tmp_path / "videos") == [record]


def test_abort_finalizes_an_active_attempt_as_aborted(tmp_path):
    factory = FakeWriterFactory()
    recorder = _enabled_recorder(tmp_path, writer_factory=factory)
    recorder.start_attempt()
    recorder.capture(_front_frame(1))

    record = recorder.abort()

    assert record["status"] == "aborted"
    assert record["success"] is None
    assert record["video_path"] == "attempt_0000_aborted.mp4"
    assert factory.writer.released
    assert (tmp_path / "videos" / "attempt_0000_aborted.mp4").is_file()
    assert _read_manifest(tmp_path / "videos") == [record]


def test_no_frame_attempt_writes_null_manifest_path_without_opening_writer(tmp_path):
    factory = FakeWriterFactory()
    recorder = _enabled_recorder(tmp_path, writer_factory=factory)
    recorder.start_attempt()

    record = recorder.finish(success=False)

    assert record["frame_count"] == 0
    assert record["source_frame_count"] == 0
    assert record["video_path"] is None
    assert factory.calls == []
    assert list((tmp_path / "videos").glob("*.mp4")) == []
    assert _read_manifest(tmp_path / "videos") == [record]


def test_manifest_appends_exactly_one_record_per_attempt(tmp_path):
    first_factory = FakeWriterFactory()
    recorder = _enabled_recorder(tmp_path, writer_factory=first_factory)
    recorder.start_attempt()
    recorder.capture(_front_frame(1))
    first_record = recorder.finish(success=True)

    recorder.writer_factory = FakeWriterFactory()
    assert recorder.start_attempt() == 1
    recorder.capture(_front_frame(2))
    second_record = recorder.finish(success=False)

    assert _read_manifest(tmp_path / "videos") == [first_record, second_record]


def test_attempt_index_scans_all_existing_video_statuses(tmp_path):
    output_dir = tmp_path / "videos"
    output_dir.mkdir()
    for name in [
        "attempt_0000_success.mp4",
        "attempt_0002_failed.mp4",
        "attempt_0004_aborted.mp4",
        "attempt_0007.partial.mp4",
        "attempt_9999_unknown.mp4",
        "not_an_attempt_10000_success.mp4",
    ]:
        (output_dir / name).touch()
    recorder = _enabled_recorder(tmp_path, writer_factory=FakeWriterFactory())

    assert recorder.start_attempt() == 8


def test_nested_start_attempt_is_rejected(tmp_path):
    recorder = _enabled_recorder(tmp_path, writer_factory=FakeWriterFactory())
    recorder.start_attempt()

    with pytest.raises(RuntimeError, match="already active"):
        recorder.start_attempt()


def test_writer_open_failure_releases_without_false_finalization(tmp_path):
    writer = FakeWriter(opened=False)
    recorder = _enabled_recorder(
        tmp_path,
        writer_factory=FakeWriterFactory(writer),
    )
    recorder.start_attempt()

    with pytest.raises(RuntimeError, match="open"):
        recorder.capture(_front_frame(1))

    assert writer.released
    assert (tmp_path / "videos" / "attempt_0000.partial.mp4").is_file()
    assert list((tmp_path / "videos").glob("attempt_0000_*.mp4")) == []
    assert not (tmp_path / "videos" / "attempts.jsonl").exists()
    assert recorder.close() is None


def test_writer_exception_releases_without_false_finalization(tmp_path):
    writer = FakeWriter(write_error=OSError("disk full"))
    recorder = _enabled_recorder(
        tmp_path,
        writer_factory=FakeWriterFactory(writer),
    )
    recorder.start_attempt()

    with pytest.raises(OSError, match="disk full"):
        recorder.capture(_front_frame(1))

    assert writer.released
    assert (tmp_path / "videos" / "attempt_0000.partial.mp4").is_file()
    assert list((tmp_path / "videos").glob("attempt_0000_*.mp4")) == []
    assert not (tmp_path / "videos" / "attempts.jsonl").exists()
    assert recorder.close() is None


def test_close_is_idempotent_while_inactive(tmp_path):
    recorder = _enabled_recorder(tmp_path, writer_factory=FakeWriterFactory())

    assert recorder.close() is None
    assert recorder.close() is None
    assert recorder.abort() is None


def test_close_aborts_an_active_attempt_once(tmp_path):
    factory = FakeWriterFactory()
    recorder = _enabled_recorder(tmp_path, writer_factory=factory)
    recorder.start_attempt()
    recorder.capture(_front_frame(1))

    record = recorder.close()

    assert record["status"] == "aborted"
    assert record["success"] is None
    assert factory.writer.released
    assert recorder.close() is None
    assert _read_manifest(tmp_path / "videos") == [record]
