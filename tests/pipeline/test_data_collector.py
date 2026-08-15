import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from data_collectors.data_collector import DataCollector, resolve_action_target
from pipeline.dataset import build_dataset_identity
from policy.dataset.act_image_dataset import ACTImageDataset


CAMERAS = [
    {
        "name": "camera_1",
        "prim_path": "/World/Camera1",
        "image_type": "rgb",
        "resolution": [5, 4],
        "focal_length": 2.0,
    }
]


def add_step(collector, value=0.0):
    joints = np.arange(9, dtype=np.float32) + value
    action = SimpleNamespace(
        joint_positions=[None, 10, None, None, None, None, None, 0.02, 0.02]
    )
    collector.record_step(
        camera_images={"camera_1_rgb": np.zeros((3, 4, 5), dtype=np.uint8)},
        joint_positions=joints,
        action=action,
        timestamp=value,
        language_instruction="test",
    )


class DataCollectorTests(unittest.TestCase):
    def test_dataset_identity_binds_successful_episodes_and_splits(self):
        manifest = {
            "config_id": "collection",
            "protocol_id": "protocol1",
            "schema_version": "1.0",
            "episodes": [
                {
                    "episode_id": "protocol1-0000",
                    "status": "completed",
                    "success": True,
                    "length": 10,
                }
            ],
        }
        splits = {
            "split_version": "1.0",
            "seed": 7,
            "splits": {
                "train": ["protocol1-0000"],
                "validation": [],
                "test": [],
            },
        }
        identity = build_dataset_identity(manifest, splits)
        manifest["episodes"][0]["length"] = 11
        self.assertNotEqual(identity, build_dataset_identity(manifest, splits))

    def test_action_target_resolves_none_and_collapses_gripper(self):
        current = np.arange(9, dtype=np.float32)
        action = SimpleNamespace(
            joint_positions=[None, 10, None, None, None, None, None, 0.02, 0.02]
        )
        resolved = resolve_action_target(action, current)
        np.testing.assert_allclose(resolved, [0, 10, 2, 3, 4, 5, 6, 0.02])

    def test_commit_is_atomic_and_resume_continues_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = DataCollector(
                CAMERAS, tmp, max_episodes=3, protocol_id="protocol1"
            )
            self.assertEqual(collector.start_episode({"episode_seed": 4}), "protocol1-0000")
            add_step(collector)
            path = collector.commit_episode(
                {"execution_success": True, "failed_step": None, "steps": []}
            )

            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".h5.partial").exists())
            manifest = json.loads(
                (Path(tmp) / "dataset/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["episodes"][0]["status"], "completed")
            resumed = DataCollector(
                CAMERAS, tmp, max_episodes=3, protocol_id="protocol1"
            )
            self.assertEqual(resumed.start_episode(), "protocol1-0001")

    def test_orphaned_committed_hdf5_is_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = DataCollector(
                CAMERAS, tmp, max_episodes=3, protocol_id="protocol1"
            )
            collector.start_episode({"episode_seed": 4})
            add_step(collector)
            path = collector.commit_episode(
                {"execution_success": True, "failed_step": None, "steps": []}
            )
            (Path(tmp) / "dataset/manifest.json").unlink()
            (Path(tmp) / "dataset/reports/episode_0000.json").unlink()

            recovered = DataCollector(
                CAMERAS, tmp, max_episodes=3, protocol_id="protocol1"
            )
            self.assertEqual(recovered.episode_count, 1)
            self.assertTrue((Path(tmp) / "dataset/reports/episode_0000.json").is_file())
            with h5py.File(path, "r") as h5_file:
                self.assertEqual(h5_file["actions"].shape, (1, 8))

    def test_act_dataset_uses_persisted_splits_and_recorded_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = DataCollector(
                CAMERAS, tmp, max_episodes=6, protocol_id="protocol1"
            )
            for index in range(6):
                collector.start_episode({"episode_seed": index})
                add_step(collector, float(index))
                collector.commit_episode(
                    {"execution_success": True, "failed_step": None, "steps": []}
                )
            dataset_path = str(Path(tmp) / "dataset")
            train = ACTImageDataset(
                {}, dataset_path, horizon=2, split="train", in_memory=True
            )
            validation = train.get_validation_dataset()

            self.assertTrue(set(train.entry_by_id).isdisjoint(validation.entry_by_id))
            sample = train[0]
            episode_id, _ = train.sequences[0]
            expected = train.episode_data[episode_id]["actions"][0]
            pose = train.episode_data[episode_id]["agent_pose"][0]
            np.testing.assert_allclose(sample["action"][0].numpy(), expected)
            self.assertFalse(np.array_equal(expected, pose))
            self.assertEqual(sample["action"].shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
