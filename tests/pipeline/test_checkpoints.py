import tempfile
import unittest
from pathlib import Path

import torch

from pipeline.checkpoints import CheckpointManager


class CheckpointTests(unittest.TestCase):
    def test_latest_best_retention_and_compatibility(self):
        compatibility = {"model_type": "act", "schema": "1.0"}
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, mode="min", keep_numbered=2)
            for epoch, metric in enumerate((3.0, 2.0, 2.5)):
                manager.save(
                    {
                        "compatibility": compatibility,
                        "epoch": epoch,
                        "tensor": torch.tensor([epoch]),
                    },
                    metric=metric,
                    epoch=epoch,
                )

            latest = manager.load_latest(compatibility=compatibility)
            self.assertEqual(latest["epoch"], 2)
            self.assertEqual(manager.manifest["best"]["epoch"], 1)
            self.assertEqual(
                manager.manifest["numbered"],
                ["epoch_0001.ckpt", "epoch_0002.ckpt"],
            )
            self.assertFalse((Path(tmp) / "checkpoints/epoch_0000.ckpt").exists())
            self.assertFalse((Path(tmp) / "checkpoints/latest.ckpt.partial").exists())
            with self.assertRaisesRegex(ValueError, "incompatible"):
                manager.load_latest(compatibility={"model_type": "dp"})


if __name__ == "__main__":
    unittest.main()
