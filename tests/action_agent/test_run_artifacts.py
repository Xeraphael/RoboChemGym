import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from agent.runtime.run_artifacts import RunArtifacts
from tests.action_agent.test_plan_models import make_plan


class RunArtifactsTests(unittest.TestCase):
    def test_open_requires_complete_existing_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                RunArtifacts.open(run_dir)
            for name in (
                "agent_plan.json",
                "validation_report.json",
            ):
                (run_dir / name).write_text("{}", encoding="utf-8")

            artifacts = RunArtifacts.open(run_dir)

            self.assertEqual(artifacts.run_dir, run_dir.resolve())
            self.assertEqual(
                artifacts.config_path,
                run_dir.resolve() / "config.yaml",
            )

    def test_writes_canonical_and_legacy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(
                Path(tmp),
                "example_protocol",
                now=datetime(2026, 7, 17, 15, 30, 0),
            )
            plan = make_plan()
            artifacts.write_protocol("test protocol")
            artifacts.write_plan(plan)
            artifacts.write_legacy_exports(plan)
            self.assertEqual(artifacts.run_dir.name, "20260717_153000_example_protocol")
            self.assertEqual(json.loads(artifacts.plan_path.read_text())["plan_id"], "example_protocol")
            self.assertIn("ErlenmeyerFlask_Solid1", artifacts.legacy_equipment_path.read_text())
            self.assertIn("pick solid_flask", artifacts.legacy_actions_path.read_text())

    def test_same_second_runs_receive_distinct_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = datetime(2026, 7, 17, 15, 30, 0)
            first = RunArtifacts.create(Path(tmp), "example_protocol", now=now)
            second = RunArtifacts.create(Path(tmp), "example_protocol", now=now)
            self.assertNotEqual(first.run_dir, second.run_dir)
            self.assertEqual(second.run_dir.name, "20260717_153000_example_protocol_01")

    def test_concurrent_same_second_runs_receive_unique_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime(2026, 7, 17, 15, 30, 0)
            workers = 8
            base_name = "20260717_153000_example_protocol"
            exists_barrier = Barrier(workers)
            original_exists = Path.exists

            def synchronized_exists(path):
                if path.parent == root and path.name == base_name:
                    exists_barrier.wait(timeout=5)
                    return False
                return original_exists(path)

            def create_artifacts(_):
                return RunArtifacts.create(root, "example_protocol", now=now)

            with patch.object(Path, "exists", synchronized_exists):
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    artifacts = list(executor.map(create_artifacts, range(workers)))

            run_dirs = [item.run_dir for item in artifacts]
            self.assertEqual(len(set(run_dirs)), workers)
            self.assertTrue(all((run_dir / "legacy").is_dir() for run_dir in run_dirs))

    def test_invalid_plan_ids_are_rejected(self):
        invalid_plan_ids = [
            "",
            "ExampleProtocol",
            "protocol-1",
            "protocol/1",
            "..",
            "a" * 129,
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for plan_id in invalid_plan_ids:
                with self.subTest(plan_id=plan_id):
                    with self.assertRaises(ValueError):
                        RunArtifacts.create(Path(tmp), plan_id)

    def test_traversal_plan_id_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            escaped = Path(tmp) / "escaped"
            error = None
            try:
                RunArtifacts.create(
                    root,
                    "x/../../escaped",
                    now=datetime(2026, 7, 17, 15, 30, 0),
                )
            except ValueError as exc:
                error = exc
            self.assertFalse(escaped.exists())
            self.assertIsInstance(error, ValueError)

    def test_relative_root_is_resolved_before_later_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            workspace = Path(tmp)
            other_cwd = workspace / "other"
            other_cwd.mkdir()
            try:
                os.chdir(workspace)
                artifacts = RunArtifacts.create(
                    Path("runs"),
                    "example_protocol",
                    now=datetime(2026, 7, 17, 15, 30, 0),
                )
                os.chdir(other_cwd)
                artifacts.write_protocol("test protocol")
            finally:
                os.chdir(original_cwd)
            self.assertTrue(artifacts.run_dir.is_absolute())
            self.assertEqual(
                artifacts.run_dir,
                workspace / "runs" / "20260717_153000_example_protocol",
            )
            self.assertEqual(
                (artifacts.run_dir / "input_protocol.txt").read_text(),
                "test protocol",
            )

    def test_write_json_rejects_external_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            artifacts = RunArtifacts.create(root, "example_protocol")
            external_path = Path(tmp) / "external.json"
            with self.assertRaises(ValueError):
                artifacts.write_json(external_path, {"status": "invalid"})
            self.assertFalse(external_path.exists())

    def test_write_json_rejects_relative_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            artifacts = RunArtifacts.create(root, "example_protocol")
            escaped_path = root / "escaped.json"
            original_cwd = Path.cwd()
            try:
                os.chdir(artifacts.run_dir)
                with self.assertRaises(ValueError):
                    artifacts.write_json(Path("../escaped.json"), {"status": "invalid"})
            finally:
                os.chdir(original_cwd)
            self.assertFalse(escaped_path.exists())

    def test_write_json_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            artifacts = RunArtifacts.create(root, "example_protocol")
            external_dir = Path(tmp) / "external"
            external_dir.mkdir()
            (artifacts.run_dir / "external").symlink_to(
                external_dir,
                target_is_directory=True,
            )
            original_cwd = Path.cwd()
            try:
                os.chdir(artifacts.run_dir)
                with self.assertRaises(ValueError):
                    artifacts.write_json(
                        Path("external/escaped.json"),
                        {"status": "invalid"},
                    )
            finally:
                os.chdir(original_cwd)
            self.assertFalse((external_dir / "escaped.json").exists())

    def test_write_json_accepts_paths_inside_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(Path(tmp) / "runs", "example_protocol")
            artifacts.write_json(artifacts.validation_path, {"canonical": True})
            original_cwd = Path.cwd()
            try:
                os.chdir(Path(tmp))
                artifacts.write_json(Path("relative.json"), {"relative": True})
            finally:
                os.chdir(original_cwd)
            self.assertEqual(
                json.loads(artifacts.validation_path.read_text()),
                {"canonical": True},
            )
            self.assertEqual(
                json.loads((artifacts.run_dir / "relative.json").read_text()),
                {"relative": True},
            )


if __name__ == "__main__":
    unittest.main()
