import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from agent.planning.registry import CapabilityRegistry
from agent.planning.validator import plan_fingerprint, registry_fingerprint
from agent.runtime.run_artifacts import RunArtifacts
from agent.scene.isaac_scene_worker import SceneWorkerError
from agent.scene.scene_compiler import SceneCompileError, SceneCompiler, ScenePreflightReport
from agent.scene.scene_preflight import ScenePreflightIssue
from tests.action_agent.test_plan_models import make_plan


ROOT = Path(__file__).resolve().parents[2]


class FakeSceneBackend:
    def __init__(self):
        self.objects = None
        self.build_calls = 0
        self.preflight_calls = 0

    def build(self, objects, *, output_usd, output_json, layout_profile):
        self.build_calls += 1
        self.objects = objects
        output_usd.write_text("fake usd", encoding="utf-8")
        output_json.write_text("{}", encoding="utf-8")

    def preflight(self, objects, *, usd_path, scene_json_path, layout_profile):
        self.preflight_calls += 1
        return ScenePreflightReport(passed=True, issues=[])


class IncompleteSceneBackend(FakeSceneBackend):
    def __init__(self, *, write_usd, write_json):
        super().__init__()
        self.write_usd = write_usd
        self.write_json = write_json

    def build(self, objects, *, output_usd, output_json, layout_profile):
        self.build_calls += 1
        if self.write_usd:
            output_usd.write_text("fake usd", encoding="utf-8")
        if self.write_json:
            output_json.write_text("{}", encoding="utf-8")


class FailedPreflightBackend(FakeSceneBackend):
    def preflight(self, objects, *, usd_path, scene_json_path, layout_profile):
        self.preflight_calls += 1
        return ScenePreflightReport(
            passed=False,
            issues=[ScenePreflightIssue(code="UNREACHABLE", message="object is unreachable")],
        )


class MutatingSceneBackend(FakeSceneBackend):
    def build(self, objects, *, output_usd, output_json, layout_profile):
        self.build_calls += 1
        objects[0].instance_name = "BuildMutated"
        objects[0].required_anchors["pick"].append("build_anchor")
        layout_profile["surface_z"] = -1.0
        output_usd.write_text("fake usd", encoding="utf-8")
        output_json.write_text("{}", encoding="utf-8")

    def preflight(self, objects, *, usd_path, scene_json_path, layout_profile):
        self.preflight_calls += 1
        self.preflight_names = [obj.instance_name for obj in objects]
        self.preflight_anchors = deepcopy(objects[0].required_anchors)
        self.preflight_layout = deepcopy(layout_profile)
        objects[0].instance_name = "PreflightMutated"
        objects[0].required_anchors["pick"].append("preflight_anchor")
        layout_profile["surface_z"] = -2.0
        return ScenePreflightReport(passed=True, issues=[])


class MaliciousPreflightBackend(FakeSceneBackend):
    def preflight(self, objects, *, usd_path, scene_json_path, layout_profile):
        self.preflight_calls += 1
        issue = ScenePreflightIssue(code="BYPASSED", message="validator was bypassed")
        return ScenePreflightReport.model_construct(passed=True, issues=[issue])


class WorkerFailingBackend(FakeSceneBackend):
    def __init__(self, phase, code):
        super().__init__()
        self.phase = phase
        self.code = code

    def build(self, objects, *, output_usd, output_json, layout_profile):
        if self.phase == "build":
            output_usd.write_text("partial usd", encoding="utf-8")
            output_json.write_text("partial json", encoding="utf-8")
            raise SceneWorkerError(
                self.code,
                f"{self.phase} worker failed",
                returncode=7,
                stdout="o" * 5000 + "STDOUT_END",
                stderr="e" * 5000 + "STDERR_END",
            )
        return super().build(
            objects,
            output_usd=output_usd,
            output_json=output_json,
            layout_profile=layout_profile,
        )

    def preflight(self, objects, *, usd_path, scene_json_path, layout_profile):
        if self.phase == "preflight":
            raise SceneWorkerError(
                self.code,
                f"{self.phase} worker failed",
                returncode=7,
                stdout="o" * 5000 + "STDOUT_END",
                stderr="e" * 5000 + "STDERR_END",
            )
        return super().preflight(
            objects,
            usd_path=usd_path,
            scene_json_path=scene_json_path,
            layout_profile=layout_profile,
        )


class UnexpectedPartialBuildBackend(FakeSceneBackend):
    def __init__(self, error=None):
        super().__init__()
        self.error = error or RuntimeError("unexpected build failure")

    def build(self, objects, *, output_usd, output_json, layout_profile):
        output_usd.write_text("partial usd", encoding="utf-8")
        output_json.write_text("partial json", encoding="utf-8")
        raise self.error


class CountingSceneCompiler(SceneCompiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolve_calls = 0

    def resolve_objects(self, plan):
        self.resolve_calls += 1
        return super().resolve_objects(plan)


class SceneCompilerTests(unittest.TestCase):
    def test_placement_target_is_aligned_with_source_at_reset(self):
        compiler = SceneCompiler(
            CapabilityRegistry.load_default(ROOT),
            FakeSceneBackend(),
            ROOT,
        )
        plan = make_plan()
        plan.scene.objects.append(
            plan.scene.objects[1].model_copy(update={
                "id": "target",
                "asset_id": "TargetPlatform",
                "instance_name": "TargetPlatform1",
                "role": "placement_target",
            })
        )
        plan.actions[1].target = "target"
        objects = compiler.resolve_objects(plan)

        alignments = compiler._placement_alignments(plan, objects)

        self.assertEqual(
            alignments,
            [{
                "object_path": "/World/ErlenmeyerFlask_Solid1",
                "target_path": "/World/TargetPlatform1",
            }],
        )

    def test_worker_error_codes_are_preserved_as_compile_errors(self):
        cases = [
            ("build", "SCENE_WORKER_BUILD_FAILED"),
            ("preflight", "USD_STAGE_OPEN_FAILED"),
        ]
        for phase, code in cases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
                backend = WorkerFailingBackend(phase, code)
                compiler = SceneCompiler(
                    CapabilityRegistry.load_default(ROOT),
                    backend,
                    ROOT,
                )

                with self.assertRaises(SceneCompileError) as raised:
                    compiler.compile(make_plan(), artifacts)

                self.assertEqual(raised.exception.code, code)
                self.assertEqual(str(raised.exception), f"{phase} worker failed")
                self.assertEqual(raised.exception.returncode, 7)
                self.assertLessEqual(len(raised.exception.stdout), 4000)
                self.assertLessEqual(len(raised.exception.stderr), 4000)
                self.assertTrue(raised.exception.stdout.endswith("STDOUT_END"))
                self.assertTrue(raised.exception.stderr.endswith("STDERR_END"))
                self.assertIsInstance(
                    raised.exception.__cause__,
                    SceneWorkerError,
                )
                self.assertFalse((artifacts.run_dir / "config.yaml").exists())
                if phase == "build":
                    self.assertFalse((artifacts.run_dir / "scene.usd").exists())
                    self.assertFalse((artifacts.run_dir / "scene.json").exists())

    def test_unexpected_build_failure_removes_partial_scene_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
            compiler = SceneCompiler(
                CapabilityRegistry.load_default(ROOT),
                UnexpectedPartialBuildBackend(),
                ROOT,
            )

            with self.assertRaisesRegex(RuntimeError, "unexpected build failure"):
                compiler.compile(make_plan(), artifacts)

            self.assertFalse((artifacts.run_dir / "scene.usd").exists())
            self.assertFalse((artifacts.run_dir / "scene.json").exists())

    def test_cleanup_failure_does_not_replace_worker_error_or_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
            compiler = SceneCompiler(
                CapabilityRegistry.load_default(ROOT),
                WorkerFailingBackend("build", "SCENE_WORKER_BUILD_FAILED"),
                ROOT,
            )

            with patch.object(
                Path,
                "unlink",
                side_effect=OSError("scene output is locked"),
            ), self.assertRaises(SceneCompileError) as raised:
                compiler.compile(make_plan(), artifacts)

            self.assertEqual(
                raised.exception.code,
                "SCENE_WORKER_BUILD_FAILED",
            )
            self.assertEqual(raised.exception.returncode, 7)
            self.assertTrue(raised.exception.stdout.endswith("STDOUT_END"))
            self.assertTrue(raised.exception.stderr.endswith("STDERR_END"))
            self.assertIsInstance(
                raised.exception.__cause__,
                SceneWorkerError,
            )

    def test_cleanup_failure_does_not_replace_unexpected_build_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
            original = RuntimeError("identity must survive cleanup")
            compiler = SceneCompiler(
                CapabilityRegistry.load_default(ROOT),
                UnexpectedPartialBuildBackend(original),
                ROOT,
            )

            with patch.object(
                Path,
                "unlink",
                side_effect=OSError("scene output is locked"),
            ), self.assertRaises(RuntimeError) as raised:
                compiler.compile(make_plan(), artifacts)

            self.assertIs(raised.exception, original)

    def test_compiles_registry_assets_and_plan_executor_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
            backend = FakeSceneBackend()
            compiler = SceneCompiler(
                CapabilityRegistry.load_default(ROOT),
                backend,
                ROOT,
            )
            path_type = type(artifacts.run_dir)
            replace_calls = []
            original_replace = path_type.replace

            def tracked_replace(source, target):
                replace_calls.append((source, Path(target)))
                return original_replace(source, target)

            with patch.object(path_type, "replace", tracked_replace):
                result = compiler.compile(make_plan(), artifacts)
            config = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
            self.assertTrue(result.preflight.passed)
            self.assertEqual(config["controller_type"], "plan_executor")
            self.assertEqual(config["mode"], "execute")
            self.assertEqual(config["agent"]["execution_backend"], "plan_executor")
            self.assertEqual(config["agent"]["plan_path"], str(artifacts.plan_path))
            self.assertEqual(config["agent"]["state_anchors"]["ErlenmeyerFlask_Solid1"], ["grisp_position"])
            self.assertEqual(config["multi_run"]["run_dir"], str(artifacts.run_dir / "data"))
            self.assertEqual(
                config["collector"]["video"],
                {
                    "enabled": True,
                    "frame_stride": 4,
                    "source_fps": 60,
                },
            )
            self.assertEqual(
                [obj.instance_name for obj in backend.objects],
                ["ErlenmeyerFlask_Solid1", "HeatingPlate"],
            )
            self.assertEqual(
                backend.objects[0].usd_path,
                "protocols/example_protocol/scene.usd",
            )
            self.assertTrue(artifacts.scene_preflight_path.is_file())
            self.assertEqual(
                replace_calls,
                [(artifacts.run_dir / ".config.yaml.tmp", artifacts.run_dir / "config.yaml")],
            )

    def test_invalid_plan_persists_validation_and_never_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
            backend = FakeSceneBackend()
            plan = make_plan().model_copy(deep=True)
            plan.actions[1] = plan.actions[0].model_copy(update={"id": "step_002"})
            compiler = SceneCompiler(CapabilityRegistry.load_default(ROOT), backend, ROOT)

            with self.assertRaises(SceneCompileError) as raised:
                compiler.compile(plan, artifacts)

            self.assertEqual(raised.exception.code, "PLAN_INVALID")
            self.assertEqual(backend.build_calls, 0)
            self.assertTrue(artifacts.validation_path.is_file())
            report = json.loads(artifacts.validation_path.read_text(encoding="utf-8"))
            self.assertFalse(report["valid"])
            self.assertEqual(report["plan_fingerprint"], plan_fingerprint(plan))
            self.assertEqual(
                report["registry_fingerprint"],
                registry_fingerprint(compiler.registry),
            )
            self.assertFalse((artifacts.run_dir / "config.yaml").exists())

    def test_missing_or_partial_outputs_fail_before_preflight(self):
        cases = [(False, False), (True, False), (False, True)]
        for write_usd, write_json in cases:
            with self.subTest(write_usd=write_usd, write_json=write_json):
                with tempfile.TemporaryDirectory() as tmp:
                    artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
                    backend = IncompleteSceneBackend(write_usd=write_usd, write_json=write_json)
                    compiler = SceneCompiler(CapabilityRegistry.load_default(ROOT), backend, ROOT)

                    with self.assertRaises(SceneCompileError) as raised:
                        compiler.compile(make_plan(), artifacts)

                    self.assertEqual(raised.exception.code, "SCENE_OUTPUT_MISSING")
                    self.assertEqual(backend.preflight_calls, 0)
                    self.assertFalse((artifacts.run_dir / "scene.usd").exists())
                    self.assertFalse((artifacts.run_dir / "scene.json").exists())
                    self.assertFalse((artifacts.run_dir / "config.yaml").exists())

    def test_failed_preflight_is_persisted_and_prevents_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
            backend = FailedPreflightBackend()
            compiler = SceneCompiler(CapabilityRegistry.load_default(ROOT), backend, ROOT)

            with self.assertRaises(SceneCompileError) as raised:
                compiler.compile(make_plan(), artifacts)

            self.assertEqual(raised.exception.code, "SCENE_PREFLIGHT_FAILED")
            report = json.loads(artifacts.scene_preflight_path.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(report["issues"][0]["code"], "UNREACHABLE")
            self.assertFalse((artifacts.run_dir / "config.yaml").exists())

    def test_preflight_report_rejects_contradictory_state(self):
        issue = ScenePreflightIssue(code="BROKEN", message="broken")
        contradictory_reports = [
            {"passed": True, "issues": [issue]},
            {"passed": False, "issues": []},
        ]
        for report in contradictory_reports:
            with self.subTest(report=report):
                with self.assertRaises(ValidationError):
                    ScenePreflightReport(**report)

    def test_preflight_report_issues_are_immutable(self):
        issue = ScenePreflightIssue(code="BROKEN", message="broken")
        report = ScenePreflightReport(passed=False, issues=[issue])

        self.assertIsInstance(report.issues, tuple)
        with self.assertRaises(AttributeError):
            report.issues.append(issue)
        with self.assertRaises(ValidationError):
            report.issues[0].message = "mutated"

    def test_bypassed_preflight_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
            backend = MaliciousPreflightBackend()
            compiler = SceneCompiler(CapabilityRegistry.load_default(ROOT), backend, ROOT)

            with self.assertRaises(SceneCompileError) as raised:
                compiler.compile(make_plan(), artifacts)

            self.assertEqual(raised.exception.code, "SCENE_PREFLIGHT_INVALID")
            self.assertFalse(artifacts.scene_preflight_path.exists())
            self.assertFalse((artifacts.run_dir / "config.yaml").exists())

    def test_backend_mutation_cannot_change_canonical_compiler_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = RunArtifacts.create(Path(tmp), "example_protocol")
            backend = MutatingSceneBackend()
            compiler = CountingSceneCompiler(CapabilityRegistry.load_default(ROOT), backend, ROOT)

            result = compiler.compile(make_plan(), artifacts)
            config = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))

            self.assertEqual(compiler.resolve_calls, 1)
            self.assertEqual(
                backend.preflight_names,
                ["ErlenmeyerFlask_Solid1", "HeatingPlate"],
            )
            self.assertEqual(backend.preflight_anchors["pick"], ["grisp_position"])
            self.assertEqual(backend.preflight_layout["surface_z"], 0.775)
            self.assertEqual(
                config["task"]["obj_paths"],
                [
                    {"path": "/World/ErlenmeyerFlask_Solid1"},
                    {"path": "/World/HeatingPlate"},
                ],
            )
            self.assertEqual(
                config["agent"]["state_anchors"]["ErlenmeyerFlask_Solid1"],
                ["grisp_position"],
            )
            published_profile = compiler.layout_profile
            published_profile["surface_z"] = -3.0
            self.assertEqual(compiler.layout_profile["surface_z"], 0.775)


if __name__ == "__main__":
    unittest.main()
