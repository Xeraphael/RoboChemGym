import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

import agent.main as agent_main
from agent.action.plan_orchestrator import (
    PlanOrchestrator,
    SimulationProcessError,
    SubprocessSimulationRunner,
)
from agent.action.action_orchestrator import (
    AgentOrchestrator,
    run_optimization_iterations,
)
from agent.plan_pipeline import ActionAgentPipeline, LegacyCodegenBackend
from agent.planning.protocol_planner import PlanningAttempt, PlanningResult
from agent.planning.registry import CapabilityRegistry
from agent.planning.validator import PlanValidator
from agent.scene.scene_compiler import SceneCompileError, SceneCompileResult
from agent.scene.scene_preflight import (
    ScenePreflightIssue,
    ScenePreflightReport,
)
from tests.action_agent.test_plan_models import make_plan
from tests.action_agent.test_plan_orchestrator import failed_report


ROOT = Path(__file__).resolve().parents[2]


def valid_planning_result():
    plan = make_plan()
    registry = CapabilityRegistry.load_default(ROOT)
    report = PlanValidator(registry).validate(plan)
    if not report.valid:
        raise AssertionError("test plan must be valid")
    return PlanningResult(
        status="valid",
        plan=plan,
        final_report=report,
        attempts=[
            PlanningAttempt(
                index=1,
                raw_response=plan.model_dump_json(),
                validation_report=report,
            )
        ],
    )


def failed_planning_result(status="planning_failed"):
    if status == "client_failed":
        attempt = PlanningAttempt(
            index=1,
            raw_response="",
            client_error="RuntimeError: unavailable",
        )
    else:
        attempt = PlanningAttempt(
            index=1,
            raw_response="not json",
            parse_error="invalid JSON",
        )
    return PlanningResult(status=status, attempts=[attempt])


class FakePlanner:
    def create_plan(self, protocol_text, max_attempts=3):
        raise AssertionError("planner should not run for an unknown backend")


class FakeCompiler:
    def compile(self, plan, artifacts):
        raise AssertionError("compiler should not run for an unknown backend")


class FakeOrchestrator:
    def run(self, plan, config_path, artifacts):
        raise AssertionError("orchestrator should not run for an unknown backend")


class FakeLegacyBackend:
    def generate(self, plan, artifacts, compiled):
        raise AssertionError("legacy generation should not run for an unknown backend")

    def execute(self, plan, artifacts, compiled):
        raise AssertionError("legacy execution should not run for an unknown backend")


class StaticPlanner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create_plan(self, protocol_text, max_attempts=3):
        self.calls.append((protocol_text, max_attempts))
        return self.result.model_copy(deep=True)


class SuccessfulCompiler:
    def __init__(self):
        self.calls = []

    def compile(self, plan, artifacts):
        self.calls.append((plan, artifacts))
        artifacts.write_plan(plan)
        artifacts.write_json(
            artifacts.validation_path,
            valid_planning_result().final_report,
        )
        output_dir = artifacts.run_dir / "compiled"
        output_dir.mkdir()
        usd_path = output_dir / "generated.usd"
        scene_json_path = output_dir / "generated.json"
        config_path = output_dir / "generated.yaml"
        usd_path.write_bytes(b"usd")
        scene_json_path.write_text("{}", encoding="utf-8")
        config_path.write_text("controller_type: plan_executor\n", encoding="utf-8")
        return SceneCompileResult(
            usd_path=usd_path,
            scene_json_path=scene_json_path,
            config_path=config_path,
            preflight=ScenePreflightReport(passed=True),
        )


class UnexpectedCompiler:
    def __init__(self):
        self.calls = []

    def compile(self, plan, artifacts):
        self.calls.append((plan, artifacts))
        raise AssertionError("compiler must not run")


class PreflightFailingCompiler:
    def __init__(self, validation):
        self.validation = validation
        self.calls = []

    def compile(self, plan, artifacts):
        self.calls.append((plan, artifacts))
        artifacts.write_plan(plan)
        artifacts.write_json(artifacts.validation_path, self.validation)
        artifacts.write_json(
            artifacts.scene_preflight_path,
            ScenePreflightReport(
                passed=False,
                issues=(
                    ScenePreflightIssue(
                        code="OUT_OF_REACH",
                        message="flask is unreachable",
                    ),
                ),
            ),
        )
        raise SceneCompileError(
            "SCENE_PREFLIGHT_FAILED",
            "scene preflight failed",
        )


class RecordingOrchestrator:
    def __init__(self, report=None):
        self.report = report or {
            "execution_success": True,
            "failed_step": None,
            "steps": [],
        }
        self.calls = []

    def run(
        self,
        plan,
        config_path,
        artifacts,
        *,
        scene_compile_result=None,
    ):
        self.calls.append(
            (plan, config_path, artifacts, scene_compile_result)
        )
        return self.report


class RecordingLegacyBackend:
    def __init__(self):
        self.generate_calls = []
        self.execute_calls = []
        self.report = {
            "execution_success": True,
            "failed_step": None,
            "steps": [],
        }

    def generate(self, plan, artifacts, config_path):
        self.generate_calls.append((plan, artifacts, config_path))

    def execute(self, plan, artifacts, config_path):
        self.execute_calls.append((plan, artifacts, config_path))
        return self.report


class ProcessFailingOrchestrator:
    def __init__(self):
        self.calls = []

    def run(self, plan, config_path, artifacts, *, scene_compile_result=None):
        self.calls.append((plan, config_path, artifacts, scene_compile_result))
        raise SimulationProcessError(
            "SIMULATION_PROCESS_FAILED",
            "simulation exited with status 3",
            returncode=3,
            stderr="failure",
        )


class RaisingCompiler:
    def __init__(self, exc):
        self.exc = exc

    def compile(self, plan, artifacts):
        raise self.exc


class RaisingOrchestrator:
    def __init__(self, exc):
        self.exc = exc

    def run(self, plan, config_path, artifacts, *, scene_compile_result=None):
        raise self.exc


class RaisingLegacyBackend:
    def __init__(self, *, generate_error=None, execute_error=None):
        self.generate_error = generate_error
        self.execute_error = execute_error

    def generate(self, plan, artifacts, config_path):
        if self.generate_error is not None:
            raise self.generate_error

    def execute(self, plan, artifacts, config_path):
        if self.execute_error is not None:
            raise self.execute_error
        return {
            "execution_success": True,
            "failed_step": None,
            "steps": [],
        }


class ActionAgentPipelineBackendTests(unittest.TestCase):
    def assert_terminal_failure(
        self,
        result,
        *,
        status,
        error_code,
        message,
    ):
        self.assertFalse(result["execution_success"])
        self.assertEqual(result["status"], status)
        self.assertEqual(result["error_code"], error_code)
        self.assertEqual(result["message"], message)
        self.assertIsNone(result["failed_step"])
        self.assertEqual(result["steps"], [])
        run_dir = Path(result["run_dir"])
        self.assertEqual(
            json.loads(
                (run_dir / "execution_report.json").read_text(encoding="utf-8")
            ),
            result,
        )

    def test_unknown_backend_fails_before_planning_or_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = ActionAgentPipeline(
                FakePlanner(),
                FakeCompiler(),
                FakeOrchestrator(),
                FakeLegacyBackend(),
                Path(tmp),
            )

            with self.assertRaisesRegex(ValueError, "unsupported execution backend"):
                pipeline.run("protocol", execution_backend="unknown")

    def test_default_backend_never_calls_legacy_codegen(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = StaticPlanner(valid_planning_result())
            compiler = SuccessfulCompiler()
            orchestrator = RecordingOrchestrator()
            legacy = RecordingLegacyBackend()
            pipeline = ActionAgentPipeline(
                planner,
                compiler,
                orchestrator,
                legacy,
                Path(tmp),
            )

            result = pipeline.run("protocol")

            self.assertTrue(result["execution_success"])
            self.assertEqual(len(orchestrator.calls), 1)
            self.assertEqual(legacy.generate_calls, [])
            self.assertEqual(legacy.execute_calls, [])
            self.assertEqual(planner.calls, [("protocol", 3)])

    def test_legacy_codegen_requires_explicit_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = RecordingOrchestrator()
            legacy = RecordingLegacyBackend()
            pipeline = ActionAgentPipeline(
                StaticPlanner(valid_planning_result()),
                SuccessfulCompiler(),
                orchestrator,
                legacy,
                Path(tmp),
            )

            result = pipeline.run(
                "protocol",
                execution_backend="legacy_codegen",
            )

            self.assertTrue(result["execution_success"])
            self.assertEqual(orchestrator.calls, [])
            self.assertEqual(len(legacy.generate_calls), 1)
            self.assertEqual(len(legacy.execute_calls), 1)

    def test_planning_terminal_states_write_artifacts_without_compiling(self):
        for status in ("planning_failed", "client_failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                compiler = UnexpectedCompiler()
                pipeline = ActionAgentPipeline(
                    StaticPlanner(failed_planning_result(status)),
                    compiler,
                    RecordingOrchestrator(),
                    RecordingLegacyBackend(),
                    Path(tmp),
                )

                result = pipeline.run("unsupported protocol")

                self.assertEqual(result["status"], status)
                self.assertFalse(result["execution_success"])
                self.assertEqual(compiler.calls, [])
                run_dir = Path(result["run_dir"])
                self.assertEqual(
                    (run_dir / "input_protocol.txt").read_text(encoding="utf-8"),
                    "unsupported protocol",
                )
                self.assertTrue((run_dir / "repair_history.json").is_file())
                self.assertTrue((run_dir / "validation_report.json").is_file())
                self.assertEqual(
                    json.loads((run_dir / "execution_report.json").read_text()),
                    result,
                )

    def test_terminal_enrichment_does_not_mutate_injected_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_report = {
                "execution_success": True,
                "failed_step": None,
                "steps": [],
                "details": {"attempt": 1},
            }
            before = deepcopy(raw_report)
            pipeline = ActionAgentPipeline(
                StaticPlanner(valid_planning_result()),
                SuccessfulCompiler(),
                RecordingOrchestrator(raw_report),
                RecordingLegacyBackend(),
                Path(tmp),
            )

            result = pipeline.run("protocol")

            self.assertEqual(raw_report, before)
            self.assertIsNot(result, raw_report)
            self.assertTrue(result["plan_valid"])
            self.assertTrue(result["scene_preflight_passed"])
            self.assertIn("protocol_coverage", result)

    def test_preflight_failure_writes_only_terminal_report_after_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            planning = valid_planning_result()
            compiler = PreflightFailingCompiler(planning.final_report)
            orchestrator = RecordingOrchestrator()
            legacy = RecordingLegacyBackend()
            pipeline = ActionAgentPipeline(
                StaticPlanner(planning),
                compiler,
                orchestrator,
                legacy,
                Path(tmp),
            )

            result = pipeline.run("protocol")

            self.assertEqual(result["status"], "scene_preflight_failed")
            self.assertEqual(result["error_code"], "SCENE_PREFLIGHT_FAILED")
            self.assertFalse(result["execution_success"])
            self.assertEqual(orchestrator.calls, [])
            self.assertEqual(legacy.generate_calls, [])
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "agent_plan.json").is_file())
            self.assertTrue((run_dir / "validation_report.json").is_file())
            preflight = json.loads(
                (run_dir / "scene_preflight.json").read_text(encoding="utf-8")
            )
            self.assertFalse(preflight["passed"])
            self.assertEqual(
                json.loads(
                    (run_dir / "execution_report.json").read_text(
                        encoding="utf-8"
                    )
                ),
                result,
            )

    def test_process_failure_writes_terminal_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = ProcessFailingOrchestrator()
            legacy = RecordingLegacyBackend()
            pipeline = ActionAgentPipeline(
                StaticPlanner(valid_planning_result()),
                SuccessfulCompiler(),
                orchestrator,
                legacy,
                Path(tmp),
            )

            result = pipeline.run("protocol")

            self.assertEqual(result["status"], "simulation_process_failed")
            self.assertEqual(result["error_code"], "SIMULATION_PROCESS_FAILED")
            self.assertEqual(result["returncode"], 3)
            self.assertEqual(legacy.generate_calls, [])
            self.assertEqual(legacy.execute_calls, [])
            self.assertEqual(
                json.loads(
                    (Path(result["run_dir"]) / "execution_report.json").read_text(
                        encoding="utf-8"
                    )
                ),
                result,
            )

    def test_unexpected_compiler_exception_writes_terminal_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = ActionAgentPipeline(
                StaticPlanner(valid_planning_result()),
                RaisingCompiler(RuntimeError("compiler exploded")),
                RecordingOrchestrator(),
                RecordingLegacyBackend(),
                Path(tmp),
            )

            result = pipeline.run("protocol")

            self.assert_terminal_failure(
                result,
                status="scene_compile_failed",
                error_code="SCENE_COMPILE_ERROR",
                message="RuntimeError: compiler exploded",
            )

    def test_scene_compile_failure_persists_bounded_worker_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            error = SceneCompileError(
                "SCENE_WORKER_PROCESS_FAILED",
                "worker failed",
                returncode=9,
                stdout="o" * 5000 + "STDOUT_END",
                stderr="e" * 5000 + "STDERR_END",
            )
            pipeline = ActionAgentPipeline(
                StaticPlanner(valid_planning_result()),
                RaisingCompiler(error),
                RecordingOrchestrator(),
                RecordingLegacyBackend(),
                Path(tmp),
            )

            result = pipeline.run("protocol")

            self.assertEqual(result["status"], "scene_compile_failed")
            self.assertEqual(
                result["error_code"],
                "SCENE_WORKER_PROCESS_FAILED",
            )
            self.assertEqual(result["returncode"], 9)
            self.assertLessEqual(len(result["stdout"]), 4000)
            self.assertLessEqual(len(result["stderr"]), 4000)
            self.assertTrue(result["stdout"].endswith("STDOUT_END"))
            self.assertTrue(result["stderr"].endswith("STDERR_END"))
            self.assertEqual(
                json.loads(
                    (
                        Path(result["run_dir"]) / "execution_report.json"
                    ).read_text(encoding="utf-8")
                ),
                result,
            )

    def test_unexpected_orchestrator_exception_writes_terminal_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = ActionAgentPipeline(
                StaticPlanner(valid_planning_result()),
                SuccessfulCompiler(),
                RaisingOrchestrator(ValueError("optimizer exploded")),
                RecordingLegacyBackend(),
                Path(tmp),
            )

            result = pipeline.run("protocol")

            self.assert_terminal_failure(
                result,
                status="execution_failed",
                error_code="PLAN_EXECUTION_ERROR",
                message="ValueError: optimizer exploded",
            )

    def test_unexpected_legacy_exceptions_write_terminal_reports(self):
        cases = (
            (
                RaisingLegacyBackend(
                    generate_error=RuntimeError("generation exploded")
                ),
                "legacy_codegen_failed",
                "LEGACY_CODEGEN_ERROR",
                "RuntimeError: generation exploded",
            ),
            (
                RaisingLegacyBackend(
                    execute_error=ValueError("execution exploded")
                ),
                "legacy_execution_failed",
                "LEGACY_EXECUTION_ERROR",
                "ValueError: execution exploded",
            ),
        )
        for backend, status, error_code, message in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                pipeline = ActionAgentPipeline(
                    StaticPlanner(valid_planning_result()),
                    SuccessfulCompiler(),
                    RecordingOrchestrator(),
                    backend,
                    Path(tmp),
                )

                result = pipeline.run(
                    "protocol",
                    execution_backend="legacy_codegen",
                )

                self.assert_terminal_failure(
                    result,
                    status=status,
                    error_code=error_code,
                    message=message,
                )

    def test_invalid_backend_report_writes_terminal_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = ActionAgentPipeline(
                StaticPlanner(valid_planning_result()),
                SuccessfulCompiler(),
                RecordingOrchestrator(report="not a report"),
                RecordingLegacyBackend(),
                Path(tmp),
            )

            result = pipeline.run("protocol")

            self.assert_terminal_failure(
                result,
                status="execution_report_failed",
                error_code="EXECUTION_REPORT_ERROR",
                message=(
                    "ValueError: execution backend must return a report object"
                ),
            )

    def test_non_json_backend_reports_write_safe_terminal_reports(self):
        cases = (
            (Path("/tmp/not-json"), "TypeError"),
            ({"not-json"}, "TypeError"),
            (float("nan"), "ValueError"),
        )
        for value, exception_name in cases:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                pipeline = ActionAgentPipeline(
                    StaticPlanner(valid_planning_result()),
                    SuccessfulCompiler(),
                    RecordingOrchestrator(
                        report={
                            "execution_success": True,
                            "failed_step": None,
                            "steps": [],
                            "invalid_value": value,
                        }
                    ),
                    RecordingLegacyBackend(),
                    Path(tmp),
                )

                result = pipeline.run("protocol")

                self.assertFalse(result["execution_success"])
                self.assertEqual(result["status"], "execution_report_failed")
                self.assertEqual(result["error_code"], "EXECUTION_REPORT_ERROR")
                self.assertTrue(result["message"].startswith(exception_name + ":"))
                report_path = Path(result["run_dir"]) / "execution_report.json"
                self.assertEqual(json.loads(report_path.read_text()), result)

    def test_process_failure_bypasses_parameter_and_scene_recovery(self):
        registry = CapabilityRegistry.load_default(ROOT)

        class ExplodingRunner:
            def run(self, config_path):
                raise SimulationProcessError(
                    "SIMULATION_TIMEOUT",
                    "timed out",
                )

        class UnexpectedParameterOptimizer:
            def propose(self, plan, report):
                raise AssertionError("parameter recovery must not run")

        def unexpected_scene_factory(artifacts):
            raise AssertionError("scene recovery must not run")

        orchestrator = PlanOrchestrator(
            ExplodingRunner(),
            UnexpectedParameterOptimizer(),
            unexpected_scene_factory,
            registry=registry,
            validator=PlanValidator(registry),
            scene_preflight=None,
            max_parameter_iterations=2,
            max_scene_iterations=2,
        )

        with self.assertRaises(SimulationProcessError):
            orchestrator.run(make_plan(), Path("config.yaml"), object())


class SubprocessSimulationRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_path = self.root / "run" / "generated.yaml"
        self.config_path.parent.mkdir()
        self.config_path.write_text("name: generated\n", encoding="utf-8")
        self.report_path = self.config_path.parent / "execution_report.json"

    def make_runner(self, *, headless=True):
        return SubprocessSimulationRunner(
            self.root,
            python_executable="python-test",
            timeout=17,
            headless=headless,
        )

    def test_success_uses_generated_config_and_removes_stale_report(self):
        self.report_path.write_text("stale", encoding="utf-8")

        def fake_run(cmd, **kwargs):
            self.assertFalse(self.report_path.exists())
            self.report_path.write_text(
                json.dumps(
                    {
                        "execution_success": True,
                        "failed_step": None,
                        "steps": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                cmd,
                [
                    "python-test",
                    "main.py",
                    "--config-dir",
                    str(self.config_path.parent),
                    "--config-name",
                    "generated",
                    "--headless",
                    "--/rtx/verifyDriverVersion/enabled=false",
                ],
            )
            self.assertNotIn("--no-video", cmd)
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertEqual(kwargs["timeout"], 17)
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        with patch(
            "agent.action.plan_orchestrator.subprocess.run",
            side_effect=fake_run,
        ):
            report = self.make_runner().run(self.config_path)

        self.assertTrue(report["execution_success"])

    def test_no_headless_flag_when_disabled(self):
        def fake_run(cmd, **kwargs):
            self.assertNotIn("--headless", cmd)
            self.report_path.write_text(
                json.dumps(
                    {
                        "execution_success": True,
                        "failed_step": None,
                        "steps": [],
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        with patch(
            "agent.action.plan_orchestrator.subprocess.run",
            side_effect=fake_run,
        ):
            self.make_runner(headless=False).run(self.config_path)

    def test_process_terminal_errors_are_typed(self):
        cases = (
            "nonzero",
            "timeout",
            "missing",
            "invalid_json",
            "invalid_report",
            "inconsistent_failure",
        )
        for case in cases:
            with self.subTest(case=case):
                self.report_path.unlink(missing_ok=True)

                def fake_run(cmd, **kwargs):
                    if case == "timeout":
                        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
                    if case == "invalid_json":
                        self.report_path.write_text("not json", encoding="utf-8")
                    elif case == "invalid_report":
                        self.report_path.write_text(
                            json.dumps({"execution_success": "yes"}),
                            encoding="utf-8",
                        )
                    elif case == "inconsistent_failure":
                        self.report_path.write_text(
                            json.dumps(
                                {
                                    "execution_success": False,
                                    "failed_step": None,
                                    "steps": [],
                                }
                            ),
                            encoding="utf-8",
                        )
                    return SimpleNamespace(
                        returncode=3 if case == "nonzero" else 0,
                        stderr="process stderr",
                        stdout="process stdout",
                    )

                with patch(
                    "agent.action.plan_orchestrator.subprocess.run",
                    side_effect=fake_run,
                ):
                    with self.assertRaises(SimulationProcessError) as caught:
                        self.make_runner().run(self.config_path)

                self.assertTrue(caught.exception.code.startswith("SIMULATION_"))


class PlanOrchestratorCompileResultTests(unittest.TestCase):
    def test_scene_factory_receives_compile_result_with_actual_paths(self):
        registry = CapabilityRegistry.load_default(ROOT)
        compiled = SceneCompileResult(
            usd_path=Path("/tmp/generated-scene.usd"),
            scene_json_path=Path("/tmp/generated-scene.json"),
            config_path=Path("/tmp/generated-config.yaml"),
            preflight=ScenePreflightReport(passed=True),
        )
        artifacts = object()
        factory_calls = []

        class FailedRunner:
            def run(self, config_path):
                return failed_report()

        class NoChangeOptimizer:
            def optimize_from_execution_report(self, report):
                return False

        def scene_factory(run_artifacts, scene_compile_result):
            factory_calls.append((run_artifacts, scene_compile_result))
            return NoChangeOptimizer()

        orchestrator = PlanOrchestrator(
            FailedRunner(),
            parameter_optimizer=None,
            scene_optimizer_factory=scene_factory,
            registry=registry,
            validator=PlanValidator(registry),
            scene_preflight=None,
            max_parameter_iterations=0,
            max_scene_iterations=1,
        )

        result = orchestrator.run(
            make_plan(),
            compiled.config_path,
            artifacts,
            scene_compile_result=compiled,
        )

        self.assertFalse(result["execution_success"])
        self.assertEqual(factory_calls, [(artifacts, compiled)])


class LegacyCodegenBackendTests(unittest.TestCase):
    def test_generate_returns_tuple_and_execute_uses_run_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller_path = root / "controllers" / "generated.py"
            config_path = root / "run" / "generated.yaml"
            config_path.parent.mkdir()
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "controller_type": "plan_executor",
                        "agent": {"execution_backend": "plan_executor"},
                    }
                ),
                encoding="utf-8",
            )
            artifacts = SimpleNamespace(
                legacy_actions_path=root / "run" / "legacy" / "actions.txt",
                run_dir=root / "run",
            )
            backend = LegacyCodegenBackend(
                root / "controllers",
                api_key="key",
                base_url="https://example.invalid/v1",
                model="legacy-model",
                max_iterations=4,
                python_executable="python-test",
                project_root=root,
                headless=True,
            )

            with patch(
                "agent.action.generation.code_generator.generate_controller_code",
                return_value=(
                    str(controller_path),
                    "GeneratedController",
                    "generated_controller",
                ),
            ) as generate:
                generated = backend.generate(make_plan(), artifacts, config_path)

            self.assertEqual(
                generated,
                (
                    str(controller_path),
                    "GeneratedController",
                    "generated_controller",
                ),
            )
            generate.assert_called_once_with(
                action_info_path=str(artifacts.legacy_actions_path),
                controllers_dir=str(root / "controllers"),
                api_key="key",
                base_url="https://example.invalid/v1",
                model="legacy-model",
            )
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["controller_type"], "generated_controller")
            self.assertEqual(config["agent"]["execution_backend"], "legacy_codegen")

            constructor_calls = []

            class FakeAgentOrchestrator:
                def __init__(self, **kwargs):
                    constructor_calls.append(kwargs)

                def run(self):
                    return True

            with patch(
                "agent.action.action_orchestrator.AgentOrchestrator",
                FakeAgentOrchestrator,
            ):
                report = backend.execute(make_plan(), artifacts, config_path)

            self.assertTrue(report["execution_success"])
            self.assertEqual(constructor_calls[0]["config_dir"], str(config_path.parent))
            self.assertEqual(constructor_calls[0]["config_name"], config_path.stem)
            self.assertTrue(constructor_calls[0]["headless"])


class CodeGeneratorClientTests(unittest.TestCase):
    def test_omitted_base_url_uses_openai_default(self):
        from agent.action.generation.code_generator import CodeGenerator

        with patch(
            "agent.action.generation.code_generator.OpenAI"
        ) as openai:
            CodeGenerator(api_key="key", base_url=None, model="test-model")

        openai.assert_called_once_with(api_key="key")

    def test_explicit_base_url_is_forwarded(self):
        from agent.action.generation.code_generator import CodeGenerator

        with patch(
            "agent.action.generation.code_generator.OpenAI"
        ) as openai:
            CodeGenerator(
                api_key="key",
                base_url="https://example.invalid/v1",
                model="test-model",
            )

        openai.assert_called_once_with(
            api_key="key",
            base_url="https://example.invalid/v1",
        )


class LegacyAgentOrchestratorTests(unittest.TestCase):
    def test_generated_config_directory_and_headless_flags_are_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = root / "controller.py"
            controller.write_text("class Controller: pass\n", encoding="utf-8")
            config_dir = root / "generated-config"
            config_dir.mkdir()
            orchestrator = AgentOrchestrator(
                controller_file=str(controller),
                config_name="generated",
                config_dir=str(config_dir),
                project_root=str(root),
                run_dir=str(root / "run"),
                python_executable="python-test",
                headless=True,
            )
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append((cmd, kwargs))
                Path(kwargs["env"]["AGENT_LOG_FILE"]).write_text(
                    "Task Complete: true\nACTION_FAILED=0\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            with patch(
                "agent.action.action_orchestrator.subprocess.run",
                side_effect=fake_run,
            ):
                success, _ = orchestrator.execute_simulation(1)

            self.assertTrue(success)
            self.assertEqual(
                calls[0][0],
                [
                    "python-test",
                    "main.py",
                    "--config-dir",
                    str(config_dir),
                    "--config-name",
                    "generated",
                    "--no-video",
                    "--headless",
                ],
            )

    def test_default_config_directory_remains_project_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = root / "controller.py"
            controller.write_text("pass\n", encoding="utf-8")
            orchestrator = AgentOrchestrator(
                controller_file=str(controller),
                config_name="legacy",
                project_root=str(root),
                run_dir=str(root / "run"),
            )

            self.assertEqual(orchestrator.config_dir, root / "config")
            self.assertFalse(orchestrator.headless)

    def test_legacy_helper_forwards_config_dir_and_headless(self):
        calls = []

        class FakeAgentOrchestrator:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def run(self):
                return True

        with patch(
            "agent.action.action_orchestrator.AgentOrchestrator",
            FakeAgentOrchestrator,
        ):
            success = run_optimization_iterations(
                "controller.py",
                "generated",
                config_dir="/tmp/generated-config",
                headless=True,
            )

        self.assertTrue(success)
        self.assertEqual(calls[0]["config_dir"], "/tmp/generated-config")
        self.assertTrue(calls[0]["headless"])
        source = (
            ROOT / "agent" / "action" / "action_orchestrator.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--config-dir"', source)
        self.assertIn('"--headless"', source)


class RootSimulationEntrypointTests(unittest.TestCase):
    def load_config_dir_resolver(self):
        source_path = ROOT / "main.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        resolver = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_resolve_config_dir"
            ),
            None,
        )
        self.assertIsNotNone(resolver, "main.py must expose a pure config resolver")
        namespace = {"Path": Path, "__file__": str(source_path)}
        exec(
            compile(
                ast.Module(body=[resolver], type_ignores=[]),
                str(source_path),
                "exec",
            ),
            namespace,
        )
        return namespace["_resolve_config_dir"]

    def test_generated_config_directory_and_attempt_video_policy_are_wired(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        main_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        def call_line(receiver, method):
            return min(
                node.lineno
                for node in ast.walk(main_function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == receiver
            )

        self.assertIn("--config-dir", source)
        self.assertIn("hydra.initialize_config_dir", source)
        self.assertIn("config_dir = _resolve_config_dir(args.config_dir)", source)
        self.assertIn(
            "show_video = not args.no_video and not args.headless",
            source,
        )
        self.assertIn(
            "attempt_video_config.enabled and not args.no_video",
            source,
        )
        self.assertNotIn("if args.no_video or args.headless:", source)
        self.assertLess(
            call_line("attempt_video_recorder", "capture"),
            call_line("task_controller", "step"),
        )
        self.assertLess(
            call_line("attempt_video_recorder", "finish"),
            call_line("task", "on_task_complete"),
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Try)
                and any(
                    isinstance(candidate, ast.Call)
                    and isinstance(candidate.func, ast.Attribute)
                    and candidate.func.attr == "close"
                    and isinstance(candidate.func.value, ast.Name)
                    and candidate.func.value.id == "attempt_video_recorder"
                    for final_node in node.finalbody
                    for candidate in ast.walk(final_node)
                )
                for node in ast.walk(main_function)
            ),
            "attempt video recorder must close from a finally block",
        )
        self.assertLess(
            source.index("args, kit_args = parse_args()"),
            source.index("prepare_isaacsim_argv(kit_args)"),
        )
        self.assertLess(
            source.index("prepare_isaacsim_argv(kit_args)"),
            source.index("from isaacsim import SimulationApp"),
        )

    def test_config_directory_default_is_repo_relative_but_explicit_is_cwd_relative(self):
        resolver = self.load_config_dir_resolver()
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                absolute = Path(tmp) / "absolute-config"
                self.assertEqual(
                    resolver(None, repository_root=ROOT),
                    ROOT / "config",
                )
                self.assertEqual(
                    resolver("relative-config", repository_root=ROOT),
                    Path(tmp) / "relative-config",
                )
                self.assertEqual(
                    resolver(str(absolute), repository_root=ROOT),
                    absolute,
                )
            finally:
                os.chdir(original_cwd)

    def test_reset_branch_checks_episode_limit_before_and_after_controller_reset(self):
        tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        main_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        def is_controller_call(node, method):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "task_controller"
            )

        reset_branch = next(
            node
            for node in ast.walk(main_function)
            if isinstance(node, ast.If)
            and any(
                is_controller_call(candidate, "need_reset")
                for candidate in ast.walk(node.test)
            )
        )
        reset_line = min(
            node.lineno
            for node in ast.walk(reset_branch)
            if is_controller_call(node, "reset")
        )
        limit_checks = [
            node
            for node in ast.walk(reset_branch)
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "task_controller.episode_num() >= cfg.max_episodes"
            and any(isinstance(candidate, ast.Break) for candidate in ast.walk(node))
        ]

        self.assertTrue(
            any(check.lineno < reset_line for check in limit_checks),
            "terminal plans must exit before controller cleanup/reset",
        )
        self.assertTrue(
            any(check.lineno > reset_line for check in limit_checks),
            "legacy controllers still need their post-reset episode check",
        )


class AgentMainWiringTests(unittest.TestCase):
    def test_legacy_constructor_none_defaults_resolve_from_repository_root(self):
        agent = agent_main.AgentMain(
            protocol_text="protocol",
            protocol_dir=None,
            scenes_dir=None,
            config_dir=None,
            controllers_dir=None,
        )

        self.assertEqual(agent.protocol_dir, ROOT / "agent" / "protocol")
        self.assertEqual(agent.scenes_dir, ROOT / "agent" / "scene" / "scenes")
        self.assertEqual(agent.config_dir, ROOT / "config")
        self.assertEqual(agent.controllers_dir, ROOT / "controllers")

    def test_cli_defaults_to_validated_headless_pipeline(self):
        parser = agent_main.build_arg_parser()
        args = parser.parse_args(["--protocol-text", "protocol"])

        self.assertEqual(args.execution_backend, "plan_executor")
        self.assertEqual(args.max_plan_attempts, 3)
        self.assertTrue(args.headless)
        self.assertEqual(args.run_root, "outputs/action_agent")
        self.assertEqual(args.controllers_dir, "controllers")
        self.assertFalse(args.allow_unsafe_codegen)

    def test_cli_rejects_legacy_codegen_without_explicit_unsafe_flag(self):
        with self.assertRaises(SystemExit):
            agent_main.main(
                [
                    "--protocol-text",
                    "protocol",
                    "--execution-backend",
                    "legacy_codegen",
                ]
            )

    def test_protocol_alias_sets_protocol_file(self):
        args = agent_main.build_arg_parser().parse_args(
            ["--protocol", "/tmp/protocol.txt"]
        )
        self.assertEqual(args.protocol_file, "/tmp/protocol.txt")
        self.assertIsNone(args.resume)

    def test_read_protocol_input_resolves_relative_protocol_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "agent" / "protocol" / "protocols" / "p.txt"
            protocol_path.parent.mkdir(parents=True)
            protocol_path.write_text("from file", encoding="utf-8")
            args = SimpleNamespace(
                protocol_text=None,
                protocol_file="p.txt",
                protocol_dir=None,
            )

            with patch.object(agent_main, "_project_root", root):
                self.assertEqual(agent_main.read_protocol_input(args), "from file")

    def test_read_protocol_input_prefers_a_cwd_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.txt"
            path.write_text("from cwd", encoding="utf-8")
            args = SimpleNamespace(
                protocol_text=None,
                protocol_file="protocol.txt",
                protocol_dir=None,
            )

            previous_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                self.assertEqual(agent_main.read_protocol_input(args), "from cwd")
            finally:
                os.chdir(previous_cwd)

    def test_read_protocol_input_honors_custom_protocol_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative_protocol_dir = root / "custom-protocol"
            relative_path = relative_protocol_dir / "protocols" / "relative.txt"
            relative_path.parent.mkdir(parents=True)
            relative_path.write_text("relative protocol", encoding="utf-8")
            absolute_protocol_dir = root / "absolute-protocol"
            absolute_path = absolute_protocol_dir / "protocols" / "absolute.txt"
            absolute_path.parent.mkdir(parents=True)
            absolute_path.write_text("absolute protocol", encoding="utf-8")

            with patch.object(agent_main, "_project_root", root):
                relative_args = SimpleNamespace(
                    protocol_text=None,
                    protocol_file="relative.txt",
                    protocol_dir="custom-protocol",
                )
                absolute_args = SimpleNamespace(
                    protocol_text=None,
                    protocol_file="absolute.txt",
                    protocol_dir=str(absolute_protocol_dir),
                )
                self.assertEqual(
                    agent_main.read_protocol_input(relative_args),
                    "relative protocol",
                )
                self.assertEqual(
                    agent_main.read_protocol_input(absolute_args),
                    "absolute protocol",
                )

    def make_build_args(self, **overrides):
        values = {
            "api_key": "explicit-key",
            "base_url": None,
            "model": "test-model",
            "simulation_timeout": 23,
            "headless": True,
            "max_action_iterations": 2,
            "max_scene_optimizations": 3,
            "run_root": "relative-runs",
            "controllers_dir": "relative-controllers",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_build_pipeline_requires_credentials_and_model(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                agent_main.build_plan_pipeline(self.make_build_args())
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "ACTION_AGENT_MODEL"):
                agent_main.build_plan_pipeline(self.make_build_args())

    def test_concrete_wiring_resolves_paths_and_uses_compile_result_scene_paths(self):
        client_calls = []
        optimizer_calls = []
        scene_backends = []

        class FakeOpenAIChatClient:
            def __init__(self, **kwargs):
                client_calls.append(kwargs)
                self.client = object()

        class FakeContinuousOptimizer:
            def __init__(self, **kwargs):
                optimizer_calls.append(kwargs)

        class FakeIsaacSubprocessSceneBackend:
            def __init__(self, root, **kwargs):
                self.root = root
                self.kwargs = kwargs
                self.position_updater_factory = object()
                scene_backends.append(self)

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "explicit-key",
                "ACTION_AGENT_MODEL": "test-model",
            },
            clear=True,
        ), patch(
            "agent.planning.llm_client.OpenAIChatClient",
            FakeOpenAIChatClient,
        ), patch(
            "agent.scene.optimization.continuous_optimizater.ContinuousOptimizer",
            FakeContinuousOptimizer,
        ), patch(
            "agent.scene.isaac_scene_worker.IsaacSubprocessSceneBackend",
            FakeIsaacSubprocessSceneBackend,
        ):
            pipeline = agent_main.build_plan_pipeline(self.make_build_args())

        self.assertEqual(client_calls, [{"api_key": "explicit-key"}])
        self.assertEqual(len(scene_backends), 1)
        self.assertEqual(scene_backends[0].root, ROOT)
        self.assertEqual(scene_backends[0].kwargs, {
            "python_executable": sys.executable,
            "timeout": 23,
        })
        self.assertIs(pipeline.scene_compiler.backend, scene_backends[0])
        self.assertEqual(pipeline.run_root, ROOT / "relative-runs")
        self.assertEqual(
            pipeline.legacy_codegen.controllers_dir,
            ROOT / "relative-controllers",
        )
        self.assertEqual(pipeline.legacy_codegen.model, "test-model")
        compiled = SceneCompileResult(
            usd_path=Path("/tmp/actual.usd"),
            scene_json_path=Path("/tmp/actual.json"),
            config_path=Path("/tmp/actual.yaml"),
            preflight=ScenePreflightReport(passed=True),
        )
        artifacts = SimpleNamespace(run_dir=Path("/tmp/run"))

        pipeline.orchestrator.scene_optimizer_factory(artifacts, compiled)

        self.assertEqual(optimizer_calls[0]["scene_usd_path"], compiled.usd_path)
        self.assertEqual(
            optimizer_calls[0]["scene_json_path"],
            compiled.scene_json_path,
        )
        self.assertEqual(optimizer_calls[0]["scenes_dir"], str(artifacts.run_dir))
        self.assertIs(
            optimizer_calls[0]["position_updater_factory"],
            scene_backends[0].position_updater_factory,
        )

    def test_building_default_pipeline_does_not_import_isaac_or_usd_modules(self):
        code = """
import sys
from types import SimpleNamespace
import agent.main as agent_main
import os

args = SimpleNamespace(
    api_key='test-key',
    base_url=None,
    model='test-model',
    simulation_timeout=23,
    headless=True,
    max_action_iterations=2,
    max_scene_optimizations=3,
    run_root='relative-runs',
    controllers_dir='relative-controllers',
)
os.environ['OPENAI_API_KEY'] = 'test-key'
os.environ['ACTION_AGENT_MODEL'] = 'test-model'
agent_main.build_plan_pipeline(args)
for prefix in ('isaacsim', 'omni', 'pxr'):
    assert not any(
        name == prefix or name.startswith(prefix + '.')
        for name in sys.modules
    ), prefix
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_environment_credentials_and_model_are_supported(self):
        client_calls = []

        class FakeOpenAIChatClient:
            def __init__(self, **kwargs):
                client_calls.append(kwargs)
                self.client = object()

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "environment-key",
                "OPENAI_BASE_URL": "https://environment.invalid/v1",
                "ACTION_AGENT_MODEL": "environment-model",
            },
            clear=True,
        ), patch(
            "agent.planning.llm_client.OpenAIChatClient",
            FakeOpenAIChatClient,
        ):
            pipeline = agent_main.build_plan_pipeline(
                self.make_build_args(api_key=None, base_url=None, model=None)
            )

        self.assertEqual(
            client_calls,
            [
                {
                    "api_key": "environment-key",
                    "base_url": "https://environment.invalid/v1",
                }
            ],
        )
        self.assertEqual(pipeline.planner.model, "environment-model")
        self.assertEqual(pipeline.legacy_codegen.model, "environment-model")

    def test_run_from_args_dispatches_default_and_explicit_legacy_backend(self):
        for backend in ("plan_executor", "legacy_codegen"):
            with self.subTest(backend=backend):
                calls = []

                class FakePipeline:
                    def run(self, protocol_text, **kwargs):
                        calls.append((protocol_text, kwargs))
                        return {"execution_success": True}

                args = self.make_build_args()
                args.protocol_text = "protocol"
                args.protocol_file = None
                args.resume = None
                args.execution_backend = backend
                args.max_plan_attempts = 2
                with patch.object(
                    agent_main,
                    "build_plan_pipeline",
                    return_value=FakePipeline(),
                ):
                    result = agent_main.run_from_args(args)

                self.assertTrue(result["execution_success"])
                self.assertEqual(
                    calls,
                    [
                        (
                            "protocol",
                            {
                                "execution_backend": backend,
                                "max_plan_attempts": 2,
                            },
                        )
                    ],
                )

    def test_run_from_args_resume_skips_protocol_and_pipeline(self):
        args = self.make_build_args()
        args.resume = Path("/tmp/existing-run")
        args.protocol_text = None
        args.protocol_file = None
        with patch.object(
            agent_main,
            "resume_run",
            return_value={"execution_success": True, "resumed": True},
        ) as resume, patch.object(
            agent_main,
            "build_plan_pipeline",
        ) as build:
            result = agent_main.run_from_args(args)

        self.assertTrue(result["resumed"])
        resume.assert_called_once_with(args)
        build.assert_not_called()

    def test_legacy_agent_main_forwards_its_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            class FakeAgentOrchestrator:
                def __init__(self, **kwargs):
                    calls.append(kwargs)
                    self.run_dir = root / "legacy-run"
                    self.run_dir.mkdir()

                def run(self):
                    return True

            agent = agent_main.AgentMain(
                protocol_text="protocol",
                protocol_dir=str(root / "protocol"),
                scenes_dir=str(root / "scenes"),
                config_dir=str(root / "generated-config"),
                controllers_dir=str(root / "controllers"),
            )
            with patch(
                "agent.action.action_orchestrator.AgentOrchestrator",
                FakeAgentOrchestrator,
            ):
                success, _ = agent.step5_run_action_orchestrator(
                    "controller.py",
                    "generated",
                )

            self.assertTrue(success)
            self.assertEqual(
                calls[0]["config_dir"],
                str(root / "generated-config"),
            )


if __name__ == "__main__":
    unittest.main()
