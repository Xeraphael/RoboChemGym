import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from agent.action.optimization.plan_parameter_optimizer import (
    ParameterPatch,
    PlanParameterOptimizer,
    apply_parameter_patch,
)
from agent.action.plan_orchestrator import PlanOrchestrator
from agent.planning.registry import CapabilityRegistry
from agent.planning.validator import (
    PlanValidator,
    plan_fingerprint,
    registry_fingerprint,
)
from agent.scene.optimization.continuous_optimizater import ContinuousOptimizer
from agent.scene.scene_preflight import ScenePreflightIssue, ScenePreflightReport
from tests.action_agent.test_plan_models import make_plan


ROOT = Path(__file__).resolve().parents[2]


def failed_report(
    step_id="step_001",
    *,
    code="GRASP_NOT_ESTABLISHED",
    measurements=None,
):
    action = "pick" if step_id == "step_001" else "place"
    return {
        "execution_success": False,
        "failed_step": step_id,
        "steps": [
            {
                "step_id": step_id,
                "action": action,
                "object_id": "solid_flask",
                "target_id": None if action == "pick" else "plate",
                "coverage_level": "supported",
                "adapter": f"{action}_adapter",
                "verifier": f"{action}_verifier",
                "attempt_count": 1,
                "success": False,
                "start_frame": 1,
                "end_frame": 2,
                "controller_completed": True,
                "semantic_requirements": [],
                "verification": {
                    "success": False,
                    "code": code,
                    "message": "verification failed",
                    "measurements": measurements or {"lift": 0.02},
                    "verification_level": "state_observed",
                },
            }
        ],
    }


def success_report(plan=None):
    plan = plan or make_plan()
    return {
        "execution_success": True,
        "failed_step": None,
        "steps": [
            {
                "step_id": step.id,
                "action": step.type.value,
                "object_id": step.object,
                "target_id": step.target,
                "coverage_level": "supported",
                "adapter": f"{step.type.value}_adapter",
                "verifier": f"{step.type.value}_verifier",
                "attempt_count": 1,
                "success": True,
                "start_frame": index * 10 + 1,
                "end_frame": index * 10 + 2,
                "controller_completed": True,
                "semantic_requirements": [],
                "verification": {
                    "success": True,
                    "code": "OK",
                    "message": "verified",
                    "measurements": {},
                    "verification_level": "state_observed",
                },
            }
            for index, step in enumerate(plan.actions)
        ],
    }


class FakeSimulationRunner:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def run(self, config_path):
        self.calls.append(config_path)
        return next(self.results)


class FakeOptimizer:
    def __init__(self, patches):
        self.patches = iter(patches)
        self.calls = []

    def propose(self, plan, report):
        self.calls.append((plan, report))
        return next(self.patches)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, messages, *, model, temperature):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
            }
        )
        return self.response


class FakeArtifacts:
    def __init__(self):
        self.plans = []
        self.json_writes = []
        self.validation_path = Path("validation_report.json")
        self.scene_preflight_path = Path("scene_preflight.json")

    def write_plan(self, plan):
        self.plans.append(plan.model_copy(deep=True))

    def write_json(self, path, value):
        self.json_writes.append((path, value))


class FakeSceneClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.response),
                )
            ]
        )


class FakePositionUpdater:
    def __init__(self, behavior="success"):
        self.behavior = behavior
        self.calls = []
        self.applied_scene = None

    def apply_positions_to_usd(
        self,
        json_file_path,
        usd_file_path,
        output_usd_path=None,
        in_place=True,
        required_prim_paths=None,
    ):
        self.calls.append(
            {
                "json_file_path": Path(json_file_path),
                "usd_file_path": Path(usd_file_path),
                "output_usd_path": output_usd_path,
                "in_place": in_place,
                "required_prim_paths": required_prim_paths,
            }
        )
        self.applied_scene = json.loads(Path(json_file_path).read_text())
        if self.behavior in {"false", "raise"}:
            Path(usd_file_path).write_bytes(b"partially-updated-usd")
        if self.behavior == "raise":
            raise RuntimeError("USD update failed")
        return self.behavior != "false"


class FakePositionUpdaterFactory:
    def __init__(self, updater):
        self.updater = updater
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.updater


class FakeSceneOptimizer:
    def __init__(self, changes):
        self.changes = iter(changes)
        self.calls = []

    def optimize_from_execution_report(self, report):
        self.calls.append(deepcopy(report))
        return next(self.changes)


class FakeSceneOptimizerFactory:
    def __init__(self, optimizers):
        self.optimizers = iter(optimizers)
        self.calls = []

    def __call__(self, artifacts):
        self.calls.append(artifacts)
        return next(self.optimizers)


class ParameterPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = CapabilityRegistry.load_default(ROOT)

    def test_patch_model_forbids_extra_fields_and_empty_parameters(self):
        with self.assertRaises(ValidationError):
            ParameterPatch(
                step_id="step_001",
                parameters={"pre_offset_z": 0.15},
                source="controller.py",
            )
        with self.assertRaises(ValidationError):
            ParameterPatch(step_id="step_001", parameters={})

    def test_patch_is_revalidated_after_nested_mutation(self):
        patch = ParameterPatch(
            step_id="step_001",
            parameters={"pre_offset_z": 0.15},
        )
        patch.parameters.clear()

        with self.assertRaises(ValidationError):
            apply_parameter_patch(make_plan(), patch, self.registry)

    def test_patch_rejects_unknown_step_and_non_tunable_parameter(self):
        plan = make_plan()
        with self.assertRaisesRegex(ValueError, "unknown step"):
            apply_parameter_patch(
                plan,
                ParameterPatch(
                    step_id="step_999",
                    parameters={"pre_offset_z": 0.15},
                ),
                self.registry,
            )
        with self.assertRaisesRegex(ValueError, "unsupported parameters"):
            apply_parameter_patch(
                plan,
                ParameterPatch(
                    step_id="step_001",
                    parameters={"object": "other"},
                ),
                self.registry,
            )

    def test_patch_rejects_range_bool_and_nonfinite_values(self):
        invalid_values = [-1.0, True, math.nan, math.inf, -math.inf]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    apply_parameter_patch(
                        make_plan(),
                        ParameterPatch(
                            step_id="step_001",
                            parameters={"pre_offset_z": value},
                        ),
                        self.registry,
                    )

    def test_patch_changes_only_failed_step_parameters(self):
        plan = make_plan()
        before = plan.model_dump(mode="python")

        updated = apply_parameter_patch(
            plan,
            ParameterPatch(
                step_id="step_001",
                parameters={"pre_offset_z": 0.15},
            ),
            self.registry,
        )

        expected = deepcopy(before)
        expected["actions"][0]["parameters"]["pre_offset_z"] = 0.15
        self.assertEqual(updated.model_dump(mode="python"), expected)
        self.assertEqual(plan.model_dump(mode="python"), before)


class PlanParameterOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = CapabilityRegistry.load_default(ROOT)

    def test_returns_validated_failed_step_patch_with_allowlisted_prompt(self):
        client = FakeClient(
            '{"step_id":"step_001","parameters":{"pre_offset_z":0.15}}'
        )
        optimizer = PlanParameterOptimizer(
            client,
            self.registry,
            model="test-model",
        )

        patch = optimizer.propose(make_plan(), failed_report())

        self.assertEqual(
            patch,
            ParameterPatch(
                step_id="step_001",
                parameters={"pre_offset_z": 0.15},
            ),
        )
        call = client.calls[0]
        self.assertEqual(call["temperature"], 0)
        request = json.loads(call["messages"][1]["content"])
        self.assertEqual(
            set(request),
            {
                "failed_step",
                "verification",
                "current_parameters",
                "allowed_tunable_parameters",
            },
        )
        self.assertEqual(request["failed_step"], {"id": "step_001", "type": "pick"})
        self.assertEqual(request["verification"]["code"], "GRASP_NOT_ESTABLISHED")
        self.assertIn("pre_offset_z", request["allowed_tunable_parameters"])
        self.assertNotIn("source_text", call["messages"][1]["content"])

    def test_null_response_means_no_parameter_patch(self):
        optimizer = PlanParameterOptimizer(
            FakeClient("null"),
            self.registry,
            model="test-model",
        )

        self.assertIsNone(optimizer.propose(make_plan(), failed_report()))

    def test_rejects_wrong_step_malformed_or_non_object_responses(self):
        responses = [
            '{"step_id":"step_002","parameters":{"pre_place_z":0.2}}',
            "```json\n{}\n```",
            "ParameterPatch(step_id='step_001', parameters={})",
            "[]",
            "not json",
        ]
        for response in responses:
            with self.subTest(response=response):
                optimizer = PlanParameterOptimizer(
                    FakeClient(response),
                    self.registry,
                    model="test-model",
                )
                with self.assertRaises(ValueError):
                    optimizer.propose(make_plan(), failed_report())

    def test_rejects_non_tunable_range_and_nonfinite_responses(self):
        responses = [
            '{"step_id":"step_001","parameters":{"object":"other"}}',
            '{"step_id":"step_001","parameters":{"pre_offset_z":-1.0}}',
            '{"step_id":"step_001","parameters":{"pre_offset_z":true}}',
            '{"step_id":"step_001","parameters":{"pre_offset_z":NaN}}',
        ]
        for response in responses:
            with self.subTest(response=response):
                optimizer = PlanParameterOptimizer(
                    FakeClient(response),
                    self.registry,
                    model="test-model",
                )
                with self.assertRaises(ValueError):
                    optimizer.propose(make_plan(), failed_report())

    def test_requires_known_failed_step_and_matching_verification_record(self):
        optimizer = PlanParameterOptimizer(
            FakeClient("null"),
            self.registry,
            model="test-model",
        )
        unknown = failed_report("step_999")
        missing_record = failed_report()
        missing_record["steps"] = []

        with self.assertRaisesRegex(ValueError, "unknown failed step"):
            optimizer.propose(make_plan(), unknown)
        with self.assertRaisesRegex(ValueError, "verification record"):
            optimizer.propose(make_plan(), missing_record)

    def test_rejects_nonfailed_record_or_verification(self):
        reports = []
        successful_record = failed_report()
        successful_record["steps"][0]["success"] = True
        successful_record["steps"][0]["verification"]["success"] = True
        reports.append(successful_record)

        successful_verification = failed_report()
        successful_verification["steps"][0]["verification"]["success"] = True
        reports.append(successful_verification)

        for report in reports:
            with self.subTest(report=report):
                optimizer = PlanParameterOptimizer(
                    FakeClient("null"),
                    self.registry,
                    model="test-model",
                )
                with self.assertRaisesRegex(ValueError, "failed record"):
                    optimizer.propose(make_plan(), report)


class PlanOrchestratorParameterTests(unittest.TestCase):
    def setUp(self):
        self.registry = CapabilityRegistry.load_default(ROOT)
        self.validator = PlanValidator(self.registry)
        self.config_path = Path("config.yaml")

    def make_orchestrator(
        self,
        runner,
        optimizer,
        *,
        max_parameter_iterations=2,
        max_scene_iterations=0,
    ):
        return PlanOrchestrator(
            runner,
            optimizer,
            scene_optimizer_factory=None,
            registry=self.registry,
            validator=self.validator,
            scene_preflight=None,
            max_parameter_iterations=max_parameter_iterations,
            max_scene_iterations=max_scene_iterations,
        )

    def test_initial_invalid_plan_is_rejected_before_simulation(self):
        plan = make_plan().model_copy(deep=True)
        plan.actions = plan.actions[1:]
        runner = FakeSimulationRunner([success_report()])
        optimizer = FakeOptimizer([])

        with self.assertRaisesRegex(ValueError, "initial plan"):
            self.make_orchestrator(runner, optimizer).run(
                plan,
                self.config_path,
                FakeArtifacts(),
            )

        self.assertEqual(runner.calls, [])

    def test_success_stops_without_requesting_a_patch(self):
        runner = FakeSimulationRunner([success_report()])
        optimizer = FakeOptimizer([])

        result = self.make_orchestrator(runner, optimizer).run(
            make_plan(),
            self.config_path,
            FakeArtifacts(),
        )

        self.assertTrue(result["execution_success"])
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(optimizer.calls, [])

    def test_patch_refreshes_plan_and_validation_fingerprints_before_retry(self):
        runner = FakeSimulationRunner([failed_report(), success_report()])
        optimizer = FakeOptimizer(
            [
                ParameterPatch(
                    step_id="step_001",
                    parameters={"pre_offset_z": 0.15},
                )
            ]
        )
        artifacts = FakeArtifacts()

        result = self.make_orchestrator(
            runner,
            optimizer,
            max_parameter_iterations=1,
        ).run(make_plan(), self.config_path, artifacts)

        self.assertTrue(result["execution_success"])
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(len(artifacts.plans), 1)
        self.assertEqual(len(artifacts.json_writes), 1)
        path, validation = artifacts.json_writes[0]
        self.assertEqual(path, artifacts.validation_path)
        self.assertEqual(
            validation.plan_fingerprint,
            plan_fingerprint(artifacts.plans[0]),
        )
        self.assertEqual(
            validation.registry_fingerprint,
            registry_fingerprint(self.registry),
        )

    def test_wrong_step_patch_is_rejected(self):
        runner = FakeSimulationRunner([failed_report()])
        optimizer = FakeOptimizer(
            [
                ParameterPatch(
                    step_id="step_002",
                    parameters={"pre_place_z": 0.2},
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "failed step"):
            self.make_orchestrator(runner, optimizer).run(
                make_plan(),
                self.config_path,
                FakeArtifacts(),
            )

    def test_parameter_iteration_limit_is_strict(self):
        runner = FakeSimulationRunner([failed_report(), failed_report()])
        optimizer = FakeOptimizer(
            [
                ParameterPatch(
                    step_id="step_001",
                    parameters={"pre_offset_z": 0.15},
                ),
                ParameterPatch(
                    step_id="step_001",
                    parameters={"pre_offset_z": 0.16},
                ),
            ]
        )

        result = self.make_orchestrator(
            runner,
            optimizer,
            max_parameter_iterations=1,
        ).run(make_plan(), self.config_path, FakeArtifacts())

        self.assertFalse(result["execution_success"])
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(len(optimizer.calls), 1)

    def test_malformed_runner_reports_fail_closed(self):
        malformed_reports = [
            {},
            {"execution_success": False, "failed_step": None, "steps": []},
            {"execution_success": False, "failed_step": "step_001", "steps": []},
        ]
        for report in malformed_reports:
            with self.subTest(report=report):
                runner = FakeSimulationRunner([report])
                with self.assertRaises(ValueError):
                    self.make_orchestrator(runner, FakeOptimizer([])).run(
                        make_plan(),
                        self.config_path,
                        FakeArtifacts(),
                    )

    def test_failed_record_metadata_must_match_plan_step(self):
        mismatches = {
            "action": "place",
            "object_id": "different_object",
            "target_id": "plate",
        }
        for field, value in mismatches.items():
            with self.subTest(field=field):
                report = failed_report()
                report["steps"][0][field] = value
                runner = FakeSimulationRunner([report])

                with self.assertRaisesRegex(ValueError, "plan step"):
                    self.make_orchestrator(
                        runner,
                        FakeOptimizer([]),
                        max_parameter_iterations=0,
                    ).run(make_plan(), self.config_path, FakeArtifacts())

    def test_success_report_must_exactly_match_every_ordered_plan_step(self):
        plan = make_plan()
        complete = success_report(plan)
        cases = {
            "empty": {**complete, "steps": []},
            "partial": {**complete, "steps": complete["steps"][:1]},
            "extra": {
                **complete,
                "steps": [*complete["steps"], complete["steps"][0]],
            },
            "reordered": {
                **complete,
                "steps": list(reversed(complete["steps"])),
            },
            "duplicate": {
                **complete,
                "steps": [complete["steps"][0], complete["steps"][0]],
            },
        }
        for field, value in {
            "step_id": "step_999",
            "action": "shake",
            "object_id": "different_object",
            "target_id": "different_target",
            "controller_completed": False,
        }.items():
            report = deepcopy(complete)
            report["steps"][0][field] = value
            cases[f"mismatched_{field}"] = report

        for label, report in cases.items():
            with self.subTest(label=label):
                runner = FakeSimulationRunner([report])
                with self.assertRaisesRegex(ValueError, "successful execution report"):
                    self.make_orchestrator(
                        runner,
                        FakeOptimizer([]),
                        max_parameter_iterations=0,
                    ).run(plan, self.config_path, FakeArtifacts())


class PlanOrchestratorSceneTests(unittest.TestCase):
    def setUp(self):
        self.registry = CapabilityRegistry.load_default(ROOT)
        self.validator = PlanValidator(self.registry)
        self.config_path = Path("config.yaml")

    def make_orchestrator(
        self,
        runner,
        factory,
        *,
        preflight=None,
        max_scene_iterations=1,
    ):
        return PlanOrchestrator(
            runner,
            parameter_optimizer=None,
            scene_optimizer_factory=factory,
            registry=self.registry,
            validator=self.validator,
            scene_preflight=preflight,
            max_parameter_iterations=0,
            max_scene_iterations=max_scene_iterations,
        )

    def test_scene_iteration_limit_is_strict(self):
        runner = FakeSimulationRunner([failed_report(), failed_report()])
        optimizer = FakeSceneOptimizer([True, True])
        factory = FakeSceneOptimizerFactory([optimizer])

        result = self.make_orchestrator(
            runner,
            factory,
            max_scene_iterations=1,
        ).run(make_plan(), self.config_path, FakeArtifacts())

        self.assertFalse(result["execution_success"])
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(len(optimizer.calls), 1)

    def test_scene_noop_stops_without_rerunning(self):
        runner = FakeSimulationRunner([failed_report()])
        optimizer = FakeSceneOptimizer([False])

        result = self.make_orchestrator(
            runner,
            FakeSceneOptimizerFactory([optimizer]),
        ).run(make_plan(), self.config_path, FakeArtifacts())

        self.assertFalse(result["execution_success"])
        self.assertEqual(len(runner.calls), 1)

    def test_zero_scene_limit_does_not_construct_optimizer(self):
        runner = FakeSimulationRunner([failed_report()])
        factory = FakeSceneOptimizerFactory([])

        result = self.make_orchestrator(
            runner,
            factory,
            max_scene_iterations=0,
        ).run(make_plan(), self.config_path, FakeArtifacts())

        self.assertFalse(result["execution_success"])
        self.assertEqual(factory.calls, [])

    def test_scene_success_stops_at_first_successful_rerun(self):
        runner = FakeSimulationRunner([failed_report(), success_report()])
        optimizer = FakeSceneOptimizer([True, True])

        result = self.make_orchestrator(
            runner,
            FakeSceneOptimizerFactory([optimizer]),
            max_scene_iterations=2,
        ).run(make_plan(), self.config_path, FakeArtifacts())

        self.assertTrue(result["execution_success"])
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(len(optimizer.calls), 1)

    def test_failed_scene_preflight_is_persisted_and_stops_rerun(self):
        runner = FakeSimulationRunner([failed_report()])
        optimizer = FakeSceneOptimizer([True])
        artifacts = FakeArtifacts()
        calls = []

        def preflight(plan, run_artifacts):
            calls.append((plan, run_artifacts))
            return ScenePreflightReport(
                passed=False,
                issues=(
                    ScenePreflightIssue(
                        code="OBJECT_OUT_OF_REACH",
                        message="flask is unreachable",
                        object_id="solid_flask",
                    ),
                ),
            )

        result = self.make_orchestrator(
            runner,
            FakeSceneOptimizerFactory([optimizer]),
            preflight=preflight,
        ).run(make_plan(), self.config_path, artifacts)

        self.assertEqual(
            result["error_code"],
            "SCENE_PREFLIGHT_FAILED_AFTER_OPTIMIZATION",
        )
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(artifacts.json_writes[0][0], artifacts.scene_preflight_path)

    def test_scene_optimizer_state_is_created_per_run(self):
        runner = FakeSimulationRunner([failed_report(), failed_report()])
        first = FakeSceneOptimizer([False])
        second = FakeSceneOptimizer([False])
        factory = FakeSceneOptimizerFactory([first, second])
        orchestrator = self.make_orchestrator(runner, factory)

        orchestrator.run(make_plan(), self.config_path, FakeArtifacts())
        orchestrator.run(make_plan(), self.config_path, FakeArtifacts())

        self.assertEqual(len(factory.calls), 2)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)


class ContinuousOptimizerStructuredTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.scene_json_path = self.directory / "scene.json"
        self.scene_usd_path = self.directory / "scene.usd"
        self.scene_data = {
            "/World/Flask": {
                "prim_name": "Flask",
                "prim_type": "Xform",
                "position": [0.0, 0.0, 0.775],
                "bounding_box": {"size": [0.1, 0.1, 0.1]},
                "annotations": {"role": "source"},
            },
            "/World/Beaker": {
                "prim_name": "Beaker",
                "prim_type": "Xform",
                "position": [0.5, 0.0, 0.775],
                "bounding_box": {"size": [0.1, 0.1, 0.1]},
            },
            "/World/FumeHood": {
                "prim_name": "FumeHood",
                "prim_type": "Xform",
                "position": [10.0, 10.0, 0.0],
                "bounding_box": {"size": [5.0, 5.0, 5.0]},
            },
        }
        self.scene_json_path.write_text(json.dumps(self.scene_data))
        self.original_usd = b"original-usd"
        self.scene_usd_path.write_bytes(self.original_usd)
        self.layout_profile = {
            "surface_z": 0.775,
            "reachable_region": {
                "type": "ellipse",
                "center": [0.0, 0.0],
                "semi_axes": [2.0, 2.0],
                "rotation": 0.0,
            },
        }

    @property
    def candidate_json_path(self):
        return self.directory / "scene.candidate.json"

    @property
    def candidate_usd_path(self):
        return self.directory / "scene.candidate.usd"

    def make_optimizer(self, response, *, updater=None):
        updater = updater or FakePositionUpdater()
        factory = FakePositionUpdaterFactory(updater)
        client = FakeSceneClient(response)
        optimizer = ContinuousOptimizer(
            logs_dir=self.directory / "logs",
            scenes_dir=self.directory / "scenes",
            client=client,
            scene_json_path=self.scene_json_path,
            scene_usd_path=self.scene_usd_path,
            layout_profile=self.layout_profile,
            position_updater_factory=factory,
            model="test-model",
        )
        return optimizer, client, updater, factory

    def assert_official_files_unchanged(self):
        self.assertEqual(
            json.loads(self.scene_json_path.read_text()),
            self.scene_data,
        )
        self.assertEqual(self.scene_usd_path.read_bytes(), self.original_usd)
        self.assertFalse(self.candidate_json_path.exists())
        self.assertFalse(self.candidate_usd_path.exists())

    def test_constructor_loads_default_profile_and_excludes_environment(self):
        optimizer = ContinuousOptimizer(
            logs_dir=self.directory / "logs",
            scenes_dir=self.directory / "scenes",
            client=FakeSceneClient("null"),
            model="test-model",
        )

        self.assertAlmostEqual(optimizer.z_height, 0.775)
        self.assertAlmostEqual(optimizer.ellipse_constraint.center_x, 0.174972)
        self.assertAlmostEqual(optimizer.ellipse_constraint.semi_minor, 0.169138)
        self.assertIn("FumeHood", optimizer.excluded_objects)

    def test_constructor_requires_credentials_and_passes_only_explicit_base_url(self):
        target = "agent.scene.optimization.continuous_optimizater.OpenAI"
        with patch.dict(os.environ, {}, clear=True):
            with patch(target) as openai:
                with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                    ContinuousOptimizer(
                        logs_dir=self.directory / "logs",
                        scenes_dir=self.directory / "scenes",
                    )
                openai.assert_not_called()

            with patch(target) as openai:
                ContinuousOptimizer(
                    logs_dir=self.directory / "logs",
                    scenes_dir=self.directory / "scenes",
                    api_key="explicit-key",
                    model="test-model",
                )
                openai.assert_called_once_with(api_key="explicit-key")

            with patch(target) as openai:
                ContinuousOptimizer(
                    logs_dir=self.directory / "logs",
                    scenes_dir=self.directory / "scenes",
                    api_key="explicit-key",
                    base_url="https://example.invalid/v1",
                    model="test-model",
                )
                openai.assert_called_once_with(
                    api_key="explicit-key",
                    base_url="https://example.invalid/v1",
                )

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "environment-key",
                "ACTION_AGENT_MODEL": "test-model",
            },
            clear=True,
        ):
            with patch(target) as openai:
                ContinuousOptimizer(
                    logs_dir=self.directory / "logs",
                    scenes_dir=self.directory / "scenes",
                )
                openai.assert_called_once_with(api_key="environment-key")

    def test_valid_failed_step_positions_update_json_and_candidate_usd(self):
        optimizer, client, updater, factory = self.make_optimizer(
            '{"positions":{"/World/Flask":[-0.5,0.0,0.775]}}'
        )

        changed = optimizer.optimize_from_execution_report(failed_report())

        self.assertTrue(changed)
        updated = json.loads(self.scene_json_path.read_text())
        self.assertEqual(updated["/World/Flask"]["position"], [-0.5, 0.0, 0.775])
        self.assertEqual(updated["/World/Flask"]["annotations"], {"role": "source"})
        self.assertEqual(updated["/World/FumeHood"], self.scene_data["/World/FumeHood"])
        self.assertEqual(len(updater.calls), 1)
        self.assertEqual(updater.calls[0]["json_file_path"], self.candidate_json_path)
        self.assertEqual(updater.calls[0]["usd_file_path"], self.candidate_usd_path)
        self.assertEqual(
            updater.calls[0]["required_prim_paths"],
            {"/World/Flask"},
        )
        self.assertEqual(len(factory.calls), 1)
        self.assertFalse(self.candidate_json_path.exists())
        self.assertFalse(self.candidate_usd_path.exists())

        call = client.calls[0]
        self.assertEqual(call["temperature"], 0)
        request = json.loads(call["messages"][1]["content"])
        self.assertEqual(
            set(request),
            {"failed_step", "verification", "current_positions"},
        )
        self.assertEqual(request["failed_step"], "step_001")
        self.assertEqual(
            request["verification"],
            {"code": "GRASP_NOT_ESTABLISHED", "measurements": {"lift": 0.02}},
        )
        self.assertNotIn("/World/FumeHood", request["current_positions"])

    def test_invalid_noop_and_excluded_proposals_write_nothing(self):
        responses = [
            "not json",
            '```json\n{"positions":{"/World/Flask":[-0.5,0.0,0.775]}}\n```',
            "[]",
            '{"positions":{}}',
            '{"positions":{"/World/Unknown":[-0.5,0.0,0.775]}}',
            '{"positions":{"/World/FumeHood":[0.0,0.0,0.775]}}',
            '{"positions":{"/World/Flask":[0.0,0.0,0.775]}}',
            '{"positions":{"/World/Flask":[true,0.0,0.775]}}',
            '{"positions":{"/World/Flask":[NaN,0.0,0.775]}}',
            '{"positions":{"/World/Flask":[Infinity,0.0,0.775]}}',
            '{"positions":{"/World/Flask":[-0.5,0.0]}}',
            '{"positions":{"/World/Flask":{"x":-0.5,"y":0.0,"z":0.775}}}',
            '{"positions":{"/World/Flask":[3.0,0.0,0.775]}}',
            '{"positions":{"/World/Flask":[-0.5,0.0,0.5]}}',
            '{"positions":{"/World/Flask":[0.5,0.0,0.775]}}',
        ]
        for response in responses:
            with self.subTest(response=response):
                self.scene_json_path.write_text(json.dumps(self.scene_data))
                self.scene_usd_path.write_bytes(self.original_usd)
                optimizer, _, updater, factory = self.make_optimizer(response)

                self.assertFalse(
                    optimizer.optimize_from_execution_report(failed_report())
                )

                self.assert_official_files_unchanged()
                self.assertEqual(updater.calls, [])
                self.assertEqual(factory.calls, [])

    def test_missing_or_mismatched_failed_record_skips_llm(self):
        reports = [
            {"failed_step": None, "steps": []},
            {"failed_step": "step_001", "steps": []},
            {
                "failed_step": "step_001",
                "steps": [{"step_id": "step_002", "verification": {}}],
            },
        ]
        for report in reports:
            with self.subTest(report=report):
                optimizer, client, updater, _ = self.make_optimizer(
                    '{"positions":{"/World/Flask":[-0.5,0.0,0.775]}}'
                )
                self.assertFalse(optimizer.optimize_from_execution_report(report))
                self.assertEqual(client.calls, [])
                self.assertEqual(updater.calls, [])
                self.assert_official_files_unchanged()

    def test_nonfailed_record_or_verification_skips_llm(self):
        reports = []
        successful_record = failed_report()
        successful_record["steps"][0]["success"] = True
        successful_record["steps"][0]["verification"]["success"] = True
        reports.append(successful_record)

        successful_verification = failed_report()
        successful_verification["steps"][0]["verification"]["success"] = True
        reports.append(successful_verification)

        for report in reports:
            with self.subTest(report=report):
                optimizer, client, updater, _ = self.make_optimizer(
                    '{"positions":{"/World/Flask":[-0.5,0.0,0.775]}}'
                )
                self.assertFalse(optimizer.optimize_from_execution_report(report))
                self.assertEqual(client.calls, [])
                self.assertEqual(updater.calls, [])
                self.assert_official_files_unchanged()

    def test_updater_failure_rolls_back_and_removes_candidates(self):
        for behavior in ("false", "raise"):
            with self.subTest(behavior=behavior):
                self.scene_json_path.write_text(json.dumps(self.scene_data))
                self.scene_usd_path.write_bytes(self.original_usd)
                updater = FakePositionUpdater(behavior)
                optimizer, _, _, _ = self.make_optimizer(
                    '{"positions":{"/World/Flask":[-0.5,0.0,0.775]}}',
                    updater=updater,
                )

                self.assertFalse(
                    optimizer.optimize_from_execution_report(failed_report())
                )

                self.assertEqual(len(updater.calls), 1)
                self.assertEqual(
                    updater.calls[0]["required_prim_paths"],
                    {"/World/Flask"},
                )
                self.assert_official_files_unchanged()

    def test_noop_and_updater_failure_do_not_touch_official_paths(self):
        original_write_text = Path.write_text
        original_write_bytes = Path.write_bytes
        original_replace = Path.replace

        write_text_paths = []
        write_bytes_paths = []
        replace_targets = []

        def track_write_text(path, *args, **kwargs):
            write_text_paths.append(path)
            return original_write_text(path, *args, **kwargs)

        def track_write_bytes(path, *args, **kwargs):
            write_bytes_paths.append(path)
            return original_write_bytes(path, *args, **kwargs)

        def track_replace(path, target):
            replace_targets.append(Path(target))
            return original_replace(path, target)

        noop, _, _, _ = self.make_optimizer(
            '{"positions":{"/World/Flask":[0.0,0.0,0.775]}}'
        )
        with patch.object(Path, "write_text", autospec=True) as write_text:
            with patch.object(Path, "write_bytes", autospec=True) as write_bytes:
                with patch.object(Path, "replace", autospec=True) as replace:
                    write_text.side_effect = track_write_text
                    write_bytes.side_effect = track_write_bytes
                    replace.side_effect = track_replace
                    self.assertFalse(noop.optimize_from_execution_report(failed_report()))
        self.assertEqual(write_text_paths, [])
        self.assertEqual(write_bytes_paths, [])
        self.assertEqual(replace_targets, [])

        updater = FakePositionUpdater("false")
        optimizer, _, _, _ = self.make_optimizer(
            '{"positions":{"/World/Flask":[-0.5,0.0,0.775]}}',
            updater=updater,
        )
        with patch.object(Path, "write_bytes", autospec=True) as write_bytes:
            with patch.object(Path, "replace", autospec=True) as replace:
                write_bytes.side_effect = track_write_bytes
                replace.side_effect = track_replace
                self.assertFalse(
                    optimizer.optimize_from_execution_report(failed_report())
                )

        self.assertNotIn(self.scene_json_path, write_bytes_paths)
        self.assertNotIn(self.scene_usd_path, write_bytes_paths)
        self.assertNotIn(self.scene_json_path, replace_targets)
        self.assertNotIn(self.scene_usd_path, replace_targets)
        self.assert_official_files_unchanged()

    def test_replace_failure_restores_both_official_files(self):
        original_replace = Path.replace
        for fail_on_call in (1, 2):
            with self.subTest(fail_on_call=fail_on_call):
                self.scene_json_path.write_text(json.dumps(self.scene_data))
                self.scene_usd_path.write_bytes(self.original_usd)
                optimizer, _, updater, _ = self.make_optimizer(
                    '{"positions":{"/World/Flask":[-0.5,0.0,0.775]}}'
                )
                replace_calls = 0

                def replace_with_failure(path, target):
                    nonlocal replace_calls
                    replace_calls += 1
                    if replace_calls == fail_on_call:
                        raise OSError("replace failed")
                    return original_replace(path, target)

                with patch.object(Path, "replace", autospec=True) as replace:
                    replace.side_effect = replace_with_failure
                    self.assertFalse(
                        optimizer.optimize_from_execution_report(failed_report())
                    )

                self.assertEqual(len(updater.calls), 1)
                self.assert_official_files_unchanged()


class PositionUpdaterRequiredPathsTests(unittest.TestCase):
    def load_module(self, stage):
        class FakeXformOp:
            PrecisionDouble = "double"
            PrecisionFloat = "float"
            TypeTranslate = "translate"

        class AddedTranslateOp:
            def __init__(self, result):
                self.result = result

            def Set(self, value, time):
                return self.result

        class FakeXformable:
            def __init__(self, prim):
                self.prim = prim

            def GetOrderedXformOps(self):
                return self.prim.existing_ops

            def AddTranslateOp(self, precision):
                return AddedTranslateOp(self.prim.add_set_result)

        class FakeTimeCode:
            @staticmethod
            def Default():
                return object()

        class FakeStageType:
            LoadAll = object()

            @staticmethod
            def Open(path, load=None):
                return stage

        pxr = types.ModuleType("pxr")
        pxr.Usd = SimpleNamespace(
            Prim=object,
            TimeCode=FakeTimeCode,
            Stage=FakeStageType,
        )
        pxr.UsdGeom = SimpleNamespace(
            Xformable=FakeXformable,
            XformOp=FakeXformOp,
        )
        pxr.Gf = SimpleNamespace(
            Vec3f=lambda value: tuple(value),
            Vec3d=lambda value: tuple(value),
        )
        pxr.Sdf = SimpleNamespace(
            Path=lambda value: value,
            ValueTypeNames=SimpleNamespace(Float3="float3"),
        )

        module_name = "_position_updater_required_paths_under_test"
        spec = importlib.util.spec_from_file_location(
            module_name,
            ROOT / "agent/scene/optimization/position_updater.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            with patch.dict(sys.modules, {"pxr": pxr}):
                spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
        return module

    def test_update_prim_position_rejects_false_set_results(self):
        class FakeStage:
            pass

        module = self.load_module(FakeStage())

        class ExistingTranslateOp:
            def __init__(self, result):
                self.result = result

            def GetOpType(self):
                return "translate"

            def GetOpName(self):
                return "xformOp:translate"

            def Set(self, value, time):
                return self.result

        class FakePrim:
            def __init__(self, *, existing_ops=(), add_set_result=True):
                self.existing_ops = list(existing_ops)
                self.add_set_result = add_set_result

            def IsValid(self):
                return True

            def IsA(self, prim_type):
                return True

            def GetAttribute(self, name):
                return None

            def GetPath(self):
                return "/World/Flask"

        updater = module.PositionUpdater.__new__(module.PositionUpdater)
        for set_result in (False, None):
            with self.subTest(existing_set_result=set_result):
                self.assertFalse(
                    updater.update_prim_position(
                        FakePrim(
                            existing_ops=[ExistingTranslateOp(set_result)]
                        ),
                        [0.0, 0.0, 0.775],
                    )
                )
            with self.subTest(added_set_result=set_result):
                self.assertFalse(
                    updater.update_prim_position(
                        FakePrim(add_set_result=set_result),
                        [0.0, 0.0, 0.775],
                    )
                )
        self.assertTrue(
            updater.update_prim_position(
                FakePrim(existing_ops=[ExistingTranslateOp(True)]),
                [0.0, 0.0, 0.775],
            )
        )

    def test_required_paths_fail_on_missing_skipped_or_update_error(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        json_path = directory / "candidate.json"
        usd_path = directory / "candidate.usd"
        usd_path.write_bytes(b"candidate-usd")
        json_path.write_text(
            json.dumps(
                {
                    "/World/Good": {"position": [0.0, 0.0, 0.775]},
                    "/World/Bad": {"position": [0.1, 0.0, 0.775]},
                    "/World/Skipped": {"prim_name": "Skipped"},
                }
            )
        )

        class FakePrim:
            def __init__(self, path):
                self.path = path

            def IsValid(self):
                return True

        class FakeLayer:
            def Export(self, path):
                return True

        class FakeStage:
            def GetPrimAtPath(self, path):
                if path in {"/World/Good", "/World/Bad", "/World/Skipped"}:
                    return FakePrim(path)
                return None

            def GetRootLayer(self):
                return FakeLayer()

        module = self.load_module(FakeStage())
        updater = module.PositionUpdater(scenes_dir=str(directory))
        updater.update_prim_position = (
            lambda prim, position, time: prim.path != "/World/Bad"
        )

        self.assertTrue(
            updater.apply_positions_to_usd(
                json_path,
                usd_path,
                required_prim_paths={"/World/Good"},
            )
        )
        for required in (
            {"/World/Missing"},
            {"/World/Skipped"},
            {"/World/Bad"},
        ):
            with self.subTest(required=required):
                self.assertFalse(
                    updater.apply_positions_to_usd(
                        json_path,
                        usd_path,
                        required_prim_paths=required,
                    )
                )

        self.assertTrue(updater.apply_positions_to_usd(json_path, usd_path))

    def test_apply_fails_when_root_layer_export_returns_false(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        json_path = directory / "candidate.json"
        usd_path = directory / "candidate.usd"
        json_path.write_text(
            json.dumps({"/World/Flask": {"position": [0.0, 0.0, 0.775]}})
        )
        usd_path.write_bytes(b"candidate-usd")

        class FakePrim:
            def IsValid(self):
                return True

        for export_result in (False, None):
            with self.subTest(export_result=export_result):
                class FakeLayer:
                    def Export(self, path):
                        return export_result

                class FakeStage:
                    def GetPrimAtPath(self, path):
                        return FakePrim()

                    def GetRootLayer(self):
                        return FakeLayer()

                module = self.load_module(FakeStage())
                updater = module.PositionUpdater(scenes_dir=str(directory))
                updater.update_prim_position = (
                    lambda prim, position, time: True
                )

                self.assertFalse(
                    updater.apply_positions_to_usd(
                        json_path,
                        usd_path,
                        required_prim_paths={"/World/Flask"},
                    )
                )


if __name__ == "__main__":
    unittest.main()
