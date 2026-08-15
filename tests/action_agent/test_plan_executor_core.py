import inspect
import unittest

from pydantic import ValidationError

import agent.action.plan_execution as plan_execution
from agent.action.plan_execution.executor import SequentialPlanExecutor
from agent.action.plan_execution.interfaces import ActionAdapter
from agent.action.plan_execution.models import (
    ExecutionReport,
    StepExecutionRecord,
    VerificationRequest,
    VerificationResult,
)
from agent.planning.models import CoverageLevel, SemanticAnnotation, AnnotationStatus
from tests.action_agent.test_plan_models import make_plan


class FakeAdapter:
    def __init__(self, label="adapter", events=None):
        self.label = label
        self.events = events if events is not None else []
        self.steps = 0
        self.resets = 0
        self.forward_calls = 0
        self.prepare_step_ids = []
        self.prepare_start_counts = []
        self.prepare_contexts = []
        self.prepare_frames = []

    def prepare(self, step, context):
        self.prepare_start_counts.append(self.steps)
        self.prepare_step_ids.append(step.id)
        self.prepare_contexts.append(context)
        self.prepare_frames.append(context.frame)
        self.steps = 0
        self.events.append(("prepare", self.label, step.id))

    def step(self, state):
        self.steps += 1
        self.forward_calls += 1
        self.events.append(("step", self.label, self.steps))
        return f"{self.label}-action-{self.steps}"

    def is_done(self):
        return self.steps >= 1

    def reset(self):
        self.resets += 1
        self.steps = 0
        self.events.append(("reset", self.label, self.resets))


class NeverDoneAdapter(FakeAdapter):
    def is_done(self):
        return False


class TwoFrameAdapter(FakeAdapter):
    def is_done(self):
        return self.steps >= 2


class FailingLifecycleAdapter(FakeAdapter):
    def __init__(self, failure=None, *, reset_failure=False, never_done=False):
        super().__init__("pick")
        self.failure = failure
        self.reset_failure = reset_failure
        self.never_done = never_done

    def prepare(self, step, context):
        super().prepare(step, context)
        if self.failure == "prepare":
            raise RuntimeError("secret prepare details")

    def step(self, state):
        if self.failure == "step":
            self.forward_calls += 1
            raise RuntimeError("secret step details")
        return super().step(state)

    def is_done(self):
        if self.failure == "is_done":
            raise RuntimeError("secret status details")
        if self.never_done:
            return False
        return super().is_done()

    def reset(self):
        self.resets += 1
        self.steps = 0
        if self.reset_failure:
            raise ValueError("secret reset details")


class MutatingStepAdapter(FakeAdapter):
    def prepare(self, step, context):
        super().prepare(step, context)
        self.received_step = step
        step.id = "step_999"
        step.type = "place"
        step.object = "mutated_object"


class ExplodingState(dict):
    def items(self):
        raise RuntimeError("secret snapshot details")


class DeepcopyBomb:
    calls = 0
    fail_on = 3

    def __deepcopy__(self, memo):
        type(self).calls += 1
        if type(self).calls == type(self).fail_on:
            raise RuntimeError("secret request details")
        clone = type(self)()
        memo[id(self)] = clone
        return clone


class SequenceVerifier:
    def __init__(self, results, label="verifier", events=None):
        self.results = iter(results)
        self.label = label
        self.events = events if events is not None else []
        self.requests = []

    def verify(self, request):
        self.requests.append(request)
        self.events.append(("verify", self.label, request.step.id))
        return next(self.results)


class SnapshotVerifier(SequenceVerifier):
    def __init__(self):
        super().__init__([])

    def verify(self, request):
        self.requests.append(request)
        return VerificationResult(
            success=True,
            measurements={"observed": request.post_state["nested"]},
        )


class RaisingVerifier(SequenceVerifier):
    def __init__(self):
        super().__init__([])

    def verify(self, request):
        self.requests.append(request)
        raise RuntimeError("secret verifier details")


class MutatingStepVerifier(SequenceVerifier):
    def __init__(self, result):
        super().__init__([])
        self.result = result

    def verify(self, request):
        self.requests.append(request)
        request.step.id = "step_998"
        return self.result


def single_step_plan():
    plan = make_plan()
    return plan.model_copy(update={"actions": [plan.actions[0]]})


def run_to_completion(runner):
    while not runner.done:
        runner.step({"frame": runner.frame})


class PlanExecutionModelContractTests(unittest.TestCase):
    def test_result_and_report_defaults_serialize_exactly(self):
        result = VerificationResult(success=True)
        self.assertEqual(result.model_dump(), {
            "success": True,
            "code": "OK",
            "message": "",
            "measurements": {},
            "verification_level": "state_observed",
        })
        self.assertEqual(ExecutionReport().model_dump(), {
            "execution_success": False,
            "failed_step": None,
            "steps": [],
        })
        with self.assertRaises(ValidationError):
            VerificationResult(success=True, unexpected=True)

    def test_request_and_step_record_match_public_contract(self):
        step = object()
        request = VerificationRequest(
            step=step,
            pre_state={"frame": 0},
            post_state={"frame": 1},
            state_history=[{"frame": 1}],
            episode_initial_state={"frame": 0},
        )
        self.assertIs(request.step, step)
        self.assertTrue(VerificationRequest.model_config["arbitrary_types_allowed"])
        with self.assertRaises(ValidationError):
            VerificationRequest(
                step=step,
                pre_state={},
                post_state={},
                state_history=[],
                episode_initial_state={},
                unexpected=True,
            )

        record = StepExecutionRecord(
            step_id="step_001",
            action="pick",
            object_id="solid_flask",
            target_id=None,
            coverage_level="supported",
            adapter="FakeAdapter",
            verifier="SequenceVerifier",
            attempt_count=1,
            success=True,
            start_frame=1,
            end_frame=1,
            controller_completed=True,
            semantic_requirements=[],
            verification=VerificationResult(success=True),
        )
        dumped = record.model_dump()
        self.assertEqual(dumped["action"], "pick")
        self.assertNotIn("action_type", dumped)
        self.assertEqual(dumped["object_id"], "solid_flask")

    def test_executor_and_adapter_signatures_match_public_contract(self):
        parameters = inspect.signature(
            SequentialPlanExecutor.__init__,
        ).parameters.values()
        keyword_only = [
            parameter.name
            for parameter in parameters
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        ]
        self.assertEqual(keyword_only, [
            "coverage_by_step",
            "action_timeouts",
            "max_retries",
        ])
        self.assertEqual(
            list(inspect.signature(ActionAdapter.prepare).parameters),
            ["self", "step", "context"],
        )

    def test_package_exports_only_public_execution_contract(self):
        self.assertEqual(plan_execution.__all__, [
            "ExecutionReport",
            "SequentialPlanExecutor",
            "VerificationRequest",
            "VerificationResult",
        ])

    def test_numpy_bool_success_is_normalized_without_weakening_strictness(self):
        import numpy as np

        result = VerificationResult(success=np.bool_(True))

        self.assertIs(result.success, True)
        with self.assertRaises(ValidationError):
            VerificationResult(success=np.int64(1))

    def test_numpy_measurement_scalars_are_recursively_normalized(self):
        import numpy as np

        result = VerificationResult(
            success=True,
            measurements={
                "integer": np.int64(7),
                "nested": [np.float32(1.5), {"flag": np.bool_(True)}],
                "tuple": (np.int32(3),),
            },
        )

        self.assertIs(type(result.measurements["integer"]), int)
        self.assertIs(type(result.measurements["nested"][0]), float)
        self.assertIs(type(result.measurements["nested"][1]["flag"]), bool)
        self.assertIs(type(result.measurements["tuple"][0]), int)
        self.assertIn('"integer":7', result.model_dump_json())

    def test_finalized_models_reject_direct_and_nested_assignment(self):
        result = VerificationResult(success=True)
        record = StepExecutionRecord(
            step_id="step_001",
            action="pick",
            coverage_level="supported",
            adapter="FakeAdapter",
            verifier="SequenceVerifier",
            attempt_count=1,
            success=True,
            start_frame=1,
            end_frame=1,
            controller_completed=True,
            verification=result,
        )
        report = ExecutionReport(execution_success=True, steps=[record])

        with self.assertRaises(ValidationError):
            result.success = False
        with self.assertRaises(ValidationError):
            record.success = False
        with self.assertRaises(ValidationError):
            report.steps[0].verification.success = False
        with self.assertRaises(ValidationError):
            report.execution_success = False

        self.assertTrue(result.success)
        self.assertTrue(record.success)
        self.assertTrue(report.execution_success)
        self.assertTrue(report.steps[0].verification.success)

    def test_record_and_report_consistency_is_validated_on_assignment(self):
        failure = VerificationResult(success=False, code="FAIL")
        with self.assertRaises(ValidationError):
            StepExecutionRecord(
                step_id="step_001",
                action="pick",
                coverage_level="supported",
                adapter="FakeAdapter",
                verifier="SequenceVerifier",
                attempt_count=1,
                success=True,
                start_frame=1,
                end_frame=1,
                controller_completed=True,
                verification=failure,
            )

        failed_record = StepExecutionRecord(
            step_id="step_001",
            action="pick",
            coverage_level="supported",
            adapter="FakeAdapter",
            verifier="SequenceVerifier",
            attempt_count=1,
            success=False,
            start_frame=1,
            end_frame=1,
            controller_completed=True,
            verification=failure,
        )
        with self.assertRaises(ValidationError):
            ExecutionReport(
                execution_success=True,
                failed_step="step_001",
                steps=[failed_record],
            )

        success_record = StepExecutionRecord(
            step_id="step_001",
            action="pick",
            coverage_level="supported",
            adapter="FakeAdapter",
            verifier="SequenceVerifier",
            attempt_count=1,
            success=True,
            start_frame=1,
            end_frame=1,
            controller_completed=True,
            verification=VerificationResult(success=True),
        )
        with self.assertRaises(ValidationError):
            success_record.end_frame = 0


class PlanExecutorCoreTests(unittest.TestCase):
    def assert_terminal_error(self, runner, adapter, code, *, resets=1):
        self.assertTrue(runner.done)
        self.assertFalse(runner.success)
        self.assertFalse(runner.report.execution_success)
        self.assertEqual(len(runner.report.steps), 1)
        self.assertEqual(runner.report.steps[0].verification.code, code)
        self.assertEqual(adapter.resets, resets)
        serialized = runner.report.model_dump_json()
        self.assertNotIn("secret", serialized)

        self.assertIsNone(runner.step({"ignored": True}))
        self.assertEqual(len(runner.report.steps), 1)
        self.assertEqual(adapter.resets, resets)

    def test_all_steps_succeed_in_plan_order_and_report_metadata(self):
        plan = make_plan()
        plan.semantic_annotations.append(SemanticAnnotation(
            source_text="keep the flask upright",
            status=AnnotationStatus.APPROXIMATED,
            reason="orientation tolerance is approximate",
            step_ids=["step_001"],
        ))
        events = []
        pick_adapter = FakeAdapter("pick", events)
        place_adapter = FakeAdapter("place", events)
        runner = SequentialPlanExecutor(
            plan,
            {"pick": pick_adapter, "place": place_adapter},
            {
                "pick": SequenceVerifier([VerificationResult(success=True)], "pick", events),
                "place": SequenceVerifier([VerificationResult(success=True)], "place", events),
            },
            coverage_by_step={
                "step_001": CoverageLevel.DEGRADED,
                "step_002": "supported",
            },
        )

        run_to_completion(runner)

        self.assertTrue(runner.done)
        self.assertTrue(runner.success)
        self.assertTrue(runner.report.execution_success)
        self.assertEqual(
            [record.step_id for record in runner.report.steps],
            ["step_001", "step_002"],
        )
        self.assertEqual(
            [event[1] for event in events if event[0] == "step"],
            ["pick", "place"],
        )
        first = runner.report.steps[0]
        self.assertEqual(first.coverage_level, "degraded")
        self.assertEqual(first.semantic_requirements, ["keep the flask upright"])
        self.assertEqual(first.adapter, "FakeAdapter")
        self.assertEqual(first.verifier, "SequenceVerifier")
        self.assertEqual(first.action, "pick")
        self.assertEqual(first.object_id, "solid_flask")
        self.assertIsNone(first.target_id)
        self.assertTrue(first.success)
        self.assertEqual((first.start_frame, first.end_frame), (1, 1))
        second = runner.report.steps[1]
        self.assertEqual(second.target_id, "plate")
        self.assertEqual((second.start_frame, second.end_frame), (2, 2))

    def test_failed_verification_retries_only_current_step_once(self):
        plan = make_plan()
        pick_adapter = FakeAdapter("pick")
        place_adapter = FakeAdapter("place")
        runner = SequentialPlanExecutor(
            plan,
            {"pick": pick_adapter, "place": place_adapter},
            {
                "pick": SequenceVerifier([
                    VerificationResult(success=False, code="GRASP_NOT_ESTABLISHED"),
                    VerificationResult(success=True),
                ]),
                "place": SequenceVerifier([VerificationResult(success=True)]),
            },
            max_retries=1,
        )

        runner.step({"frame": 0})

        self.assertFalse(runner.done)
        self.assertEqual(runner.index, 0)
        self.assertEqual(runner.report.steps, [])
        self.assertEqual(pick_adapter.resets, 1)
        self.assertEqual(pick_adapter.prepare_step_ids, ["step_001", "step_001"])
        self.assertEqual(place_adapter.forward_calls, 0)

        run_to_completion(runner)

        self.assertTrue(runner.success)
        self.assertEqual(pick_adapter.resets, 2)
        self.assertEqual(runner.report.steps[0].attempt_count, 2)
        self.assertEqual(
            (runner.report.steps[0].start_frame, runner.report.steps[0].end_frame),
            (1, 2),
        )
        self.assertEqual(
            [record.step_id for record in runner.report.steps],
            ["step_001", "step_002"],
        )

    def test_repeated_action_type_reuses_adapter_only_after_reset(self):
        original = make_plan()
        repeated = original.actions[0].model_copy(update={"id": "step_003"})
        plan = original.model_copy(update={"actions": [*original.actions, repeated]})
        pick_adapter = FakeAdapter("pick")
        runner = SequentialPlanExecutor(
            plan,
            {"pick": pick_adapter, "place": FakeAdapter("place")},
            {
                "pick": SequenceVerifier([
                    VerificationResult(success=True),
                    VerificationResult(success=True),
                ]),
                "place": SequenceVerifier([VerificationResult(success=True)]),
            },
        )

        run_to_completion(runner)

        self.assertTrue(runner.success)
        self.assertEqual(pick_adapter.forward_calls, 2)
        self.assertEqual(pick_adapter.resets, 2)
        self.assertEqual(pick_adapter.prepare_step_ids, ["step_001", "step_003"])
        self.assertEqual(pick_adapter.prepare_start_counts, [0, 0])

    def test_second_failed_verification_stops_before_later_steps(self):
        plan = make_plan()
        pick_adapter = FakeAdapter("pick")
        place_adapter = FakeAdapter("place")
        runner = SequentialPlanExecutor(
            plan,
            {"pick": pick_adapter, "place": place_adapter},
            {
                "pick": SequenceVerifier([
                    VerificationResult(success=False, code="FAIL"),
                    VerificationResult(success=False, code="FAIL"),
                ]),
                "place": SequenceVerifier([VerificationResult(success=True)]),
            },
            max_retries=1,
        )

        run_to_completion(runner)

        self.assertTrue(runner.done)
        self.assertFalse(runner.success)
        self.assertFalse(runner.report.execution_success)
        self.assertEqual(runner.report.failed_step, "step_001")
        self.assertEqual([record.step_id for record in runner.report.steps], ["step_001"])
        self.assertEqual(runner.report.steps[0].attempt_count, 2)
        self.assertFalse(runner.report.steps[0].success)
        self.assertEqual(
            (runner.report.steps[0].start_frame, runner.report.steps[0].end_frame),
            (1, 2),
        )
        self.assertEqual(place_adapter.forward_calls, 0)
        self.assertEqual(pick_adapter.resets, 2)

    def test_action_timeout_is_structured_and_skips_verifier(self):
        adapter = NeverDoneAdapter("pick")
        verifier = SequenceVerifier([])
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": verifier},
            action_timeouts={"pick": 2},
            max_retries=0,
        )

        run_to_completion(runner)

        self.assertFalse(runner.success)
        self.assertEqual(adapter.forward_calls, 2)
        self.assertEqual(adapter.resets, 1)
        self.assertEqual(verifier.requests, [])
        record = runner.report.steps[0]
        self.assertFalse(record.controller_completed)
        self.assertEqual(record.verification.code, "ACTION_TIMEOUT")
        self.assertEqual(record.verification.verification_level, "controller_state")
        self.assertEqual(record.verification.measurements, {"elapsed_frames": 2})
        self.assertEqual((record.start_frame, record.end_frame), (1, 2))

    def test_empty_plan_completes_successfully_without_components(self):
        plan = make_plan().model_copy(update={"actions": []})

        runner = SequentialPlanExecutor(plan, {}, {})

        self.assertTrue(runner.done)
        self.assertTrue(runner.success)
        self.assertTrue(runner.report.execution_success)
        self.assertEqual(runner.report.steps, [])
        self.assertIsNone(runner.step({"unused": True}))

    def test_verifier_snapshots_filter_cameras_include_names_and_are_isolated(self):
        verifier = SnapshotVerifier()
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": FakeAdapter("pick")},
            {"pick": verifier},
        )
        state = {
            "nested": {"value": 1},
            "camera_data": {"rgb": bytearray(b"large")},
            "camera_display": object(),
        }

        runner.step(state)
        state["nested"]["value"] = 999
        state["camera_data"]["rgb"].clear()

        request = verifier.requests[0]
        expected_names = {
            obj.id: obj.instance_name for obj in single_step_plan().scene.objects
        }
        snapshots = [
            request.pre_state,
            request.post_state,
            *request.state_history,
            request.episode_initial_state,
        ]
        for snapshot in snapshots:
            self.assertNotIn("camera_data", snapshot)
            self.assertNotIn("camera_display", snapshot)
            self.assertEqual(snapshot["plan_instance_names"], expected_names)
            self.assertEqual(snapshot["nested"]["value"], 1)
        self.assertIsNot(request.pre_state["nested"], request.post_state["nested"])
        self.assertEqual(
            runner.report.steps[0].verification.measurements["observed"]["value"],
            1,
        )

    def test_verifier_mapping_result_is_coerced_to_strict_model(self):
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": FakeAdapter("pick")},
            {"pick": SequenceVerifier([{"success": True, "code": "MAPPING_OK"}])},
        )

        run_to_completion(runner)

        result = runner.report.steps[0].verification
        self.assertIsInstance(result, VerificationResult)
        self.assertEqual(result.code, "MAPPING_OK")

    def test_constructor_rejects_missing_components_and_invalid_limits(self):
        plan = single_step_plan()
        with self.assertRaisesRegex(ValueError, "missing adapters.*pick"):
            SequentialPlanExecutor(plan, {}, {"pick": SequenceVerifier([])})
        with self.assertRaisesRegex(ValueError, "missing verifiers.*pick"):
            SequentialPlanExecutor(plan, {"pick": FakeAdapter()}, {})
        with self.assertRaisesRegex(ValueError, "max_retries"):
            SequentialPlanExecutor(
                plan,
                {"pick": FakeAdapter()},
                {"pick": SequenceVerifier([])},
                max_retries=-1,
            )
        with self.assertRaisesRegex(ValueError, "action timeout.*positive"):
            SequentialPlanExecutor(
                plan,
                {"pick": FakeAdapter()},
                {"pick": SequenceVerifier([])},
                action_timeouts={"pick": 0},
            )

    def test_default_configuration_retries_once_and_uses_long_timeout(self):
        adapter = FakeAdapter("pick")
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": SequenceVerifier([
                VerificationResult(success=False, code="RETRY"),
                VerificationResult(success=True),
            ])},
        )

        runner.step({"frame": 0})
        self.assertEqual(runner._current_timeout, 10000)
        run_to_completion(runner)

        self.assertTrue(runner.success)
        self.assertEqual(runner.report.steps[0].attempt_count, 2)
        self.assertEqual(adapter.resets, 2)

    def test_prepare_receives_executor_as_runner_context(self):
        adapter = FakeAdapter("pick")
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": SequenceVerifier([VerificationResult(success=True)])},
        )

        runner.step({"frame": 0})

        self.assertIs(adapter.prepare_contexts[0], runner)
        self.assertEqual(adapter.prepare_frames, [1])

    def test_retry_pre_state_is_first_state_after_adapter_cleanup(self):
        verifier = SequenceVerifier([
            VerificationResult(success=False, code="RETRY"),
            VerificationResult(success=True),
        ])
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": FakeAdapter("pick")},
            {"pick": verifier},
        )

        runner.step({"marker": {"attempt": "first"}})
        runner.step({"marker": {"attempt": "second"}})

        first_request, second_request = verifier.requests
        self.assertEqual(first_request.post_state["marker"], {"attempt": "first"})
        self.assertEqual(second_request.pre_state["marker"], {"attempt": "second"})
        self.assertEqual(second_request.post_state["marker"], {"attempt": "second"})
        self.assertIsNot(
            first_request.post_state["marker"],
            second_request.pre_state["marker"],
        )

    def test_reset_restarts_partial_execution_and_resets_only_active_adapter(self):
        adapter = TwoFrameAdapter("pick")
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": SequenceVerifier([VerificationResult(success=True)])},
        )
        runner.step({"nested": {"value": 1}})

        runner.reset()

        self.assertEqual(adapter.resets, 1)
        self.assertEqual(runner.index, 0)
        self.assertEqual(runner.frame, 0)
        self.assertFalse(runner.done)
        self.assertFalse(runner.success)
        self.assertEqual(runner.report, ExecutionReport())
        self.assertIsNone(runner._episode_initial_state)

    def test_reset_after_terminal_states_does_not_double_reset_adapter(self):
        success_adapter = FakeAdapter("pick")
        successful = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": success_adapter},
            {"pick": SequenceVerifier([VerificationResult(success=True)])},
        )
        run_to_completion(successful)
        self.assertEqual(success_adapter.resets, 1)
        successful.reset()
        self.assertEqual(success_adapter.resets, 1)
        self.assertFalse(successful.done)

        failed_adapter = NeverDoneAdapter("pick")
        failed = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": failed_adapter},
            {"pick": SequenceVerifier([])},
            action_timeouts={"pick": 1},
            max_retries=0,
        )
        run_to_completion(failed)
        self.assertEqual(failed_adapter.resets, 1)
        failed.reset()
        self.assertEqual(failed_adapter.resets, 1)
        self.assertFalse(failed.done)

        empty = SequentialPlanExecutor(
            make_plan().model_copy(update={"actions": []}),
            {},
            {},
        )
        empty.reset()
        self.assertTrue(empty.done)
        self.assertTrue(empty.success)

    def test_prepare_error_cleans_up_once_and_terminalizes(self):
        adapter = FailingLifecycleAdapter("prepare", reset_failure=True)
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": SequenceVerifier([])},
        )

        self.assertIsNone(runner.step({"frame": 1}))

        self.assert_terminal_error(runner, adapter, "ADAPTER_PREPARE_ERROR")
        measurements = runner.report.steps[0].verification.measurements
        self.assertEqual(measurements["exception_type"], "RuntimeError")
        self.assertEqual(measurements["cleanup_exception_type"], "ValueError")

    def test_step_and_status_errors_clean_up_and_terminalize(self):
        for failure, code in (
            ("step", "ADAPTER_STEP_ERROR"),
            ("is_done", "ADAPTER_STATUS_ERROR"),
        ):
            with self.subTest(failure=failure):
                adapter = FailingLifecycleAdapter(failure)
                runner = SequentialPlanExecutor(
                    single_step_plan(),
                    {"pick": adapter},
                    {"pick": SequenceVerifier([])},
                )

                action = runner.step({"frame": 1})
                if failure == "step":
                    self.assertIsNone(action)

                self.assert_terminal_error(runner, adapter, code)
                self.assertEqual(
                    runner.report.steps[0].verification.measurements["exception_type"],
                    "RuntimeError",
                )

    def test_verifier_and_invalid_result_errors_terminalize(self):
        cases = (
            (RaisingVerifier(), "VERIFIER_ERROR"),
            (SequenceVerifier([{"success": "not-a-bool"}]), "VERIFICATION_RESULT_INVALID"),
        )
        for verifier, code in cases:
            with self.subTest(code=code):
                adapter = FakeAdapter("pick")
                runner = SequentialPlanExecutor(
                    single_step_plan(),
                    {"pick": adapter},
                    {"pick": verifier},
                )

                runner.step({"frame": 1})

                self.assert_terminal_error(runner, adapter, code)
                self.assertTrue(runner.report.steps[0].controller_completed)

    def test_reset_error_replaces_success_or_retry_transition(self):
        for result in (
            VerificationResult(success=True, code="COMPLETE"),
            VerificationResult(success=False, code="RETRY_ME"),
        ):
            with self.subTest(prior_code=result.code):
                adapter = FailingLifecycleAdapter(reset_failure=True)
                runner = SequentialPlanExecutor(
                    single_step_plan(),
                    {"pick": adapter},
                    {"pick": SequenceVerifier([result])},
                    max_retries=1,
                )

                runner.step({"frame": 1})

                self.assert_terminal_error(runner, adapter, "ADAPTER_RESET_ERROR")
                record = runner.report.steps[0]
                self.assertFalse(record.success)
                self.assertEqual(
                    record.verification.measurements["prior_verification_code"],
                    result.code,
                )
                self.assertEqual(adapter.prepare_step_ids, ["step_001"])

    def test_public_reset_failure_terminalizes_until_cleanup_succeeds(self):
        adapter = FailingLifecycleAdapter(reset_failure=True, never_done=True)
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": SequenceVerifier([])},
            action_timeouts={"pick": 10},
        )
        runner.step({"frame": 1})

        with self.assertRaisesRegex(ValueError, "secret reset details"):
            runner.reset()

        self.assertEqual(adapter.resets, 1)
        self.assertEqual(runner.index, 0)
        self.assertEqual(runner.frame, 1)
        self.assertTrue(runner.done)
        self.assertFalse(runner.success)
        self.assertEqual(len(runner.report.steps), 1)
        self.assertEqual(
            runner.report.steps[0].verification.code,
            "ADAPTER_RESET_ERROR",
        )
        self.assertIsNone(runner.step({"ignored": True}))
        self.assertEqual(adapter.forward_calls, 1)

        adapter.reset_failure = False
        runner.reset()
        self.assertEqual(adapter.resets, 2)
        self.assertEqual(runner.report, ExecutionReport())
        self.assertFalse(runner.done)

    def test_internal_reset_failure_stays_pending_until_public_cleanup(self):
        adapter = FailingLifecycleAdapter(reset_failure=True)
        verifier = SequenceVerifier([
            VerificationResult(success=True),
            VerificationResult(success=True),
        ])
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": verifier},
        )

        runner.step({"frame": 1})
        terminal_report = runner.report
        self.assert_terminal_error(runner, adapter, "ADAPTER_RESET_ERROR")
        self.assertEqual(adapter.forward_calls, 1)

        adapter.reset_failure = False
        runner.reset()
        self.assertEqual(adapter.resets, 2)
        self.assertEqual(runner.report, ExecutionReport())
        self.assertFalse(runner.done)

        runner.step({"frame": 1})
        self.assertTrue(runner.success)
        self.assertEqual(adapter.forward_calls, 2)
        self.assertEqual(adapter.resets, 3)
        self.assertIsNot(runner.report, terminal_report)

    def test_persistent_pending_cleanup_preserves_terminal_report(self):
        adapter = FailingLifecycleAdapter(reset_failure=True)
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": SequenceVerifier([VerificationResult(success=True)])},
        )
        runner.step({"frame": 1})
        terminal_report = runner.report

        with self.assertRaisesRegex(ValueError, "secret reset details"):
            runner.reset()

        self.assertEqual(adapter.resets, 2)
        self.assertIs(runner.report, terminal_report)
        self.assertTrue(runner.done)
        self.assertEqual(
            runner.report.steps[0].verification.code,
            "ADAPTER_RESET_ERROR",
        )
        self.assertIsNone(runner.step({"ignored": True}))
        self.assertEqual(adapter.forward_calls, 1)

        adapter.reset_failure = False
        runner.reset()
        self.assertEqual(adapter.resets, 3)
        self.assertFalse(runner.done)
        self.assertEqual(runner.report, ExecutionReport())

    def test_state_snapshot_error_cleans_up_and_terminalizes(self):
        adapter = FakeAdapter("pick")
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": SequenceVerifier([])},
        )

        self.assertIsNone(runner.step(ExplodingState()))

        self.assert_terminal_error(runner, adapter, "STATE_SNAPSHOT_ERROR")
        self.assertEqual(adapter.forward_calls, 0)

    def test_verification_request_error_cleans_up_and_terminalizes(self):
        DeepcopyBomb.calls = 0
        adapter = FakeAdapter("pick")
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": adapter},
            {"pick": SequenceVerifier([])},
        )

        runner.step({"bomb": DeepcopyBomb()})

        self.assert_terminal_error(runner, adapter, "VERIFICATION_REQUEST_INVALID")
        self.assertEqual(adapter.forward_calls, 1)

    def test_plan_steps_and_verification_results_are_defensively_owned(self):
        plan = single_step_plan()
        adapter = MutatingStepAdapter("pick")
        result = VerificationResult(
            success=True,
            measurements={"nested": {"value": 1}},
        )
        verifier = MutatingStepVerifier(result)
        runner = SequentialPlanExecutor(
            plan,
            {"pick": adapter},
            {"pick": verifier},
        )

        plan.actions[0].id = "step_997"
        plan.actions[0].type = "place"
        plan_view = runner.plan
        plan_view.actions[0].id = "step_996"
        current_step = runner.current_step
        current_step.id = "step_995"

        runner.step({"frame": 1})
        result.measurements["nested"]["value"] = 999

        record = runner.report.steps[0]
        self.assertEqual(record.step_id, "step_001")
        self.assertEqual(record.action, "pick")
        self.assertEqual(record.object_id, "solid_flask")
        self.assertEqual(record.verification.measurements["nested"]["value"], 1)
        self.assertEqual(adapter.prepare_step_ids, ["step_001"])
        self.assertEqual(runner.plan.actions[0].id, "step_001")

    def test_verifier_mapping_result_is_detached_from_caller_mutation(self):
        raw_result = {
            "success": True,
            "measurements": {"nested": {"value": 1}},
        }
        runner = SequentialPlanExecutor(
            single_step_plan(),
            {"pick": FakeAdapter("pick")},
            {"pick": SequenceVerifier([raw_result])},
        )

        runner.step({"frame": 1})
        raw_result["measurements"]["nested"]["value"] = 999

        self.assertEqual(
            runner.report.steps[0].verification.measurements["nested"]["value"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
