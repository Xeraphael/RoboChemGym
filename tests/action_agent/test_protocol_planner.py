import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from agent.planning.models import AgentPlan, UnresolvedCapability
from agent.planning.protocol_planner import (
    PlanningAttempt,
    PlanningResult,
    ProtocolPlanningService,
)
from agent.planning.registry import CapabilityRegistry
from agent.planning.validator import (
    PlanValidator,
    ValidationReport,
    plan_fingerprint,
    registry_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = CapabilityRegistry.load_default(ROOT)
DEFAULT_REGISTRY_FINGERPRINT = registry_fingerprint(DEFAULT_REGISTRY)


class FakeClient:
    def __init__(self, responses: list[object]):
        self.responses = iter(responses)
        self.messages = []

    def complete(self, messages, *, model, temperature):
        self.messages.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def plan_json(actions: list[dict]) -> str:
    return AgentPlan(
        plan_id="protocol1",
        scene={"objects": [
            {"id": "a", "asset_id": "ErlenmeyerFlask", "instance_name": "FlaskA", "role": "container"},
            {"id": "b", "asset_id": "ErlenmeyerFlask", "instance_name": "FlaskB", "role": "container"},
        ]},
        actions=actions,
    ).model_dump_json()


class ProtocolPlanningServiceTests(unittest.TestCase):
    def setUp(self):
        self.validator = PlanValidator(DEFAULT_REGISTRY)

    def test_invalid_first_plan_is_repaired_and_revalidated(self):
        client = FakeClient([
            plan_json([{"id": "step_001", "type": "pour", "object": "b", "target": "a"}]),
            plan_json([
                {"id": "step_001", "type": "pick", "object": "b"},
                {"id": "step_002", "type": "pour", "object": "b", "target": "a"},
            ]),
        ])
        result = ProtocolPlanningService(client, self.validator, ROOT, model="test-model").create_plan("pour B into A")
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.attempt_count, 2)
        self.assertIn("OBJECT_NOT_HELD", str(client.messages[1]))
        system_prompt = client.messages[0][0]["content"]
        self.assertIn('"assets"', system_prompt)
        self.assertIn("锥形瓶", system_prompt)
        self.assertIn('"pour"', system_prompt)
        self.assertIn('"$defs"', system_prompt)

    def test_erlenmeyer_instance_names_are_canonicalized_by_variant(self):
        plan = AgentPlan(
            plan_id="canonical_names",
            scene={"objects": [
                {
                    "id": "source",
                    "asset_id": "ErlenmeyerFlask",
                    "instance_name": "anything",
                    "role": "container",
                    "properties": {"content_phase": "liquid"},
                },
                {
                    "id": "target",
                    "asset_id": "ErlenmeyerFlask",
                    "instance_name": "another_name",
                    "role": "container",
                    "properties": {"content_phase": "solid"},
                },
            ]},
            actions=[
                {"id": "step_001", "type": "pick", "object": "source"},
                {
                    "id": "step_002",
                    "type": "pour",
                    "object": "source",
                    "target": "target",
                },
            ],
        )
        result = ProtocolPlanningService(
            FakeClient([plan.model_dump_json()]),
            self.validator,
            ROOT,
            model="test-model",
        ).create_plan("pour")

        self.assertEqual(result.status, "valid")
        self.assertEqual(
            [obj.instance_name for obj in result.plan.scene.objects],
            ["ErlenmeyerFlask_Liquid1", "ErlenmeyerFlask_Solid1"],
        )

    def test_system_prompt_separates_degraded_semantics_from_core_blockers(self):
        client = FakeClient([plan_json([])])

        ProtocolPlanningService(
            client,
            self.validator,
            ROOT,
            model="test-model",
        ).create_plan("place a solid reagent container and heat until dissolved")

        system_prompt = client.messages[0][0]["content"]
        self.assertIn(
            "Existing container assets represent their contents through properties",
            system_prompt,
        )
        self.assertIn(
            "Unobservable outcomes, temperature limits, simultaneous-action wording, and duration conditions",
            system_prompt,
        )
        self.assertIn(
            "Never put the same protocol requirement in both semantic_annotations and unresolved_capabilities",
            system_prompt,
        )
        self.assertIn(
            "Only use unresolved_capabilities when no registered asset/action can perform the closest required physical operation",
            system_prompt,
        )
        self.assertIn('"variant_property": "content_phase"', system_prompt)
        self.assertIn('"parameter_constraints"', system_prompt)
        self.assertIn('"maximum": -0.1', system_prompt)
        self.assertIn('"default_parameters"', system_prompt)
        self.assertIn('"unresolved_capabilities": []', system_prompt)

    def test_repair_context_requires_a_complete_plan_without_blocked_issues(self):
        blocked_plan = AgentPlan.model_validate_json(
            plan_json([{"id": "step_001", "type": "pick", "object": "b"}])
        )
        blocked_plan.unresolved_capabilities.append(
            UnresolvedCapability(
                source_text="until dissolved",
                missing_action="stir_until_dissolved",
                reason="state is not observable",
            )
        )
        repaired_plan = blocked_plan.model_copy(deep=True)
        repaired_plan.unresolved_capabilities = []
        client = FakeClient(
            [blocked_plan.model_dump_json(), repaired_plan.model_dump_json()]
        )

        result = ProtocolPlanningService(
            client,
            self.validator,
            ROOT,
            model="test-model",
        ).create_plan("pick the container until dissolved")

        self.assertEqual(result.status, "valid")
        repair_context = json.loads(client.messages[1][1]["content"])
        self.assertEqual(
            repair_context["repair_instruction"],
            "Return a complete replacement AgentPlan and resolve every blocked validator issue. Do not preserve an unresolved capability when the closest registered physical action exists and only its outcome or modifier is unobservable.",
        )

    def test_three_invalid_attempts_return_planning_failed(self):
        bad = plan_json([{"id": "step_001", "type": "pour", "object": "b", "target": "a"}])
        result = ProtocolPlanningService(FakeClient([bad, bad, bad]), self.validator, ROOT, model="test-model").create_plan("bad")
        self.assertEqual(result.status, "planning_failed")
        self.assertEqual(result.attempt_count, 3)
        self.assertIsNone(result.plan)

    def test_json_parse_error_is_returned_on_the_next_attempt(self):
        client = FakeClient(["not json", plan_json([
            {"id": "step_001", "type": "pick", "object": "b"},
            {"id": "step_002", "type": "pour", "object": "b", "target": "a"},
        ])])
        result = ProtocolPlanningService(client, self.validator, ROOT, model="test-model").create_plan("pour B into A")
        self.assertEqual(result.status, "valid")
        self.assertIn("PLAN_JSON_INVALID", client.messages[1][1]["content"])

    def test_provider_exception_returns_client_failed_with_last_report(self):
        blocked = plan_json([
            {"id": "step_001", "type": "pour", "object": "b", "target": "a"},
        ])
        result = ProtocolPlanningService(
            FakeClient([blocked, RuntimeError("provider unavailable")]),
            self.validator,
            ROOT,
            model="test-model",
        ).create_plan("pour B into A")
        self.assertEqual(result.status, "client_failed")
        self.assertEqual(result.attempt_count, 2)
        self.assertIsNone(result.plan)
        self.assertIsNotNone(result.final_report)
        self.assertIn("OBJECT_NOT_HELD", str(result.final_report))
        self.assertEqual(result.attempts[-1].raw_response, "")
        self.assertIn(
            "RuntimeError: provider unavailable",
            result.attempts[-1].client_error,
        )

    def test_none_response_is_a_repairable_parse_error(self):
        valid = plan_json([
            {"id": "step_001", "type": "pick", "object": "b"},
            {"id": "step_002", "type": "pour", "object": "b", "target": "a"},
        ])
        client = FakeClient([None, valid])
        result = ProtocolPlanningService(
            client,
            self.validator,
            ROOT,
            model="test-model",
        ).create_plan("pour B into A")
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.attempts[0].raw_response, "")
        self.assertIn("response must be a string", result.attempts[0].parse_error)
        self.assertIn("PLAN_JSON_INVALID", client.messages[1][1]["content"])

    def test_truncated_or_trailed_fence_is_a_parse_error(self):
        for response, message in [
            ("```json\n{}", "closing fence"),
            ("```json\n{}\n``` trailing", "trailing text"),
        ]:
            with self.subTest(response=response):
                result = ProtocolPlanningService(
                    FakeClient([response]),
                    self.validator,
                    ROOT,
                    model="test-model",
                ).create_plan("bad", max_attempts=1)
                self.assertEqual(result.status, "planning_failed")
                self.assertIn(message, result.attempts[0].parse_error)

    def test_parse_failures_preserve_the_last_validator_diagnosis(self):
        blocked = plan_json([
            {"id": "step_001", "type": "pour", "object": "b", "target": "a"},
        ])
        client = FakeClient([blocked, "not json", "still not json"])
        result = ProtocolPlanningService(
            client,
            self.validator,
            ROOT,
            model="test-model",
        ).create_plan("pour B into A")
        self.assertEqual(result.status, "planning_failed")
        self.assertIsNotNone(result.final_report)
        self.assertIn("OBJECT_NOT_HELD", str(result.final_report))
        third_context = client.messages[2][1]["content"]
        self.assertIn("OBJECT_NOT_HELD", third_context)
        self.assertIn("PLAN_JSON_INVALID", third_context)


class PlanningModelInvariantTests(unittest.TestCase):
    def test_attempt_requires_exactly_one_outcome(self):
        with self.assertRaises(ValidationError):
            PlanningAttempt(index=1, raw_response="")
        with self.assertRaises(ValidationError):
            PlanningAttempt(
                index=1,
                raw_response="",
                parse_error="invalid",
                client_error="provider failed",
            )

    def test_result_rejects_invalid_status_combinations(self):
        plan = AgentPlan.model_validate_json(plan_json([]))
        valid_report = ValidationReport(
            plan_fingerprint=plan_fingerprint(plan),
            registry_fingerprint=DEFAULT_REGISTRY_FINGERPRINT,
            valid=True,
        )
        invalid_report = ValidationReport(
            plan_fingerprint=plan_fingerprint(plan),
            registry_fingerprint=DEFAULT_REGISTRY_FINGERPRINT,
            valid=False,
        )
        parse_attempt = PlanningAttempt(
            index=1,
            raw_response="bad",
            parse_error="invalid",
        )
        validation_attempt = PlanningAttempt(
            index=1,
            raw_response=plan.model_dump_json(),
            validation_report=invalid_report,
        )

        invalid_cases = [
            {"status": "unknown", "attempts": [parse_attempt]},
            {"status": "valid", "attempts": [parse_attempt]},
            {
                "status": "valid",
                "plan": plan,
                "final_report": invalid_report,
                "attempts": [validation_attempt],
            },
            {
                "status": "planning_failed",
                "plan": plan,
                "attempts": [parse_attempt],
            },
            {
                "status": "planning_failed",
                "final_report": valid_report,
                "attempts": [parse_attempt],
            },
            {"status": "client_failed", "attempts": [parse_attempt]},
        ]
        for kwargs in invalid_cases:
            with self.subTest(status=kwargs["status"]):
                with self.assertRaises(ValidationError):
                    PlanningResult(**kwargs)

    def test_valid_result_rejects_report_for_a_different_plan(self):
        plan = AgentPlan.model_validate_json(plan_json([]))
        other_plan = plan.model_copy(update={"plan_id": "protocol2"})
        other_report = ValidationReport(
            plan_fingerprint=plan_fingerprint(other_plan),
            registry_fingerprint=DEFAULT_REGISTRY_FINGERPRINT,
            valid=True,
        )

        with self.assertRaisesRegex(ValidationError, "fingerprint"):
            PlanningResult(
                status="valid",
                plan=plan,
                final_report=other_report,
            )

    def test_validation_report_requires_registry_fingerprint(self):
        plan = AgentPlan.model_validate_json(plan_json([]))

        with self.assertRaisesRegex(ValidationError, "registry_fingerprint"):
            ValidationReport(
                plan_fingerprint=plan_fingerprint(plan),
                valid=True,
            )


if __name__ == "__main__":
    unittest.main()
