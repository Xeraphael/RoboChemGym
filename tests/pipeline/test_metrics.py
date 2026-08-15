import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agent.action.plan_execution.models import VerificationResult
from pipeline.evaluation import (
    PassivePlanEvaluator,
    bound_policy_action,
    expand_policy_action,
    validate_checkpoint_compatibility,
)
from pipeline.metrics import summarize_evaluation
from evaluate import create_run_directory
from tests.action_agent.test_plan_models import make_plan


class AchievementVerifier:
    def verify(self, request):
        success = request.step.id in request.post_state.get("achieved", [])
        return VerificationResult(
            success=success,
            code="OK" if success else "CONDITION_NOT_REACHED",
            measurements={"object_target_distance": 0.1},
        )


class EvaluationMetricsTests(unittest.TestCase):
    def test_evaluation_run_directory_is_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = datetime(2026, 8, 14, tzinfo=timezone.utc)
            first_id, first = create_run_directory(Path(tmp), now)
            second_id, second = create_run_directory(Path(tmp), now)
            self.assertEqual(first_id, "20260814T000000Z")
            self.assertEqual(second_id, "20260814T000000Z_01")
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_policy_action_expands_mirrored_gripper(self):
        expanded = expand_policy_action(np.arange(8, dtype=np.float32))
        np.testing.assert_allclose(expanded, [0, 1, 2, 3, 4, 5, 6, 7, 7])
        with self.assertRaises(ValueError):
            expand_policy_action([0] * 7)

        bounded = bound_policy_action(
            np.ones(8),
            np.zeros(9),
            max_step=0.05,
        )
        np.testing.assert_allclose(bounded, np.full(9, 0.05))

    def test_checkpoint_contract_matches_runtime_observations(self):
        compatibility = {
            "model_type": "act",
            "model_arguments": {
                "camera_names": ["camera_1_rgb"],
                "robot_state_dim": 8,
                "action_dim": 8,
                "num_queries": 60,
            },
            "dataset_schema_version": "1.0",
            "split_version": "1.0",
            "camera_order": ["camera_1_rgb"],
            "camera_shapes": {"camera_1_rgb": [3, 256, 256]},
            "action_convention": "absolute_joint_position",
            "gripper_convention": "first_finger_position_second_mirrored",
            "dataset_identity": {
                "config_id": "fixture",
                "episodes": [{"episode_id": "episode_0000", "length": 10}],
            },
        }
        arguments = validate_checkpoint_compatibility(
            compatibility,
            ["camera_1_rgb"],
            {"camera_1_rgb": [3, 256, 256]},
        )
        self.assertEqual(arguments["num_queries"], 60)
        with self.assertRaisesRegex(ValueError, "camera order"):
            validate_checkpoint_compatibility(
                compatibility,
                ["camera_2_rgb"],
                {"camera_2_rgb": [3, 256, 256]},
            )

    def test_passive_evaluator_finishes_on_first_sequential_achievement(self):
        plan = make_plan()
        evaluator = PassivePlanEvaluator(
            plan,
            {step.id: "supported" for step in plan.actions},
            {"pick": AchievementVerifier, "place": AchievementVerifier},
        )

        self.assertFalse(evaluator.update({"achieved": []}))
        self.assertFalse(evaluator.update({"achieved": ["step_001"]}))
        self.assertTrue(
            evaluator.update({"achieved": ["step_001", "step_002"]})
        )

        report = evaluator.report()
        self.assertTrue(report["success"])
        self.assertEqual(report["length"], 3)
        self.assertEqual(
            [step["achieved_frame"] for step in report["steps"]],
            [2, 3],
        )

    def test_passive_evaluator_reports_failed_and_unreached_steps(self):
        plan = make_plan()
        evaluator = PassivePlanEvaluator(
            plan,
            {step.id: "supported" for step in plan.actions},
            {"pick": AchievementVerifier, "place": AchievementVerifier},
        )
        evaluator.update({"achieved": []})

        report = evaluator.report()

        self.assertFalse(report["success"])
        self.assertEqual(report["failed_step"], "step_001")
        self.assertEqual(report["failure_code"], "CONDITION_NOT_REACHED")
        self.assertEqual(report["steps"][1]["code"], "NOT_REACHED")

    def test_summary_reports_counts_steps_failures_and_distances(self):
        summary = summarize_evaluation(
            [
                {
                    "status": "completed",
                    "success": True,
                    "length": 10,
                    "steps": [
                        {
                            "step_id": "step_001",
                            "action": "place",
                            "success": True,
                            "measurements": {"object_target_distance": 0.1},
                        }
                    ],
                },
                {
                    "status": "completed",
                    "success": False,
                    "failure_code": "TIMEOUT",
                    "length": 20,
                    "steps": [
                        {
                            "step_id": "step_001",
                            "action": "place",
                            "success": False,
                            "measurements": {},
                        }
                    ],
                },
            ]
        )
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["failure_code_distribution"], {"TIMEOUT": 1})
        self.assertEqual(summary["per_action_step_success_rate"]["step_001"], 0.5)
        self.assertEqual(summary["per_action_type_success_rate"]["place"], 0.5)
        self.assertEqual(summary["placement_terminal_distance_mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
