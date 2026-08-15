import json
import math
import subprocess
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

from agent.action.plan_execution.models import VerificationRequest
from agent.action.plan_execution.verifiers import (
    CloseVerifier,
    OpenVerifier,
    PickVerifier,
    PlaceVerifier,
    PourVerifier,
    PressVerifier,
    PressZVerifier,
    ShakeVerifier,
)
from agent.planning.models import ActionStep, ActionType


ROOT = Path(__file__).resolve().parents[2]


def make_request(step, *, pre=None, post=None, history=None, initial=None):
    return VerificationRequest(
        step=step,
        pre_state={} if pre is None else pre,
        post_state={} if post is None else post,
        state_history=[] if history is None else history,
        episode_initial_state={} if initial is None else initial,
    )


def radial_position(angle_degrees):
    angle = np.deg2rad(angle_degrees)
    return np.array([np.cos(angle), np.sin(angle), 0.8])


class ActionVerifierTests(unittest.TestCase):
    def test_pick_requires_lift_and_gripper_proximity(self):
        step = ActionStep(id="step_001", type=ActionType.PICK, object="flask")
        history = [
            {
                "Flask_position": np.array([0.0, 0.0, z]),
                "gripper_position": np.array([0.0, 0.0, z + 0.01]),
            }
            for z in [0.88, 0.89, 0.90, 0.91, 0.92]
        ]
        result = PickVerifier().verify(
            VerificationRequest(
                step=step,
                pre_state={"Flask_position": np.array([0.0, 0.0, 0.8])},
                post_state=history[-1],
                state_history=history,
                episode_initial_state={"Flask_position": np.array([0.0, 0.0, 0.8])},
            )
        )
        self.assertTrue(result.success)

    def test_place_requires_target_proximity(self):
        step = ActionStep(
            id="step_002",
            type=ActionType.PLACE,
            object="flask",
            target="plate",
        )
        result = PlaceVerifier().verify(
            VerificationRequest(
                step=step,
                pre_state={},
                post_state={
                    "Flask_position": np.array([0.01, 0.0, 0.8]),
                    "Plate_place_position": np.array([0.0, 0.0, 0.8]),
                    "joint_positions": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0.04, 0.04]
                    ),
                },
                state_history=[],
                episode_initial_state={},
            )
        )
        self.assertTrue(result.success)

    def test_pour_reports_motion_only_success(self):
        step = ActionStep(
            id="step_003",
            type=ActionType.POUR,
            object="flask",
            target="target",
        )
        history = []
        for wrist in [0.0, 0.9, 1.0, 1.0, 1.0, 0.2]:
            history.append(
                {
                    "joint_positions": np.array(
                        [0, 0, 0, 0, 0, 0, wrist, 0.01, 0.01]
                    ),
                    "gripper_position": np.array([0.0, 0.0, 1.0]),
                    "Flask_position": np.array([0.0, 0.0, 0.99]),
                    "Target_position": np.array([0.03, 0.0, 0.9]),
                }
            )
        result = PourVerifier().verify(
            VerificationRequest(
                step=step,
                pre_state={
                    "joint_positions": np.array([0, 0, 0, 0, 0, 0, 0.0, 0, 0])
                },
                post_state=history[-1],
                state_history=history,
                episode_initial_state={},
            )
        )
        self.assertTrue(result.success)
        self.assertEqual(result.verification_level, "motion_only")

    def test_press_requires_gripper_to_reach_button(self):
        step = ActionStep(id="step_004", type=ActionType.PRESS, target="plate")
        history = [
            {
                "gripper_position": np.array([x, 0.0, 0.85]),
                "Plate_press_position": np.array([0.0, 0.0, 0.8]),
            }
            for x in [0.10, 0.05, 0.02, 0.01, 0.08]
        ]
        result = PressVerifier().verify(
            VerificationRequest(
                step=step,
                pre_state={"Plate_press_position": np.array([0.0, 0.0, 0.8])},
                post_state=history[-1],
                state_history=history,
                episode_initial_state={},
            )
        )
        self.assertTrue(result.success)
        self.assertEqual(result.verification_level, "motion_only")

    def test_horizontal_press_uses_controller_tool_target_above_button(self):
        step = ActionStep(id="step_104", type=ActionType.PRESS, target="plate")
        button = np.array([0.0874837, -0.2561596, 0.8316344])
        gripper = np.array([0.0785648, -0.2546161, 0.8840139])

        result = PressVerifier().verify(
            make_request(
                step,
                pre={"Plate_press_position": button},
                post={"Plate_press_position": button},
                history=[
                    {
                        "gripper_position": gripper,
                        "Plate_press_position": button,
                    }
                ],
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.code, "OK")
        self.assertEqual(result.verification_level, "motion_only")
        self.assertGreater(
            result.measurements["minimum_gripper_button_distance"], 0.04
        )
        self.assertLess(
            result.measurements["minimum_gripper_target_pose_distance"], 0.04
        )

    def test_press_z_keeps_button_anchor_as_controller_target(self):
        step = ActionStep(id="step_105", type=ActionType.PRESS_Z, target="plate")
        button = np.array([0.0874837, -0.2561596, 0.8316344])
        gripper = np.array([0.0785648, -0.2546161, 0.8840139])

        result = PressZVerifier().verify(
            make_request(
                step,
                pre={"Plate_pressz_position": button},
                post={"Plate_pressz_position": button},
                history=[
                    {
                        "gripper_position": gripper,
                        "Plate_pressz_position": button,
                    }
                ],
            )
        )

        self.assertFalse(result.success)
        self.assertEqual(result.code, "BUTTON_NOT_REACHED")
        self.assertAlmostEqual(
            result.measurements["minimum_gripper_target_pose_distance"],
            result.measurements["minimum_gripper_button_distance"],
        )

    def test_horizontal_press_rejects_a_true_tool_target_miss(self):
        step = ActionStep(id="step_106", type=ActionType.PRESS, target="plate")
        button = np.array([0.0, 0.0, 0.8])

        result = PressVerifier().verify(
            make_request(
                step,
                pre={"Plate_press_position": button},
                post={"Plate_press_position": button},
                history=[
                    {
                        "gripper_position": np.array([0.041, 0.0, 0.85]),
                        "Plate_press_position": button,
                    }
                ],
            )
        )

        self.assertFalse(result.success)
        self.assertEqual(result.code, "BUTTON_NOT_REACHED")
        self.assertGreater(
            result.measurements["minimum_gripper_target_pose_distance"], 0.04
        )

    def test_shake_requires_amplitude_and_direction_changes(self):
        step = ActionStep(id="step_005", type=ActionType.SHAKE, object="flask")
        history = [
            {
                "gripper_position": np.array([0.0, y, 1.0]),
                "Flask_position": np.array([0.0, y, 0.99]),
            }
            for y in [0.0, -0.1, 0.1, -0.1, 0.1, 0.0]
        ]
        result = ShakeVerifier().verify(
            VerificationRequest(
                step=step,
                pre_state={},
                post_state=history[-1],
                state_history=history,
                episode_initial_state={},
            )
        )
        self.assertTrue(result.success)

    def test_open_and_close_use_handle_motion_around_the_joint_center(self):
        initial = {
            "Door_handle_position": np.array([1.0, 0.0, 0.8]),
            "Door_revolute_joint_position": np.array([0.0, 0.0, 0.8]),
        }
        opened = {
            "Door_handle_position": np.array([0.5, 0.866, 0.8]),
            "Door_revolute_joint_position": np.array([0.0, 0.0, 0.8]),
        }
        open_step = ActionStep(
            id="step_006",
            type=ActionType.OPEN,
            target="door",
            parameters={"angle": 50.0},
        )
        open_result = OpenVerifier().verify(
            VerificationRequest(
                step=open_step,
                pre_state=initial,
                post_state=opened,
                state_history=[initial, opened],
                episode_initial_state=initial,
            )
        )
        self.assertTrue(open_result.success)

        close_step = ActionStep(
            id="step_007",
            type=ActionType.CLOSE,
            target="door",
            parameters={"angle": 50.0},
        )
        close_result = CloseVerifier().verify(
            VerificationRequest(
                step=close_step,
                pre_state=opened,
                post_state=initial,
                state_history=[opened, initial],
                episode_initial_state=initial,
            )
        )
        self.assertTrue(close_result.success)

    def test_pick_uses_exact_plan_instance_mapping_with_state_priority(self):
        step = ActionStep(
            id="step_008",
            type=ActionType.PICK,
            object="source_flask",
        )
        instance = "Exact_Source_Instance_17"
        history = [
            {
                f"{instance}_position": np.array([0.0, 0.0, 0.9]),
                "gripper_position": np.array([0.0, 0.0, 0.91]),
            }
            for _ in range(5)
        ]
        result = PickVerifier().verify(
            make_request(
                step,
                pre={
                    "plan_instance_names": {"source_flask": "Wrong_Pre_Name"},
                    f"{instance}_position": np.array([0.0, 0.0, 0.8]),
                },
                post={
                    "plan_instance_names": {"source_flask": instance},
                    f"{instance}_position": np.array([0.0, 0.0, 0.9]),
                },
                history=history,
                initial={
                    "plan_instance_names": {"source_flask": "Wrong_Initial_Name"}
                },
            )
        )
        self.assertTrue(result.success)

    def test_instance_mapping_search_continues_when_newer_mapping_lacks_id(self):
        step = ActionStep(
            id="step_009",
            type=ActionType.PLACE,
            object="sample_vial",
            target="destination_rack",
        )
        result = PlaceVerifier().verify(
            make_request(
                step,
                pre={
                    "plan_instance_names": {"sample_vial": "Mapped_Vial_9"}
                },
                post={
                    "plan_instance_names": {"unrelated": "Other_Instance"},
                    "Mapped_Vial_9_position": np.array([0.08, 0.0, 0.8]),
                    "Mapped_Rack_4_place_position": np.array([0.0, 0.0, 0.8]),
                    "joint_positions": np.array([0.0] * 7 + [0.025, -0.025]),
                },
                initial={
                    "plan_instance_names": {
                        "destination_rack": "Mapped_Rack_4"
                    }
                },
            )
        )
        self.assertTrue(result.success)

    def test_instance_mapping_present_but_missing_ref_fails_closed(self):
        step = ActionStep(id="step_036", type=ActionType.PICK, object="flask")
        history = [
            {
                "Flask_position": np.array([0.0, 0.0, 0.9]),
                "gripper_position": np.array([0.0, 0.0, 0.9]),
            }
            for _ in range(5)
        ]
        result = PickVerifier().verify(
            make_request(
                step,
                pre={
                    "plan_instance_names": {},
                    "Flask_position": np.array([0.0, 0.0, 0.8]),
                },
                post={
                    "plan_instance_names": {"other": "Other_Instance"},
                    "Flask_position": np.array([0.0, 0.0, 0.9]),
                },
                history=history,
                initial={"plan_instance_names": {"flask": ""}},
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(result.code, "PICK_STATE_MISSING")

    def test_instance_fallback_applies_without_mapping_valued_state(self):
        step = ActionStep(id="step_037", type=ActionType.PICK, object="flask")
        history = [
            {
                "Flask_position": np.array([0.0, 0.0, 0.9]),
                "gripper_position": np.array([0.0, 0.0, 0.9]),
            }
            for _ in range(5)
        ]
        result = PickVerifier().verify(
            make_request(
                step,
                pre={
                    "plan_instance_names": None,
                    "Flask_position": np.array([0.0, 0.0, 0.8]),
                },
                post={
                    "plan_instance_names": ["not", "a", "mapping"],
                    "Flask_position": np.array([0.0, 0.0, 0.9]),
                },
                history=history,
            )
        )
        self.assertTrue(result.success)

    def test_pick_thresholds_include_exact_lift_distance_and_frame_boundaries(self):
        step = ActionStep(id="step_010", type=ActionType.PICK, object="flask")
        history = [
            {
                "Flask_position": np.array([0.0, 0.0, 0.08]),
                "gripper_position": np.array([0.10, 0.0, 0.08]),
            }
            for _ in range(5)
        ]
        request = make_request(
            step,
            pre={"Flask_position": np.array([0.0, 0.0, 0.0])},
            post=history[-1],
            history=history,
        )
        result = PickVerifier().verify(request)
        self.assertTrue(result.success)
        self.assertEqual(result.measurements["stable_frames"], 5)

        short_result = PickVerifier().verify(
            request.model_copy(update={"state_history": history[:-1]})
        )
        self.assertFalse(short_result.success)
        self.assertEqual(short_result.code, "PICK_STATE_MISSING")

    def test_pick_thresholds_reject_values_just_outside_contract(self):
        step = ActionStep(id="step_038", type=ActionType.PICK, object="flask")

        def verify(lift, distance):
            history = [
                {
                    "Flask_position": np.array([0.0, 0.0, lift]),
                    "gripper_position": np.array([distance, 0.0, lift]),
                }
                for _ in range(5)
            ]
            return PickVerifier().verify(
                make_request(
                    step,
                    pre={"Flask_position": np.array([0.0, 0.0, 0.0])},
                    post=history[-1],
                    history=history,
                )
            )

        for lift, distance in (
            (0.07999999995, 0.0),
            (0.08, 0.10000000005),
        ):
            with self.subTest(lift=lift, distance=distance):
                result = verify(lift, distance)
                self.assertFalse(result.success)
                self.assertEqual(result.code, "GRASP_NOT_ESTABLISHED")

    def test_pick_decimal_boundary_allows_only_ulp_scale_rounding(self):
        step = ActionStep(id="step_043", type=ActionType.PICK, object="flask")
        rounded_distance = math.nextafter(0.10, math.inf)
        history = [
            {
                "Flask_position": np.array([0.0, 0.0, 0.88]),
                "gripper_position": np.array(
                    [rounded_distance, 0.0, 0.88]
                ),
            }
            for _ in range(5)
        ]
        result = PickVerifier().verify(
            make_request(
                step,
                pre={"Flask_position": np.array([0.0, 0.0, 0.80])},
                post=history[-1],
                history=history,
            )
        )
        self.assertLess(result.measurements["lift"], 0.08)
        self.assertGreater(
            result.measurements["stable_object_gripper_distance"],
            0.10,
        )
        self.assertTrue(result.success)

    def test_pick_requires_complete_terminal_window_matching_post_state(self):
        step = ActionStep(id="step_044", type=ActionType.PICK, object="flask")

        def frame():
            return {
                "Flask_position": np.array([0.0, 0.0, 0.9]),
                "gripper_position": np.array([0.0, 0.0, 0.9]),
            }

        early_history = [frame() for _ in range(5)]
        malformed_terminal = [frame() for _ in range(5)]
        malformed_terminal[2]["Flask_position"] = np.array(
            [0.0, 0.0, np.nan]
        )
        cases = [
            (
                "malformed_terminal_frame",
                [*early_history, *malformed_terminal],
                malformed_terminal[-1],
            ),
            (
                "post_state_mismatch",
                [*early_history, *[frame() for _ in range(5)]],
                {
                    "Flask_position": np.array([0.01, 0.0, 0.9]),
                    "gripper_position": np.array([0.0, 0.0, 0.9]),
                },
            ),
        ]
        for label, history, post_state in cases:
            with self.subTest(case=label):
                result = PickVerifier().verify(
                    make_request(
                        step,
                        pre={"Flask_position": np.array([0.0, 0.0, 0.8])},
                        post=post_state,
                        history=history,
                    )
                )
                self.assertFalse(result.success)
                self.assertEqual(result.code, "PICK_STATE_MISSING")

    def test_pick_derived_overflow_returns_json_safe_missing_state(self):
        step = ActionStep(id="step_045", type=ActionType.PICK, object="flask")
        history = [
            {
                "Flask_position": np.array([0.0, 0.0, 1e308]),
                "gripper_position": np.array([0.0, 0.0, 1e308]),
            }
            for _ in range(5)
        ]
        result = PickVerifier().verify(
            make_request(
                step,
                pre={"Flask_position": np.array([0.0, 0.0, -1e308])},
                post=history[-1],
                history=history,
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(result.code, "PICK_STATE_MISSING")
        json.dumps(result.model_dump(mode="python"), allow_nan=False)

    def test_pick_complete_observation_reports_grasp_failure(self):
        step = ActionStep(id="step_011", type=ActionType.PICK, object="flask")
        history = [
            {
                "Flask_position": np.array([0.0, 0.0, 0.87]),
                "gripper_position": np.array([0.11, 0.0, 0.87]),
            }
            for _ in range(5)
        ]
        result = PickVerifier().verify(
            make_request(
                step,
                pre={"Flask_position": np.array([0.0, 0.0, 0.8])},
                post=history[-1],
                history=history,
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(result.code, "GRASP_NOT_ESTABLISHED")
        self.assertEqual(result.verification_level, "state_observed")

    def test_place_thresholds_and_failure_code(self):
        step = ActionStep(
            id="step_012",
            type=ActionType.PLACE,
            object="flask",
            target="plate",
        )
        post = {
            "Flask_position": np.array([0.08, 0.0, 0.8]),
            "Plate_place_position": np.array([0.0, 0.0, 0.8]),
            "joint_positions": np.array([0.0] * 7 + [0.025, -0.025]),
        }
        result = PlaceVerifier().verify(make_request(step, post=post))
        self.assertTrue(result.success)

        failed = PlaceVerifier().verify(
            make_request(
                step,
                post={**post, "Flask_position": np.array([0.081, 0.0, 0.8])},
            )
        )
        self.assertFalse(failed.success)
        self.assertEqual(failed.code, "PLACE_OR_RELEASE_FAILED")

    def test_pour_requires_three_rotated_hold_frames_at_thresholds(self):
        step = ActionStep(
            id="step_013",
            type=ActionType.POUR,
            object="flask",
            target="target",
        )
        base = {
            "gripper_position": np.array([0.1, 0.0, 0.0]),
            "Flask_position": np.array([0.0, 0.0, 0.0]),
            "Target_position": np.array([0.1, 0.35, 0.0]),
        }
        history = [
            {
                **base,
                "joint_positions": np.array(
                    [0.0] * 6 + [np.deg2rad(45.0), 0.0, 0.0]
                ),
            }
            for _ in range(3)
        ]
        request = make_request(
            step,
            pre={"joint_positions": np.zeros(9)},
            post=history[-1],
            history=history,
        )
        result = PourVerifier().verify(request)
        self.assertTrue(result.success)
        self.assertEqual(result.verification_level, "motion_only")
        self.assertIs(type(result.measurements["rotated_hold_frames"]), int)
        self.assertEqual(result.measurements["rotated_hold_frames"], 3)

        outside_target_radius = [
            {
                **frame,
                "Target_position": np.array([0.1, 0.350001, 0.0]),
            }
            for frame in history
        ]
        outside = PourVerifier().verify(
            request.model_copy(
                update={
                    "post_state": outside_target_radius[-1],
                    "state_history": outside_target_radius,
                }
            )
        )
        self.assertFalse(outside.success)
        self.assertEqual(outside.code, "POUR_POSE_NOT_REACHED")

        two_rotated = [
            *history[:2],
            {**base, "joint_positions": np.zeros(9)},
        ]
        failed = PourVerifier().verify(
            request.model_copy(
                update={"post_state": two_rotated[-1], "state_history": two_rotated}
            )
        )
        self.assertFalse(failed.success)
        self.assertEqual(failed.code, "POUR_POSE_NOT_REACHED")
        self.assertEqual(failed.measurements["rotated_hold_frames"], 2)

    def test_pour_requires_consecutive_qualifying_hold_frames(self):
        step = ActionStep(
            id="step_046",
            type=ActionType.POUR,
            object="flask",
            target="target",
        )

        def frame(wrist):
            if wrist is None:
                return {
                    "joint_positions": np.ones(9),
                    "gripper_position": np.array([0.0, np.nan, 0.0]),
                    "Flask_position": np.array([0.0, 0.0, 0.0]),
                    "Target_position": np.array([0.0, 0.1, 0.0]),
                }
            return {
                "joint_positions": np.array([0.0] * 6 + [wrist, 0.0, 0.0]),
                "gripper_position": np.array([0.0, 0.0, 0.0]),
                "Flask_position": np.array([0.0, 0.0, 0.0]),
                "Target_position": np.array([0.0, 0.1, 0.0]),
            }

        cases = [
            ("alternating", [1.0, 0.0, 1.0, 0.0, 1.0], False, 1),
            ("malformed_break", [1.0, 1.0, None, 1.0, 1.0], False, 2),
            ("return_after_hold", [1.0, 1.0, 1.0, 0.0], True, 3),
        ]
        for label, wrists, success, longest_run in cases:
            with self.subTest(case=label):
                history = [frame(wrist) for wrist in wrists]
                result = PourVerifier().verify(
                    make_request(
                        step,
                        pre={"joint_positions": np.zeros(9)},
                        post=history[-1],
                        history=history,
                    )
                )
                self.assertEqual(result.success, success)
                self.assertEqual(
                    result.code,
                    "OK" if success else "POUR_POSE_NOT_REACHED",
                )
                self.assertEqual(
                    result.measurements["rotated_hold_frames"],
                    longest_run,
                )

    def test_pour_fails_if_source_is_dropped_after_completed_hold(self):
        step = ActionStep(
            id="step_058",
            type=ActionType.POUR,
            object="flask",
            target="target",
        )
        held_frame = {
            "joint_positions": np.array([0.0] * 6 + [1.0, 0.0, 0.0]),
            "gripper_position": np.zeros(3),
            "Flask_position": np.zeros(3),
            "Target_position": np.array([0.0, 0.1, 0.0]),
        }
        dropped_return = {
            "joint_positions": np.zeros(9),
            "gripper_position": np.zeros(3),
            "Flask_position": np.array([0.5, 0.0, 0.0]),
            "Target_position": np.array([0.0, 0.1, 0.0]),
        }
        result = PourVerifier().verify(
            make_request(
                step,
                pre={"joint_positions": np.zeros(9)},
                post=dropped_return,
                history=[held_frame, held_frame, held_frame, dropped_return],
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(result.code, "POUR_POSE_NOT_REACHED")
        self.assertEqual(result.measurements["rotated_hold_frames"], 3)
        self.assertEqual(result.measurements["max_source_gripper_distance"], 0.5)

    def test_press_levels_follow_observed_button_displacement(self):
        step = ActionStep(id="step_014", type=ActionType.PRESS, target="plate")
        before = np.array([0.0, 0.0, 0.0])
        history = [
            {
                "gripper_position": np.array([0.04, 0.0, 0.05]),
                "Plate_press_position": before,
            }
        ]
        observed = PressVerifier().verify(
            make_request(
                step,
                pre={"Plate_press_position": before},
                post={"Plate_press_position": np.array([0.0, 0.0, -0.003])},
                history=history,
            )
        )
        self.assertTrue(observed.success)
        self.assertEqual(observed.verification_level, "state_observed")

        missed = PressVerifier().verify(
            make_request(
                step,
                pre={"Plate_press_position": before},
                post={"Plate_press_position": before},
                history=[
                    {
                        "gripper_position": np.array([0.041, 0.0, 0.05]),
                        "Plate_press_position": before,
                    }
                ],
            )
        )
        self.assertFalse(missed.success)
        self.assertEqual(missed.code, "BUTTON_NOT_REACHED")
        self.assertEqual(missed.verification_level, "motion_only")

    def test_press_z_uses_pressz_anchor_key(self):
        step = ActionStep(id="step_015", type=ActionType.PRESS_Z, target="plate")
        before = np.array([0.0, 0.0, 0.0])
        result = PressZVerifier().verify(
            make_request(
                step,
                pre={
                    "Plate_pressz_position": before,
                    "Plate_press_position": np.array([1.0, 0.0, 0.0]),
                },
                post={
                    "Plate_pressz_position": np.array([0.0, 0.0, -0.003]),
                    "Plate_press_position": np.array([1.0, 0.0, 0.0]),
                },
                history=[
                    {
                        "gripper_position": np.array([0.04, 0.0, 0.0]),
                        "Plate_pressz_position": before,
                        "Plate_press_position": np.array([1.0, 0.0, 0.0]),
                    }
                ],
            )
        )
        self.assertTrue(result.success)
        self.assertEqual(result.verification_level, "state_observed")

    def test_shake_thresholds_require_two_real_direction_change_cycles(self):
        step = ActionStep(id="step_016", type=ActionType.SHAKE, object="flask")

        def history_for(y_values):
            return [
                {
                    "gripper_position": np.array([0.0, y, 0.0]),
                    "Flask_position": np.array([0.1, y, 0.0]),
                }
                for y in y_values
            ]

        boundary_history = history_for([0.0, -0.06, 0.06, -0.06, 0.06, -0.06])
        result = ShakeVerifier().verify(
            make_request(step, post=boundary_history[-1], history=boundary_history)
        )
        self.assertTrue(result.success)
        self.assertEqual(result.measurements["cycles"], 2)

        zero_padded = history_for([-0.06, 0.0, 0.06, 0.0, -0.06, 0.06])
        failed = ShakeVerifier().verify(
            make_request(step, post=zero_padded[-1], history=zero_padded)
        )
        self.assertFalse(failed.success)
        self.assertEqual(failed.code, "SHAKE_PATTERN_INCOMPLETE")
        self.assertEqual(failed.measurements["cycles"], 1)

    def test_shake_ignores_tiny_jitter_near_a_single_extreme(self):
        step = ActionStep(id="step_053", type=ActionType.SHAKE, object="flask")
        history = [
            {
                "gripper_position": np.array([0.0, y, 0.0]),
                "Flask_position": np.array([0.0, y, 0.01]),
            }
            for y in [-0.06, 0.06, 0.059, 0.06, 0.059, 0.06]
        ]
        result = ShakeVerifier().verify(
            make_request(step, post=history[-1], history=history)
        )
        self.assertFalse(result.success)
        self.assertEqual(result.code, "SHAKE_PATTERN_INCOMPLETE")
        self.assertEqual(result.measurements["cycles"], 0)

    def test_open_threshold_uses_seventy_percent_of_default_angle(self):
        step = ActionStep(id="step_017", type=ActionType.OPEN, target="door")
        center = np.array([0.0, 0.0, 0.8])
        before = {
            "Door_handle_position": radial_position(0.0),
            "Door_revolute_joint_position": center,
        }
        boundary = {
            "Door_handle_position": radial_position(35.0),
            "Door_revolute_joint_position": center,
        }
        result = OpenVerifier().verify(
            make_request(step, pre=before, post=boundary, initial=before)
        )
        self.assertLess(
            result.measurements["observed_angle"],
            result.measurements["required_angle"],
        )
        self.assertTrue(result.success)

        below = {
            **boundary,
            "Door_handle_position": radial_position(34.9),
        }
        failed = OpenVerifier().verify(
            make_request(step, pre=before, post=below, initial=before)
        )
        self.assertFalse(failed.success)
        self.assertEqual(failed.code, "OPEN_ANGLE_NOT_REACHED")

    def test_open_exact_angle_boundary_passes(self):
        step = ActionStep(
            id="step_039",
            type=ActionType.OPEN,
            target="door",
            parameters={"angle": 90.0 / 0.7},
        )
        center = np.array([0.0, 0.0, 0.8])
        before = {
            "Door_handle_position": np.array([1.0, 0.0, 0.8]),
            "Door_revolute_joint_position": center,
        }
        boundary = {
            "Door_handle_position": np.array([0.0, 1.0, 0.8]),
            "Door_revolute_joint_position": center,
        }
        result = OpenVerifier().verify(
            make_request(step, pre=before, post=boundary, initial=before)
        )
        self.assertTrue(result.success)
        self.assertEqual(
            result.measurements["observed_angle"],
            result.measurements["required_angle"],
        )

    def test_open_requires_progress_away_from_episode_closed_pose(self):
        step = ActionStep(id="step_054", type=ActionType.OPEN, target="door")
        center = np.array([0.0, 0.0, 0.8])

        def state(angle):
            return {
                "Door_handle_position": radial_position(angle),
                "Door_revolute_joint_position": center,
            }

        closed = state(0.0)
        with self.subTest(case="toward_closed"):
            toward_closed = OpenVerifier().verify(
                make_request(
                    step,
                    pre=state(80.0),
                    post=state(40.0),
                    initial=closed,
                )
            )
            self.assertFalse(toward_closed.success)
            self.assertEqual(toward_closed.code, "OPEN_ANGLE_NOT_REACHED")

        with self.subTest(case="reverse_hinge"):
            reverse_hinge = OpenVerifier().verify(
                make_request(
                    step,
                    pre=closed,
                    post=state(-40.0),
                    initial=closed,
                )
            )
            self.assertTrue(reverse_hinge.success)

        with self.subTest(case="missing_closed"):
            missing_closed = OpenVerifier().verify(
                make_request(
                    step,
                    pre=closed,
                    post=state(40.0),
                    initial={},
                )
            )
            self.assertFalse(missing_closed.success)
            self.assertEqual(missing_closed.code, "OPEN_STATE_MISSING")

    def test_close_thresholds_include_residual_and_requested_motion_boundaries(self):
        step = ActionStep(
            id="step_018",
            type=ActionType.CLOSE,
            target="door",
            parameters={"angle": 50.0},
        )
        center = np.array([0.0, 0.0, 0.8])
        initial = {
            "Door_handle_position": radial_position(0.0),
            "Door_revolute_joint_position": center,
        }
        before = {
            "Door_handle_position": radial_position(47.0),
            "Door_revolute_joint_position": center,
        }
        boundary = {
            "Door_handle_position": radial_position(12.0),
            "Door_revolute_joint_position": center,
        }
        result = CloseVerifier().verify(
            make_request(step, pre=before, post=boundary, initial=initial)
        )
        self.assertTrue(result.success)

        residual_too_large = {
            **boundary,
            "Door_handle_position": radial_position(12.1),
        }
        failed = CloseVerifier().verify(
            make_request(
                step,
                pre=before,
                post=residual_too_large,
                initial=initial,
            )
        )
        self.assertFalse(failed.success)
        self.assertEqual(failed.code, "CLOSE_ANGLE_NOT_REACHED")

    def test_missing_state_returns_stable_codes_and_handles_none_references(self):
        cases = [
            (
                PickVerifier(),
                ActionStep(id="step_019", type=ActionType.PICK),
                "PICK_STATE_MISSING",
            ),
            (
                PlaceVerifier(),
                ActionStep(id="step_020", type=ActionType.PLACE),
                "PLACE_STATE_MISSING",
            ),
            (
                PourVerifier(),
                ActionStep(id="step_021", type=ActionType.POUR),
                "POUR_STATE_MISSING",
            ),
            (
                PressVerifier(),
                ActionStep(id="step_022", type=ActionType.PRESS),
                "PRESS_STATE_MISSING",
            ),
            (
                PressZVerifier(),
                ActionStep(id="step_023", type=ActionType.PRESS_Z),
                "PRESS_STATE_MISSING",
            ),
            (
                ShakeVerifier(),
                ActionStep(id="step_024", type=ActionType.SHAKE),
                "SHAKE_HISTORY_TOO_SHORT",
            ),
            (
                OpenVerifier(),
                ActionStep(id="step_025", type=ActionType.OPEN),
                "OPEN_STATE_MISSING",
            ),
            (
                CloseVerifier(),
                ActionStep(id="step_026", type=ActionType.CLOSE),
                "CLOSE_STATE_MISSING",
            ),
        ]
        for verifier, step, code in cases:
            with self.subTest(verifier=type(verifier).__name__):
                result = verifier.verify(make_request(step))
                self.assertFalse(result.success)
                self.assertEqual(result.code, code)

    def test_pick_malformed_vectors_return_missing_state(self):
        step = ActionStep(id="step_027", type=ActionType.PICK, object="flask")
        valid_history = [
            {
                "Flask_position": np.array([0.0, 0.0, 0.9]),
                "gripper_position": np.array([0.0, 0.0, 0.9]),
            }
            for _ in range(5)
        ]
        for invalid in (
            np.array([0.0, 0.0]),
            np.array(["bad", 0.0, 0.8]),
            np.array([0.0, 0.0, np.nan]),
        ):
            with self.subTest(invalid=repr(invalid)):
                result = PickVerifier().verify(
                    make_request(
                        step,
                        pre={"Flask_position": invalid},
                        post=valid_history[-1],
                        history=valid_history,
                    )
                )
                self.assertFalse(result.success)
                self.assertEqual(result.code, "PICK_STATE_MISSING")

    def test_place_malformed_vectors_and_joints_return_missing_state(self):
        step = ActionStep(
            id="step_028",
            type=ActionType.PLACE,
            object="flask",
            target="plate",
        )
        valid = {
            "Flask_position": np.array([0.0, 0.0, 0.8]),
            "Plate_place_position": np.array([0.0, 0.0, 0.8]),
            "joint_positions": np.ones(9) * 0.03,
        }
        mutations = [
            {"Flask_position": np.array([0.0, 0.0])},
            {"Plate_place_position": np.array(["bad", 0.0, 0.8])},
            {"joint_positions": np.array(0.03)},
            {"joint_positions": np.array([0.0] * 8 + [np.nan])},
        ]
        for mutation in mutations:
            with self.subTest(mutation=repr(mutation)):
                result = PlaceVerifier().verify(
                    make_request(step, post={**valid, **mutation})
                )
                self.assertFalse(result.success)
                self.assertEqual(result.code, "PLACE_STATE_MISSING")

    def test_pour_malformed_pre_or_history_state_returns_missing_state(self):
        step = ActionStep(
            id="step_029",
            type=ActionType.POUR,
            object="flask",
            target="target",
        )
        valid_frame = {
            "joint_positions": np.array([0.0] * 6 + [1.0, 0.0, 0.0]),
            "gripper_position": np.array([0.0, 0.0, 0.0]),
            "Flask_position": np.array([0.0, 0.0, 0.0]),
            "Target_position": np.array([0.0, 0.0, 0.0]),
        }
        for invalid_joints in (
            np.array(0.0),
            np.array([0.0] * 6 + ["bad"]),
            np.array([0.0] * 6 + [np.nan]),
        ):
            with self.subTest(pre=repr(invalid_joints)):
                result = PourVerifier().verify(
                    make_request(
                        step,
                        pre={"joint_positions": invalid_joints},
                        post=valid_frame,
                        history=[valid_frame] * 3,
                    )
                )
                self.assertEqual(result.code, "POUR_STATE_MISSING")
                self.assertEqual(result.verification_level, "motion_only")

        malformed_history = [
            {**valid_frame, "gripper_position": np.array([0.0, np.nan, 0.0])},
            valid_frame,
            valid_frame,
        ]
        result = PourVerifier().verify(
            make_request(
                step,
                pre={"joint_positions": np.zeros(9)},
                post=valid_frame,
                history=malformed_history,
            )
        )
        self.assertEqual(result.code, "POUR_STATE_MISSING")

    def test_press_malformed_vectors_return_missing_state(self):
        for verifier, action_type, anchor in (
            (PressVerifier(), ActionType.PRESS, "press_position"),
            (PressZVerifier(), ActionType.PRESS_Z, "pressz_position"),
        ):
            step = ActionStep(id="step_030", type=action_type, target="plate")
            key = f"Plate_{anchor}"
            valid = np.array([0.0, 0.0, 0.0])
            for invalid in (
                np.array([0.0, 0.0]),
                np.array(["bad", 0.0, 0.0]),
                np.array([0.0, 0.0, np.nan]),
            ):
                with self.subTest(verifier=type(verifier).__name__, invalid=repr(invalid)):
                    result = verifier.verify(
                        make_request(
                            step,
                            pre={key: invalid},
                            post={key: valid},
                            history=[
                                {key: valid, "gripper_position": valid}
                            ],
                        )
                    )
                    self.assertEqual(result.code, "PRESS_STATE_MISSING")

    def test_shake_short_and_malformed_tracking_return_structured_failures(self):
        step = ActionStep(id="step_031", type=ActionType.SHAKE, object="flask")
        short = ShakeVerifier().verify(
            make_request(
                step,
                history=[
                    {"gripper_position": np.array([0.0, 0.0, 0.0])}
                    for _ in range(3)
                ],
            )
        )
        self.assertEqual(short.code, "SHAKE_HISTORY_TOO_SHORT")

        missing_object = [
            {
                "gripper_position": np.array([0.0, y, 0.0]),
                "Flask_position": np.array([0.0, y, 0.0]),
            }
            for y in [0.0, -0.1, 0.1, -0.1]
        ]
        missing_object[-1].pop("Flask_position")
        tracking = ShakeVerifier().verify(
            make_request(step, history=missing_object)
        )
        self.assertEqual(tracking.code, "SHAKE_OBJECT_TRACKING_MISSING")

        malformed = [
            {
                "gripper_position": np.array([0.0, y, 0.0]),
                "Flask_position": np.array([0.0, y, 0.0]),
            }
            for y in [0.0, -0.1, 0.1, -0.1]
        ]
        malformed[0]["gripper_position"] = np.array([0.0, np.nan, 0.0])
        malformed_result = ShakeVerifier().verify(
            make_request(step, history=malformed)
        )
        self.assertEqual(malformed_result.code, "SHAKE_HISTORY_TOO_SHORT")

    def test_open_and_close_malformed_state_return_missing_codes(self):
        center = np.array([0.0, 0.0, 0.8])
        valid = {
            "Door_handle_position": np.array([1.0, 0.0, 0.8]),
            "Door_revolute_joint_position": center,
        }
        invalid_values = (
            np.array([1.0]),
            np.array(["bad", 0.0, 0.8]),
            np.array([np.nan, 0.0, 0.8]),
        )
        for invalid in invalid_values:
            with self.subTest(verifier="open", invalid=repr(invalid)):
                step = ActionStep(id="step_032", type=ActionType.OPEN, target="door")
                result = OpenVerifier().verify(
                    make_request(
                        step,
                        pre={**valid, "Door_handle_position": invalid},
                        post=valid,
                        initial=valid,
                    )
                )
                self.assertEqual(result.code, "OPEN_STATE_MISSING")

            with self.subTest(verifier="close", invalid=repr(invalid)):
                step = ActionStep(id="step_033", type=ActionType.CLOSE, target="door")
                result = CloseVerifier().verify(
                    make_request(
                        step,
                        pre={**valid, "Door_handle_position": invalid},
                        post=valid,
                        initial=valid,
                    )
                )
                self.assertEqual(result.code, "CLOSE_STATE_MISSING")

        bad_angle = ActionStep(
            id="step_034",
            type=ActionType.OPEN,
            target="door",
            parameters={"angle": "bad"},
        )
        result = OpenVerifier().verify(
            make_request(bad_angle, pre=valid, post=valid, initial=valid)
        )
        self.assertEqual(result.code, "OPEN_STATE_MISSING")

    def test_numeric_overflow_returns_structured_missing_state(self):
        huge = 10**1000

        pick_step = ActionStep(
            id="step_040",
            type=ActionType.PICK,
            object="flask",
        )
        pick_history = [
            {
                "Flask_position": np.array([0.0, 0.0, 0.1]),
                "gripper_position": np.array([0.0, 0.0, 0.1]),
            }
            for _ in range(5)
        ]
        with self.subTest(path="position"):
            pick_result = PickVerifier().verify(
                make_request(
                    pick_step,
                    pre={"Flask_position": [huge, 0, 0]},
                    post=pick_history[-1],
                    history=pick_history,
                )
            )
            self.assertEqual(pick_result.code, "PICK_STATE_MISSING")

        place_step = ActionStep(
            id="step_041",
            type=ActionType.PLACE,
            object="flask",
            target="plate",
        )
        with self.subTest(path="joints"):
            place_result = PlaceVerifier().verify(
                make_request(
                    place_step,
                    post={
                        "Flask_position": np.array([0.0, 0.0, 0.0]),
                        "Plate_place_position": np.array([0.0, 0.0, 0.0]),
                        "joint_positions": [0] * 8 + [huge],
                    },
                )
            )
            self.assertEqual(place_result.code, "PLACE_STATE_MISSING")

        open_step = ActionStep(
            id="step_042",
            type=ActionType.OPEN,
            target="door",
            parameters={"angle": huge},
        )
        door_state = {
            "Door_handle_position": np.array([1.0, 0.0, 0.8]),
            "Door_revolute_joint_position": np.array([0.0, 0.0, 0.8]),
        }
        with self.subTest(path="angle"):
            open_result = OpenVerifier().verify(
                make_request(
                    open_step,
                    pre=door_state,
                    post=door_state,
                    initial=door_state,
                )
            )
            self.assertEqual(open_result.code, "OPEN_STATE_MISSING")

    def test_derived_overflow_is_missing_and_json_safe_for_each_verifier(self):
        place_step = ActionStep(
            id="step_047",
            type=ActionType.PLACE,
            object="flask",
            target="plate",
        )
        place_request = make_request(
            place_step,
            post={
                "Flask_position": np.array([1e308, 0.0, 0.0]),
                "Plate_place_position": np.array([-1e308, 0.0, 0.0]),
                "joint_positions": np.array([0.0] * 7 + [0.03, 0.03]),
            },
        )

        pour_step = ActionStep(
            id="step_048",
            type=ActionType.POUR,
            object="flask",
            target="target",
        )
        pour_frame = {
            "joint_positions": np.array([0.0] * 6 + [1e308, 0.0, 0.0]),
            "gripper_position": np.zeros(3),
            "Flask_position": np.zeros(3),
            "Target_position": np.array([0.0, 0.1, 0.0]),
        }
        pour_request = make_request(
            pour_step,
            pre={
                "joint_positions": np.array(
                    [0.0] * 6 + [-1e308, 0.0, 0.0]
                )
            },
            post=pour_frame,
            history=[pour_frame] * 3,
        )

        press_step = ActionStep(
            id="step_049",
            type=ActionType.PRESS,
            target="plate",
        )
        button = np.array([-1e308, 0.0, 0.0])
        press_request = make_request(
            press_step,
            pre={"Plate_press_position": button},
            post={"Plate_press_position": button},
            history=[
                {
                    "Plate_press_position": button,
                    "gripper_position": np.array([1e308, 0.0, 0.0]),
                }
            ],
        )

        shake_step = ActionStep(
            id="step_050",
            type=ActionType.SHAKE,
            object="flask",
        )
        shake_history = [
            {
                "gripper_position": np.array([0.0, y, 0.0]),
                "Flask_position": np.array([0.0, y, 0.0]),
            }
            for y in [-1e308, 1e308, -1e308, 1e308]
        ]
        shake_request = make_request(
            shake_step,
            post=shake_history[-1],
            history=shake_history,
        )

        center = np.array([-1e308, 0.0, 0.0])
        closed_handle = np.array([1e308, 0.0, 0.0])
        offset_handle = np.array([1e308, 1.0, 0.0])
        open_step = ActionStep(
            id="step_051",
            type=ActionType.OPEN,
            target="door",
        )
        open_request = make_request(
            open_step,
            pre={
                "Door_handle_position": closed_handle,
                "Door_revolute_joint_position": center,
            },
            post={
                "Door_handle_position": offset_handle,
                "Door_revolute_joint_position": center,
            },
            initial={
                "Door_handle_position": closed_handle,
                "Door_revolute_joint_position": center,
            },
        )
        close_step = ActionStep(
            id="step_052",
            type=ActionType.CLOSE,
            target="door",
        )
        close_request = make_request(
            close_step,
            pre={
                "Door_handle_position": offset_handle,
                "Door_revolute_joint_position": center,
            },
            post={
                "Door_handle_position": closed_handle,
                "Door_revolute_joint_position": center,
            },
            initial={
                "Door_handle_position": closed_handle,
                "Door_revolute_joint_position": center,
            },
        )

        cases = [
            (PlaceVerifier(), place_request, "PLACE_STATE_MISSING"),
            (PourVerifier(), pour_request, "POUR_STATE_MISSING"),
            (PressVerifier(), press_request, "PRESS_STATE_MISSING"),
            (
                ShakeVerifier(),
                shake_request,
                "SHAKE_OBJECT_TRACKING_MISSING",
            ),
            (OpenVerifier(), open_request, "OPEN_STATE_MISSING"),
            (CloseVerifier(), close_request, "CLOSE_STATE_MISSING"),
        ]
        for verifier, request, expected_code in cases:
            with self.subTest(verifier=type(verifier).__name__):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    result = verifier.verify(request)
                self.assertFalse(result.success)
                self.assertEqual(result.code, expected_code)
                json.dumps(result.model_dump(mode="python"), allow_nan=False)

    def test_large_finite_derived_values_are_scaled_without_warnings(self):
        place_step = ActionStep(
            id="step_055",
            type=ActionType.PLACE,
            object="flask",
            target="plate",
        )
        place_request = make_request(
            place_step,
            post={
                "Flask_position": np.zeros(3),
                "Plate_place_position": np.zeros(3),
                "joint_positions": np.array([0.0] * 7 + [1e308, 1e308]),
            },
        )

        center = np.array([0.0, 0.0, 0.8])
        closed = {
            "Door_handle_position": np.array([1e308, 0.0, 0.8]),
            "Door_revolute_joint_position": center,
        }
        opened = {
            "Door_handle_position": np.array([0.0, 1e308, 0.8]),
            "Door_revolute_joint_position": center,
        }
        open_request = make_request(
            ActionStep(
                id="step_056",
                type=ActionType.OPEN,
                target="door",
                parameters={"angle": 90.0 / 0.7},
            ),
            pre=closed,
            post=opened,
            initial=closed,
        )
        close_request = make_request(
            ActionStep(
                id="step_057",
                type=ActionType.CLOSE,
                target="door",
                parameters={"angle": 90.0 / 0.7},
            ),
            pre=opened,
            post=closed,
            initial=closed,
        )

        cases = [
            (PlaceVerifier(), place_request),
            (OpenVerifier(), open_request),
            (CloseVerifier(), close_request),
        ]
        for verifier, request in cases:
            with self.subTest(verifier=type(verifier).__name__):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    result = verifier.verify(request)
                self.assertTrue(result.success)
                json.dumps(result.model_dump(mode="python"), allow_nan=False)

    def test_result_payloads_are_json_safe(self):
        step = ActionStep(id="step_035", type=ActionType.PRESS, target="plate")
        point = np.array([0.0, 0.0, 0.0])
        result = PressVerifier().verify(
            make_request(
                step,
                pre={"Plate_press_position": point},
                post={"Plate_press_position": point},
                history=[
                    {
                        "Plate_press_position": point,
                        "gripper_position": np.array([0.0, 0.0, 0.05]),
                    }
                ],
            )
        )
        encoded = json.dumps(result.model_dump(mode="python"), allow_nan=False)
        self.assertIn('"success": true', encoded)
        self.assertIs(type(result.success), bool)
        self.assertIs(type(result.code), str)
        for measurement in result.measurements.values():
            self.assertIn(type(measurement), (int, float, bool, str, type(None)))

    def test_import_does_not_load_simulation_modules(self):
        code = """
import sys
import agent.action.plan_execution.verifiers
for prefix in ('isaacsim', 'omni', 'pxr'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
