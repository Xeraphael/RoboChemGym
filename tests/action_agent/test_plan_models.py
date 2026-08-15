import unittest
from pydantic import ValidationError

from agent.planning.models import (
    ActionStep,
    ActionType,
    AgentPlan,
    AnnotationStatus,
    SceneObject,
    ScenePlan,
    SemanticAnnotation,
)


def make_plan() -> AgentPlan:
    return AgentPlan(
        schema_version="1.0",
        plan_id="example_protocol",
        scene=ScenePlan(objects=[
            SceneObject(
                id="solid_flask",
                asset_id="ErlenmeyerFlask",
                instance_name="ErlenmeyerFlask_Solid1",
                role="reagent_container",
                properties={"content_phase": "solid"},
            ),
            SceneObject(
                id="plate",
                asset_id="HeatingPlate",
                instance_name="HeatingPlate",
                role="device",
            ),
        ]),
        actions=[
            ActionStep(id="step_001", type=ActionType.PICK, object="solid_flask"),
            ActionStep(
                id="step_002",
                type=ActionType.PLACE,
                object="solid_flask",
                target="plate",
            ),
        ],
    )


class AgentPlanModelTests(unittest.TestCase):
    def test_round_trip_preserves_enum_values(self):
        plan = make_plan()
        restored = AgentPlan.model_validate_json(plan.model_dump_json())
        self.assertIsInstance(restored.actions[0].type, ActionType)
        self.assertEqual(restored.actions[0].type, ActionType.PICK)
        self.assertEqual(restored.scene.objects[0].instance_name, "ErlenmeyerFlask_Solid1")

    def test_duplicate_object_ids_are_rejected(self):
        obj = make_plan().scene.objects[0]
        with self.assertRaises(ValidationError):
            AgentPlan(
                schema_version="1.0",
                plan_id="bad",
                scene=ScenePlan(objects=[obj, obj.model_copy()]),
                actions=[],
            )

    def test_unknown_action_types_are_rejected(self):
        with self.assertRaises(ValidationError):
            ActionStep(id="step_001", type="aspirate", object="solid_flask")

    def test_semantic_annotations_must_reference_known_steps(self):
        plan = make_plan()
        data = plan.model_dump(mode="python")
        data["semantic_annotations"] = [SemanticAnnotation(
            source_text="avoid overheating",
            status=AnnotationStatus.NOT_OBSERVABLE,
            reason="temperature is not observed",
            step_ids=["step_999"],
        ).model_dump(mode="python")]
        with self.assertRaises(ValidationError):
            AgentPlan.model_validate(data)


if __name__ == "__main__":
    unittest.main()
