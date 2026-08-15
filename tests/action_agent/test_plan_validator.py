import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from agent.planning.models import (
    ActionStep,
    ActionType,
    AgentPlan,
    AnnotationStatus,
    CoverageLevel,
    SceneObject,
    ScenePlan,
    SemanticAnnotation,
    UnresolvedCapability,
)
from agent.planning.registry import (
    ActionDefinition,
    ActionRegistry,
    AssetDefinition,
    AssetRegistry,
    CapabilityRegistry,
)
from agent.planning import validator as validator_module
from agent.planning.validator import PlanValidator


ROOT = Path(__file__).resolve().parents[2]


def copy_registry_manifests(destination_root: Path) -> None:
    destination = destination_root / "agent" / "planning" / "registry"
    destination.parent.mkdir(parents=True)
    shutil.copytree(ROOT / "agent" / "planning" / "registry", destination)


def reverse_mapping_order(value):
    if isinstance(value, dict):
        return {
            key: reverse_mapping_order(value[key])
            for key in reversed(tuple(value))
        }
    if isinstance(value, list):
        return [reverse_mapping_order(item) for item in value]
    return value


def base_plan(actions: list[ActionStep]) -> AgentPlan:
    return AgentPlan(
        plan_id="validator_case",
        scene=ScenePlan(objects=[
            SceneObject(id="a", asset_id="ErlenmeyerFlask", instance_name="FlaskA", role="container"),
            SceneObject(id="b", asset_id="ErlenmeyerFlask", instance_name="FlaskB", role="container"),
            SceneObject(id="plate", asset_id="HeatingPlate", instance_name="HeatingPlate", role="device"),
        ]),
        actions=actions,
    )


class PlanValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = PlanValidator(CapabilityRegistry.load_default(ROOT))

    def test_valid_pick_place_pick_pour_sequence(self):
        plan = base_plan([
            ActionStep(id="step_001", type=ActionType.PICK, object="a"),
            ActionStep(id="step_002", type=ActionType.PLACE, object="a", target="plate"),
            ActionStep(id="step_003", type=ActionType.PICK, object="b"),
            ActionStep(id="step_004", type=ActionType.POUR, object="b", target="a"),
        ])
        report = self.validator.validate(plan)
        self.assertTrue(report.valid)
        self.assertEqual(report.blocked_count, 0)

    def test_validation_report_fingerprint_binds_canonical_plan_content(self):
        fingerprint = getattr(validator_module, "plan_fingerprint", None)
        self.assertIsNotNone(fingerprint)
        plan = base_plan([
            ActionStep(
                id="step_001",
                type=ActionType.PICK,
                object="a",
                parameters={"pre_offset_z": 0.1},
            ),
            ActionStep(
                id="step_002",
                type=ActionType.PLACE,
                object="a",
                target="plate",
            ),
        ])

        report = self.validator.validate(plan)

        self.assertRegex(report.plan_fingerprint, r"^[0-9a-f]{64}$")
        self.assertEqual(report.plan_fingerprint, fingerprint(plan))
        round_trip = AgentPlan.model_validate_json(plan.model_dump_json())
        self.assertEqual(fingerprint(round_trip), report.plan_fingerprint)

        mutations = []
        action_changed = plan.model_copy(deep=True)
        action_changed.actions[0].type = ActionType.SHAKE
        mutations.append(action_changed)
        parameter_changed = plan.model_copy(deep=True)
        parameter_changed.actions[0].parameters["pre_offset_z"] = 0.2
        mutations.append(parameter_changed)
        order_changed = plan.model_copy(deep=True)
        order_changed.actions = list(reversed(order_changed.actions))
        mutations.append(order_changed)
        scene_changed = plan.model_copy(deep=True)
        scene_changed.scene.objects[0].instance_name = "ChangedFlask"
        mutations.append(scene_changed)

        for mutated in mutations:
            with self.subTest(plan=mutated):
                self.assertNotEqual(fingerprint(mutated), report.plan_fingerprint)

    def test_registry_fingerprint_binds_manifests_not_order_or_checkout_root(self):
        fingerprint = getattr(validator_module, "registry_fingerprint", None)
        version = getattr(
            validator_module,
            "REGISTRY_FINGERPRINT_SCHEMA_VERSION",
            None,
        )
        self.assertIsNotNone(fingerprint)
        self.assertIsNotNone(version)

        registry = self.validator.registry
        expected = fingerprint(registry)
        report = self.validator.validate(base_plan([]))
        self.assertRegex(report.registry_fingerprint, r"^[0-9a-f]{64}$")
        self.assertEqual(report.registry_fingerprint, expected)

        reordered_assets = {
            name: AssetDefinition.model_validate(
                reverse_mapping_order(definition.model_dump(mode="python"))
            )
            for name, definition in reversed(
                tuple(registry.assets.definitions.items())
            )
        }
        reordered_actions = {
            name: ActionDefinition.model_validate(
                reverse_mapping_order(definition.model_dump(mode="python"))
            )
            for name, definition in reversed(
                tuple(registry.actions.definitions.items())
            )
        }
        reordered = CapabilityRegistry(
            AssetRegistry(reordered_assets),
            ActionRegistry(reordered_actions),
            Path("/different/checkout/root"),
        )
        self.assertEqual(fingerprint(reordered), expected)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            base_root = temporary_root / "base"
            copy_registry_manifests(base_root)
            base_registry = CapabilityRegistry.load_default(base_root)
            self.assertEqual(fingerprint(base_registry), expected)

            mutations = {
                "max_frames": lambda pick: pick.__setitem__(
                    "max_frames", pick["max_frames"] + 1
                ),
                "default_pre_offset_z": lambda pick: pick[
                    "default_parameters"
                ].__setitem__("pre_offset_z", 0.13),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    checkout_root = temporary_root / label
                    copy_registry_manifests(checkout_root)
                    actions_path = (
                        checkout_root
                        / "agent"
                        / "planning"
                        / "registry"
                        / "actions.yaml"
                    )
                    manifest = yaml.safe_load(
                        actions_path.read_text(encoding="utf-8")
                    )
                    mutate(manifest["actions"]["pick"])
                    actions_path.write_text(
                        yaml.safe_dump(
                            manifest,
                            allow_unicode=True,
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                    changed = CapabilityRegistry.load_default(checkout_root)
                    self.assertNotEqual(fingerprint(changed), expected)

    def test_pour_without_pick_is_blocked(self):
        report = self.validator.validate(base_plan([
            ActionStep(id="step_001", type=ActionType.POUR, object="b", target="a"),
        ]))
        self.assertFalse(report.valid)
        self.assertIn("OBJECT_NOT_HELD", [issue.code for issue in report.issues])

    def test_second_pick_while_holding_is_blocked(self):
        report = self.validator.validate(base_plan([
            ActionStep(id="step_001", type=ActionType.PICK, object="a"),
            ActionStep(id="step_002", type=ActionType.PICK, object="b"),
        ]))
        self.assertIn("GRIPPER_OCCUPIED", [issue.code for issue in report.issues])

    def test_blocked_pick_with_unknown_reference_does_not_occupy_gripper(self):
        report = self.validator.validate(base_plan([
            ActionStep(id="step_001", type=ActionType.PICK, object="missing"),
            ActionStep(id="step_002", type=ActionType.PICK, object="a"),
        ]))
        self.assertEqual(report.step_coverage["step_001"], CoverageLevel.BLOCKED)
        self.assertEqual(report.step_coverage["step_002"], CoverageLevel.SUPPORTED)
        self.assertNotIn(
            "GRIPPER_OCCUPIED",
            [issue.code for issue in report.issues if issue.step_id == "step_002"],
        )

    def test_blocked_place_with_unknown_target_does_not_release_object(self):
        report = self.validator.validate(base_plan([
            ActionStep(id="step_001", type=ActionType.PICK, object="a"),
            ActionStep(id="step_002", type=ActionType.PLACE, object="a", target="missing"),
            ActionStep(id="step_003", type=ActionType.PICK, object="b"),
        ]))
        self.assertEqual(report.step_coverage["step_002"], CoverageLevel.BLOCKED)
        self.assertEqual(report.step_coverage["step_003"], CoverageLevel.BLOCKED)
        self.assertIn(
            "GRIPPER_OCCUPIED",
            [issue.code for issue in report.issues if issue.step_id == "step_003"],
        )

    def test_unobservable_semantics_are_degraded_not_blocked(self):
        plan = base_plan([ActionStep(id="step_001", type=ActionType.PRESS, target="plate")])
        plan.semantic_annotations.append(SemanticAnnotation(
            source_text="搅拌至部分溶解",
            status=AnnotationStatus.NOT_OBSERVABLE,
            reason="dissolution state unavailable",
        ))
        report = self.validator.validate(plan)
        self.assertTrue(report.valid)
        self.assertEqual(report.degraded_count, 1)

    def test_unresolved_core_capability_blocks_execution(self):
        plan = base_plan([])
        plan.unresolved_capabilities.append(UnresolvedCapability(
            source_text="使用移液枪吸取液体",
            missing_action="aspirate",
            reason="action is not registered",
        ))
        report = self.validator.validate(plan)
        self.assertFalse(report.valid)
        self.assertEqual(report.blocked_count, 1)

    def test_not_executable_annotation_cannot_be_downgraded_to_warning(self):
        plan = base_plan([])
        plan.semantic_annotations.append(SemanticAnnotation(
            source_text="aspirate with a pipette",
            status=AnnotationStatus.NOT_EXECUTABLE,
            reason="no aspirate action exists",
        ))
        report = self.validator.validate(plan)
        self.assertFalse(report.valid)
        self.assertIn("SEMANTIC_NOT_EXECUTABLE", [issue.code for issue in report.issues])

    def test_close_requires_a_prior_open_effect(self):
        plan = base_plan([])
        plan.scene.objects.append(SceneObject(id="door", asset_id="DryingBox", instance_name="DryingBox", role="device"))
        plan.actions = [ActionStep(id="step_001", type=ActionType.CLOSE, target="door")]
        report = self.validator.validate(plan)
        self.assertFalse(report.valid)
        self.assertIn("TARGET_NOT_OPEN", [issue.code for issue in report.issues])

    def test_unknown_asset_variant_is_blocked_instead_of_silently_falling_back(self):
        plan = base_plan([
            ActionStep(id="step_001", type=ActionType.PICK, object="a"),
            ActionStep(id="step_002", type=ActionType.PICK, object="b"),
        ])
        plan.scene.objects[0].properties["content_phase"] = "liqid"
        report = self.validator.validate(plan)
        self.assertFalse(report.valid)
        self.assertIn("UNKNOWN_ASSET_VARIANT", [issue.code for issue in report.issues])
        self.assertEqual(report.step_coverage["step_001"], CoverageLevel.BLOCKED)
        self.assertEqual(report.step_coverage["step_002"], CoverageLevel.SUPPORTED)
        self.assertEqual(report.supported_count, 1)

    def test_unknown_asset_blocks_referencing_step_without_mutating_state(self):
        plan = base_plan([
            ActionStep(id="step_001", type=ActionType.PICK, object="a"),
            ActionStep(id="step_002", type=ActionType.PICK, object="b"),
        ])
        plan.scene.objects[0].asset_id = "MissingAsset"
        report = self.validator.validate(plan)
        self.assertIn("UNKNOWN_ASSET", [issue.code for issue in report.issues])
        self.assertEqual(report.step_coverage["step_001"], CoverageLevel.BLOCKED)
        self.assertEqual(report.step_coverage["step_002"], CoverageLevel.SUPPORTED)
        self.assertEqual(report.supported_count, 1)

    def test_missing_asset_file_blocks_referencing_step_without_mutating_state(self):
        definitions = dict(self.validator.registry.assets.definitions)
        definitions["Beaker"] = definitions["Beaker"].model_copy(
            update={"usd_path": "missing-validator-asset.usd"},
        )
        registry = CapabilityRegistry(
            assets=AssetRegistry(definitions),
            actions=self.validator.registry.actions,
            root=ROOT,
        )
        plan = base_plan([
            ActionStep(id="step_001", type=ActionType.PICK, object="a"),
            ActionStep(id="step_002", type=ActionType.PICK, object="b"),
        ])
        plan.scene.objects[0].asset_id = "Beaker"
        report = PlanValidator(registry).validate(plan)
        self.assertIn("ASSET_FILE_MISSING", [issue.code for issue in report.issues])
        self.assertEqual(report.step_coverage["step_001"], CoverageLevel.BLOCKED)
        self.assertEqual(report.step_coverage["step_002"], CoverageLevel.SUPPORTED)
        self.assertEqual(report.supported_count, 1)

    def test_missing_action_definition_is_blocked_without_mutating_state(self):
        asset_definitions = deepcopy(self.validator.registry.assets.definitions)
        for definition in asset_definitions.values():
            definition.supported_actions = [
                action
                for action in definition.supported_actions
                if action != ActionType.PICK.value
            ]
            definition.required_anchors.pop(ActionType.PICK.value, None)
            definition.action_defaults.pop(ActionType.PICK.value, None)
            for defaults in definition.variant_action_defaults.values():
                defaults.pop(ActionType.PICK.value, None)
        registry = CapabilityRegistry(
            assets=AssetRegistry(asset_definitions),
            actions=ActionRegistry({
                name: definition
                for name, definition in self.validator.registry.actions.definitions.items()
                if name != ActionType.PICK.value
            }),
            root=ROOT,
        )
        report = PlanValidator(registry).validate(base_plan([
            ActionStep(id="step_001", type=ActionType.PICK, object="a"),
            ActionStep(id="step_002", type=ActionType.PRESS, target="plate"),
        ]))
        self.assertFalse(report.valid)
        self.assertIn("ACTION_NOT_REGISTERED", [issue.code for issue in report.issues])
        self.assertEqual(report.step_coverage["step_001"], CoverageLevel.BLOCKED)
        self.assertEqual(report.step_coverage["step_002"], CoverageLevel.SUPPORTED)


if __name__ == "__main__":
    unittest.main()
