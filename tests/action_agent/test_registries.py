import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from agent.planning.registry import (
    AssetDefinition,
    AssetRegistry,
    ActionRegistry,
    CapabilityRegistry,
    DuplicateAliasError,
    ParameterConstraint,
)


ROOT = Path(__file__).resolve().parents[2]


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = CapabilityRegistry.load_default(ROOT)

    def assert_duplicate_yaml_rejected(self, assets_yaml: str, actions_yaml: str):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry_dir = root / "agent" / "planning" / "registry"
            registry_dir.mkdir(parents=True)
            (registry_dir / "assets.yaml").write_text(assets_yaml, encoding="utf-8")
            (registry_dir / "actions.yaml").write_text(actions_yaml, encoding="utf-8")
            with self.assertRaises(yaml.constructor.ConstructorError):
                CapabilityRegistry.load_default(root)

    def test_example_protocol_assets_resolve_to_existing_files(self):
        solid = self.registry.assets.resolve("ErlenmeyerFlask", {"content_phase": "solid"})
        liquid = self.registry.assets.resolve("ErlenmeyerFlask", {"content_phase": "liquid"})
        self.assertTrue((ROOT / solid.usd_path).is_file())
        self.assertTrue((ROOT / liquid.usd_path).is_file())
        self.assertEqual(solid.usd_path, liquid.usd_path)
        self.assertEqual(solid.usd_path, "protocols/example_protocol/scene.usd")
        self.assertTrue((ROOT / self.registry.assets.get("HeatingPlate").usd_path).is_file())
        self.assertTrue((ROOT / self.registry.assets.get("TargetPlatform").usd_path).is_file())

    def test_aliases_resolve_to_canonical_asset_ids(self):
        self.assertEqual(self.registry.assets.canonical_id("锥形瓶"), "ErlenmeyerFlask")
        self.assertEqual(self.registry.assets.canonical_id("target_plat"), "TargetPlatform")

    def test_target_platform_uses_its_root_transform_as_the_place_target(self):
        target = self.registry.assets.get("TargetPlatform")

        self.assertNotIn("place_target", target.required_anchors)

    def test_action_registry_exposes_adapter_verifier_and_tunable_parameters(self):
        pour = self.registry.actions.get("pour")
        self.assertEqual(pour.adapter, "pour")
        self.assertEqual(pour.verifier, "pour")
        self.assertIn("pour_speed", pour.tunable_parameters)
        pour.validate_parameters({"pour_speed": -1.0}, tunable_only=True)
        with self.assertRaises(ValueError):
            pour.validate_parameters({"pour_speed": 3.0}, tunable_only=True)

    def test_public_manifest_paths_are_repository_relative_and_exist(self):
        for asset_id in (
            "ErlenmeyerFlask",
            "HeatingPlate",
            "TargetPlatform",
            "DryingBox",
        ):
            definition = self.registry.assets.definitions[asset_id]
            for usd_path in definition.all_usd_paths():
                self.assertFalse(Path(usd_path).is_absolute())
                self.assertTrue((ROOT / usd_path).is_file(), usd_path)

    def test_duplicate_aliases_are_rejected(self):
        first = AssetDefinition(aliases=["shared"], category="container", usd_path="a.usd", supported_actions=[])
        second = AssetDefinition(aliases=["shared"], category="container", usd_path="b.usd", supported_actions=[])
        with self.assertRaises(DuplicateAliasError):
            AssetRegistry({"A": first, "B": second})

    def test_asset_and_action_registries_defensively_own_models(self):
        asset_source = AssetDefinition(
            aliases=["asset_alias"],
            category="container",
            usd_path="asset.usd",
            supported_actions=[],
        )
        asset_registry = AssetRegistry({"Asset": asset_source})
        asset_source.aliases.append("source_mutation")
        asset_registry.get("Asset").aliases.append("get_mutation")
        asset_registry.definitions["Asset"].aliases.append(
            "definitions_mutation"
        )

        fresh_asset = asset_registry.get("Asset")
        self.assertEqual(fresh_asset.aliases, ["asset_alias"])

        action_source = deepcopy(self.registry.actions.get("pick"))
        expected_offset = action_source.default_parameters["pre_offset_z"]
        action_registry = ActionRegistry({"pick": action_source})
        action_source.default_parameters["pre_offset_z"] = 0.2
        action_registry.get("pick").default_parameters["pre_offset_z"] = 0.21
        action_registry.definitions["pick"].default_parameters[
            "pre_offset_z"
        ] = 0.22

        fresh_action = action_registry.get("pick")
        self.assertEqual(
            fresh_action.default_parameters["pre_offset_z"], expected_offset
        )

    def test_capability_registry_rejects_invalid_asset_defaults(self):
        actions = ActionRegistry(self.registry.actions.definitions)
        invalid_assets = {
            "unknown variant": AssetDefinition(
                category="container",
                variants={"default": "asset.usd", "known": "known.usd"},
                variant_property="phase",
                supported_actions=["pick"],
                variant_action_defaults={
                    "missing": {
                        "pick": {"orientation_profile": "default"}
                    }
                },
            ),
            "unknown action": AssetDefinition(
                category="container",
                usd_path="asset.usd",
                supported_actions=["pick", "pock"],
            ),
            "unknown parameter": AssetDefinition(
                category="container",
                usd_path="asset.usd",
                supported_actions=["pick"],
                action_defaults={"pick": {"typo_distance": 0.1}},
            ),
            "unsupported capability": AssetDefinition(
                category="container",
                usd_path="asset.usd",
                supported_actions=[],
                action_defaults={
                    "pick": {"orientation_profile": "default"}
                },
            ),
            "wrong object category": AssetDefinition(
                category="door_device",
                usd_path="asset.usd",
                supported_actions=["pick"],
            ),
            "wrong pseudo target category": AssetDefinition(
                category="door_device",
                usd_path="asset.usd",
                supported_actions=["place_target"],
            ),
        }

        for label, asset in invalid_assets.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "Asset"):
                    CapabilityRegistry(
                        AssetRegistry({"Asset": asset}), actions, ROOT
                    )

    def test_pseudo_target_capabilities_without_defaults_remain_valid(self):
        registry = CapabilityRegistry(
            AssetRegistry(
                {
                    "Platform": AssetDefinition(
                        category="placement_target",
                        usd_path="platform.usd",
                        supported_actions=["place_target"],
                    )
                }
            ),
            ActionRegistry(self.registry.actions.definitions),
            ROOT,
        )
        self.assertEqual(
            registry.assets.get("Platform").supported_actions,
            ["place_target"],
        )

    def test_numeric_constraints_reject_non_finite_values(self):
        constraint = ParameterConstraint(kind="number")
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    constraint.validate_value("value", value)

    def test_absolute_asset_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            AssetDefinition(
                category="container",
                usd_path="/absolute/asset.usd",
                supported_actions=[],
            )

    def test_parent_traversal_asset_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            AssetDefinition(
                category="container",
                usd_path="../outside.usd",
                supported_actions=[],
            )

    def test_variant_asset_paths_are_validated(self):
        for unsafe_path in ("/absolute/asset.usd", "../outside.usd", ""):
            with self.subTest(unsafe_path=unsafe_path):
                with self.assertRaises(ValueError):
                    AssetDefinition(
                        category="container",
                        variants={"default": "asset.usd", "unsafe": unsafe_path},
                        variant_property="content_phase",
                        supported_actions=[],
                    )

    def test_duplicate_asset_manifest_keys_are_rejected(self):
        self.assert_duplicate_yaml_rejected(
            assets_yaml="""assets:
  Flask:
    aliases: []
    category: container
    usd_path: first.usd
    usd_path: second.usd
    supported_actions: []
""",
            actions_yaml="actions: {}\n",
        )

    def test_duplicate_action_manifest_keys_are_rejected(self):
        self.assert_duplicate_yaml_rejected(
            assets_yaml="assets: {}\n",
            actions_yaml="""actions:
  press:
    object_categories: []
    target_categories: []
    adapter: press
    adapter: overwritten
    verifier: press
    required_object: false
    required_target: false
    max_frames: 1
    preconditions: []
    effects: []
    supported_parameters: []
    tunable_parameters: []
    parameter_constraints: {}
    degradable_modifiers: []
""",
        )


if __name__ == "__main__":
    unittest.main()
