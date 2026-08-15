import inspect
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_position_optimizer_module():
    module_name = "_position_optimizer_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "agent/scene/optimization/position_optimizer.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def fake_package(name):
    module = fake_module(name)
    module.__path__ = []
    return module


class LegacySceneBackendMappingTests(unittest.TestCase):
    def test_equipment_requests_use_registry_paths_instance_names_and_physics_mode(self):
        from agent.scene.legacy_scene_backend import LegacySceneBackend
        from agent.scene.scene_compiler import ResolvedSceneObject

        objects = [ResolvedSceneObject(
            id="solid_flask",
            asset_id="ErlenmeyerFlask",
            instance_name="ErlenmeyerFlask_Solid1",
            category="container",
            usd_path=(
                "Instruments/InteractiveAssets/ErlenmeyerFlask/"
                "ErlenmeyerFlask_01/ErlenmeyerFlask_01.usd"
            ),
            supported_actions=["pick", "place", "pour"],
            required_anchors={},
        )]

        requests = LegacySceneBackend.equipment_requests(objects)

        self.assertEqual(requests, [(
            "ErlenmeyerFlask_Solid1",
            Path(
                "Instruments/InteractiveAssets/ErlenmeyerFlask/"
                "ErlenmeyerFlask_01/ErlenmeyerFlask_01.usd"
            ),
            True,
        )])

    def test_import_does_not_load_simulation_or_usd_modules(self):
        code = """
import sys
import agent.scene.legacy_scene_backend
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

    def test_build_uses_public_reference_scene(self):
        from agent.scene.legacy_scene_backend import LegacySceneBackend
        from agent.scene.scene_compiler import ResolvedSceneObject

        calls = {}

        class FakeExtractor:
            def __init__(self, scenes_dir):
                calls["extractor_init"] = scenes_dir

            def extract(self, filename):
                calls["extract"] = filename
                return output_dir / "extracted.json"

        class FakeOptimizer:
            @classmethod
            def from_profile(cls, json_path, profile):
                calls["from_profile"] = (json_path, profile)
                return cls()

            def optimize(self, **kwargs):
                calls["optimize"] = kwargs
                return {"/World/Flask": [0.0, 0.0, 1.0]}

            def save_optimized_positions(self, positions, output_path):
                calls["save"] = (positions, output_path)

        class FakeUpdater:
            def __init__(self, scenes_dir):
                calls["updater_init"] = scenes_dir

            def apply_positions_to_usd(self, *args, **kwargs):
                calls["update"] = (args, kwargs)
                return True

        modules = {
            "agent.scene.extractor": fake_package("agent.scene.extractor"),
            "agent.scene.extractor.scene_extractor": fake_module(
                "agent.scene.extractor.scene_extractor",
                SceneExtractor=FakeExtractor,
            ),
            "agent.scene.optimization": fake_package("agent.scene.optimization"),
            "agent.scene.optimization.position_optimizer": fake_module(
                "agent.scene.optimization.position_optimizer",
                PositionOptimizer=FakeOptimizer,
            ),
            "agent.scene.optimization.position_updater": fake_module(
                "agent.scene.optimization.position_updater",
                PositionUpdater=FakeUpdater,
            ),
        }
        objects = [ResolvedSceneObject(
            id="flask",
            asset_id="ErlenmeyerFlask",
            instance_name="Flask",
            category="container",
            usd_path="Instruments/flask.usd",
            supported_actions=["pick"],
            required_anchors={},
        )]
        profile = {"grid_resolution": "0.125"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            output_dir = Path(tmp) / "run"
            output_dir.mkdir()
            reference_scene = root / "protocols/example_protocol/scene.usd"
            reference_scene.parent.mkdir(parents=True)
            reference_scene.write_bytes(b"reference-usd")
            output_usd = output_dir / "scene.usd"
            output_json = output_dir / "scene.json"
            backend = LegacySceneBackend(root)

            with patch.dict(sys.modules, modules):
                backend.build(
                    objects,
                    output_usd=output_usd,
                    output_json=output_json,
                    layout_profile=profile,
                )

                self.assertEqual(output_usd.read_bytes(), b"reference-usd")

        self.assertEqual(calls["extractor_init"], str(output_dir))
        self.assertEqual(calls["extract"], "scene.usd")
        self.assertEqual(calls["from_profile"], (output_dir / "extracted.json", profile))
        self.assertEqual(calls["optimize"], {"grid_resolution": 0.125})
        self.assertEqual(
            calls["save"],
            ({"/World/Flask": [0.0, 0.0, 1.0]}, output_json),
        )
        self.assertEqual(calls["updater_init"], str(output_dir))
        self.assertEqual(
            calls["update"],
            ((output_json, output_usd), {"in_place": True}),
        )

    def test_preflight_resolves_required_anchors_with_instance_proxy_traversal(self):
        from agent.scene.legacy_scene_backend import LegacySceneBackend
        from agent.scene.scene_compiler import ResolvedSceneObject

        class FakePath:
            def __init__(self, value):
                self.pathString = value

        class FakePrim:
            def __init__(self, path, valid=True):
                self.path = path
                self.valid = valid
                self.subtree = [self]

            def __bool__(self):
                return self.valid

            def IsValid(self):
                return self.valid

            def GetName(self):
                return self.path.rsplit("/", 1)[-1]

            def GetPath(self):
                return FakePath(self.path)

        instance = FakePrim("/World/TubeRack")
        direct = FakePrim("/World/TubeRack/direct_anchor")
        nested = FakePrim("/World/TubeRack/tube_rack/place_position")
        second_nested = FakePrim("/World/TubeRack/other/place_position")
        instance.subtree = [instance, direct, nested, second_nested]
        outside = FakePrim("/World/Other/place_position")
        prims = {
            prim.path: prim
            for prim in [instance, direct, nested, second_nested, outside]
        }

        class FakeStage:
            def GetPrimAtPath(self, path):
                return prims.get(path, FakePrim(path, valid=False))

        stage = FakeStage()

        class FakeStageType:
            LoadAll = object()

            @staticmethod
            def Open(path, load):
                return stage

        proxy_predicate = object()
        prim_range_calls = []

        def prim_range(prim, predicate=None):
            prim_range_calls.append((prim, predicate))
            return prim.subtree

        fake_usd = types.SimpleNamespace(
            Stage=FakeStageType,
            PrimRange=prim_range,
            TraverseInstanceProxies=lambda: proxy_predicate,
        )

        class FakeOptimizer:
            scene_data = {"/World/TubeRack": {"prim_name": "TubeRack"}}
            objects = []

            @classmethod
            def from_profile(cls, json_path, profile):
                return cls()

        modules = {
            "pxr": fake_module("pxr", Usd=fake_usd),
            "numpy": fake_module("numpy", asarray=lambda value, dtype=None: value),
            "agent.scene.optimization": fake_package("agent.scene.optimization"),
            "agent.scene.optimization.position_optimizer": fake_module(
                "agent.scene.optimization.position_optimizer",
                PositionOptimizer=FakeOptimizer,
            ),
        }
        valid_object = ResolvedSceneObject(
            id="rack",
            asset_id="TubeRack",
            instance_name="TubeRack",
            category="placement_target",
            usd_path="Instruments/tube_rack.usd",
            supported_actions=["place_target"],
            required_anchors={
                "place_target": [
                    "direct_anchor",
                    "tube_rack/place_position",
                ],
            },
            required_capabilities=["place_target"],
        )
        missing_object = valid_object.model_copy(update={
            "required_anchors": {
                "place_target": ["outside_only", "wrong/place_position"]
            },
        })
        ambiguous_object = valid_object.model_copy(update={
            "required_anchors": {"place_target": ["place_position"]},
        })
        prims["/World/Other/outside_only"] = FakePrim("/World/Other/outside_only")
        backend = LegacySceneBackend(ROOT)

        with tempfile.TemporaryDirectory() as tmp:
            scene_json_path = Path(tmp) / "scene.json"
            scene_json_path.write_text(
                json.dumps({
                    "/World/TubeRack": {
                        "prim_name": "TubeRack",
                        "position": [0.0, 0.0, 1.0],
                    }
                }),
                encoding="utf-8",
            )
            with patch.dict(sys.modules, modules):
                valid_report = backend.preflight(
                    [valid_object],
                    usd_path=ROOT / "scene.usd",
                    scene_json_path=scene_json_path,
                    layout_profile={},
                )
                missing_report = backend.preflight(
                    [missing_object],
                    usd_path=ROOT / "scene.usd",
                    scene_json_path=scene_json_path,
                    layout_profile={},
                )
                ambiguous_report = backend.preflight(
                    [ambiguous_object],
                    usd_path=ROOT / "scene.usd",
                    scene_json_path=scene_json_path,
                    layout_profile={},
                )

        self.assertTrue(valid_report.passed)
        self.assertEqual(valid_report.issues, ())
        self.assertFalse(missing_report.passed)
        self.assertEqual(
            [issue.code for issue in missing_report.issues],
            ["MISSING_ANCHOR", "MISSING_ANCHOR"],
        )
        self.assertFalse(ambiguous_report.passed)
        self.assertEqual(
            [issue.code for issue in ambiguous_report.issues],
            ["AMBIGUOUS_ANCHOR"],
        )
        self.assertTrue(prim_range_calls)
        self.assertTrue(all(prim is instance for prim, _ in prim_range_calls))
        self.assertTrue(
            all(predicate is proxy_predicate for _, predicate in prim_range_calls)
        )

    def test_preflight_rejects_layout_below_profile_minimum_spacing(self):
        from agent.scene.legacy_scene_backend import LegacySceneBackend
        from agent.scene.scene_compiler import ResolvedSceneObject

        class FakePrim:
            def __bool__(self):
                return True

            def IsValid(self):
                return True

        class FakeStage:
            def GetPrimAtPath(self, path):
                return FakePrim()

        class FakeStageType:
            LoadAll = object()

            @staticmethod
            def Open(path, load):
                return FakeStage()

        class FakeOptimizer:
            scene_data = {
                "/World/A": {"prim_name": "A"},
                "/World/B": {"prim_name": "B"},
            }
            objects = [
                types.SimpleNamespace(prim_name="A", position=[0.0, 0.0, 0.0]),
                types.SimpleNamespace(prim_name="B", position=[0.0, 0.5, 0.0]),
            ]
            ellipse_constraint = types.SimpleNamespace(
                contains=lambda x, y: True
            )
            z_height = 0.0
            minimum_spacing = 1.0

            @classmethod
            def from_profile(cls, json_path, profile):
                return cls()

            @staticmethod
            def check_collision(first, first_position, second, second_position):
                return False

        modules = {
            "pxr": fake_module(
                "pxr",
                Usd=types.SimpleNamespace(Stage=FakeStageType),
            ),
            "agent.scene.optimization": fake_package("agent.scene.optimization"),
            "agent.scene.optimization.position_optimizer": fake_module(
                "agent.scene.optimization.position_optimizer",
                PositionOptimizer=FakeOptimizer,
            ),
        }
        objects = [
            ResolvedSceneObject(
                id=name.lower(),
                asset_id=name,
                instance_name=name,
                category="container",
                usd_path=f"Instruments/{name}.usd",
                supported_actions=["pick"],
                required_anchors={},
            )
            for name in ("A", "B")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            scene_json_path = Path(tmp) / "scene.json"
            scene_json_path.write_text(
                json.dumps(
                    {
                        f"/World/{name}": {
                            "prim_name": name,
                            "position": [0.0, index * 0.5, 0.0],
                        }
                        for index, name in enumerate(("A", "B"))
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(sys.modules, modules):
                report = LegacySceneBackend(ROOT).preflight(
                    objects,
                    usd_path=ROOT / "scene.usd",
                    scene_json_path=scene_json_path,
                    layout_profile={"minimum_spacing": 1.0},
                )

        self.assertFalse(report.passed)
        self.assertEqual(
            [issue.code for issue in report.issues],
            ["INSUFFICIENT_SPACING"],
        )

    def test_preflight_reports_invalid_scene_json_and_retains_prior_issues(self):
        from agent.scene.legacy_scene_backend import LegacySceneBackend
        from agent.scene.scene_compiler import ResolvedSceneObject

        class FakePrim:
            def __bool__(self):
                return False

            def IsValid(self):
                return False

        class FakeStage:
            def GetPrimAtPath(self, path):
                return FakePrim()

        stage = FakeStage()

        class FakeStageType:
            LoadAll = object()

            @staticmethod
            def Open(path, load):
                return stage

        optimizer_calls = []

        class FakeOptimizer:
            scene_data = {"/World/Flask": {"prim_name": "Flask"}}
            objects = []

            @classmethod
            def from_profile(cls, json_path, profile):
                optimizer_calls.append((json_path, profile))
                return cls()

        modules = {
            "pxr": fake_module(
                "pxr",
                Usd=types.SimpleNamespace(Stage=FakeStageType),
            ),
            "numpy": fake_module("numpy", asarray=lambda value, dtype=None: value),
            "agent.scene.optimization": fake_package("agent.scene.optimization"),
            "agent.scene.optimization.position_optimizer": fake_module(
                "agent.scene.optimization.position_optimizer",
                PositionOptimizer=FakeOptimizer,
            ),
        }
        scene_object = ResolvedSceneObject(
            id="flask",
            asset_id="ErlenmeyerFlask",
            instance_name="Flask",
            category="container",
            usd_path="Instruments/flask.usd",
            supported_actions=["pick"],
            required_anchors={},
        )
        invalid_documents = [
            ("missing", "missing", None),
            ("unreadable", "directory", None),
            ("invalid_utf8", "bytes", b"\xff"),
            ("malformed", "text", "{"),
            ("non_object_root", "text", "[]"),
            ("non_dict_entry", "text", '{"/World/Flask": []}'),
            (
                "missing_prim_name",
                "text",
                '{"/World/Flask": {"position": [0, 0, 1]}}',
            ),
            (
                "non_string_prim_name",
                "text",
                '{"/World/Flask": {"prim_name": 7, "position": [0, 0, 1]}}',
            ),
            (
                "missing_position",
                "text",
                '{"/World/Flask": {"prim_name": "Flask"}}',
            ),
            (
                "short_position",
                "text",
                (
                    '{"/World/Flask": {"prim_name": "Flask", '
                    '"position": [0, 1]}}'
                ),
            ),
            (
                "non_numeric_position",
                "text",
                (
                    '{"/World/Flask": {"prim_name": "Flask", '
                    '"position": [0, "bad", 1]}}'
                ),
            ),
            (
                "non_finite_position",
                "text",
                (
                    '{"/World/Flask": {"prim_name": "Flask", '
                    '"position": [0, NaN, 1]}}'
                ),
            ),
            (
                "non_dict_bounding_box",
                "text",
                (
                    '{"/World/Flask": {"prim_name": "Flask", '
                    '"position": [0, 0, 1], "bounding_box": []}}'
                ),
            ),
            (
                "short_bounding_box_size",
                "text",
                (
                    '{"/World/Flask": {"prim_name": "Flask", '
                    '"position": [0, 0, 1], '
                    '"bounding_box": {"size": [1, 2]}}}'
                ),
            ),
            (
                "non_numeric_bounding_box_size",
                "text",
                (
                    '{"/World/Flask": {"prim_name": "Flask", '
                    '"position": [0, 0, 1], '
                    '"bounding_box": {"size": [1, "bad", 3]}}}'
                ),
            ),
            (
                "non_finite_bounding_box_size",
                "text",
                (
                    '{"/World/Flask": {"prim_name": "Flask", '
                    '"position": [0, 0, 1], '
                    '"bounding_box": {"size": [1, NaN, 3]}}}'
                ),
            ),
        ]

        backend = LegacySceneBackend(ROOT)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            for name, document_type, payload in invalid_documents:
                with self.subTest(document=name):
                    scene_json_path = Path(tmp) / f"{name}.json"
                    if document_type == "directory":
                        scene_json_path.mkdir()
                    elif document_type == "bytes":
                        scene_json_path.write_bytes(payload)
                    elif document_type == "text":
                        scene_json_path.write_text(payload, encoding="utf-8")

                    report = backend.preflight(
                        [scene_object],
                        usd_path=ROOT / "scene.usd",
                        scene_json_path=scene_json_path,
                        layout_profile={},
                    )

                    self.assertFalse(report.passed)
                    self.assertEqual(
                        [issue.code for issue in report.issues],
                        ["MISSING_PRIM", "INVALID_SCENE_JSON"],
                    )
                    self.assertIsInstance(report.issues, tuple)
                    with self.assertRaises(AttributeError):
                        report.issues.append(report.issues[0])

        self.assertEqual(optimizer_calls, [])

    def test_preflight_propagates_optimizer_errors_after_valid_scene_json(self):
        from agent.scene.legacy_scene_backend import LegacySceneBackend
        from agent.scene.scene_compiler import ResolvedSceneObject

        class FakePrim:
            def __bool__(self):
                return False

            def IsValid(self):
                return False

        class FakeStage:
            def GetPrimAtPath(self, path):
                return FakePrim()

        class FakeStageType:
            LoadAll = object()

            @staticmethod
            def Open(path, load):
                return FakeStage()

        class FakeOptimizer:
            @classmethod
            def from_profile(cls, json_path, profile):
                raise ValueError("invalid layout profile")

        modules = {
            "pxr": fake_module(
                "pxr",
                Usd=types.SimpleNamespace(Stage=FakeStageType),
            ),
            "numpy": fake_module("numpy", asarray=lambda value, dtype=None: value),
            "agent.scene.optimization": fake_package("agent.scene.optimization"),
            "agent.scene.optimization.position_optimizer": fake_module(
                "agent.scene.optimization.position_optimizer",
                PositionOptimizer=FakeOptimizer,
            ),
        }
        scene_object = ResolvedSceneObject(
            id="flask",
            asset_id="ErlenmeyerFlask",
            instance_name="Flask",
            category="container",
            usd_path="Instruments/flask.usd",
            supported_actions=["pick"],
            required_anchors={},
        )

        with tempfile.TemporaryDirectory() as tmp:
            scene_json_path = Path(tmp) / "scene.json"
            scene_json_path.write_text(
                json.dumps({
                    "/World/Flask": {
                        "prim_name": "Flask",
                        "position": [0.0, 0.0, 1.0],
                        "bounding_box": {"size": [0.1, 0.1, 0.2]},
                    }
                }),
                encoding="utf-8",
            )
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, "invalid layout profile"):
                    LegacySceneBackend(ROOT).preflight(
                        [scene_object],
                        usd_path=ROOT / "scene.usd",
                        scene_json_path=scene_json_path,
                        layout_profile={},
                    )


class SceneGeneratorAdapterTests(unittest.TestCase):
    def _generator(self, root: Path):
        from agent.scene.generation.scene_generator import SceneGenerator

        instruments = root / "Instruments"
        scene_info = root / "scene_information"
        output = root / "scenes"
        instruments.mkdir()
        scene_info.mkdir()
        base_usd = root / "base.usd"
        base_usd.write_text("base", encoding="utf-8")
        return SceneGenerator(
            instruments_dir=str(instruments),
            base_usd_path=str(base_usd),
            scene_info_dir=str(scene_info),
            output_dir=str(output),
        )

    def _fake_usd_runtime(self, occupied_paths):
        class FakePrim:
            def __init__(self, valid):
                self.valid = valid

            def __bool__(self):
                return self.valid

            def IsValid(self):
                return self.valid

        class FakeLayer:
            def Export(self, path):
                pass

        class FakeStage:
            def GetPrimAtPath(self, path):
                return FakePrim(path == "/World" or path in occupied_paths)

            def Load(self):
                pass

            def Flatten(self):
                return FakeLayer()

        stage = FakeStage()

        class FakeStageType:
            @staticmethod
            def Open(path):
                return stage

        fake_usd = types.SimpleNamespace(Stage=FakeStageType)
        fake_usd_geom = types.SimpleNamespace(
            GetStageMetersPerUnit=lambda stage: 1.0,
            SetStageMetersPerUnit=lambda stage, value: None,
        )
        fake_sdf = types.SimpleNamespace(Path=lambda value: value)
        return fake_usd, fake_usd_geom, fake_sdf

    def test_generate_scene_from_assets_rejects_duplicate_names_deterministically(self):
        from agent.scene.generation import scene_generator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generator = self._generator(root)
            asset = root / "asset.usd"
            asset.write_text("asset", encoding="utf-8")
            fake_usd, fake_usd_geom, fake_sdf = self._fake_usd_runtime(set())
            additions = []

            with (
                patch.object(scene_generator, "Usd", fake_usd, create=True),
                patch.object(
                    scene_generator,
                    "UsdGeom",
                    fake_usd_geom,
                    create=True,
                ),
                patch.object(scene_generator, "Sdf", fake_sdf, create=True),
                patch.object(
                    generator,
                    "add_reference_to_stage",
                    side_effect=lambda *args, **kwargs: additions.append(args[2]) or True,
                ),
                patch.object(generator, "add_physics_properties"),
            ):
                with self.assertRaises(ValueError) as raised:
                    generator.generate_scene_from_assets(
                        [
                            ("Zeta", asset, True),
                            ("Alpha", asset, False),
                            ("Zeta", asset, False),
                            ("Alpha", asset, True),
                        ],
                        output_filename="scene.usd",
                    )

            self.assertEqual(
                str(raised.exception),
                "scene instance conflicts: duplicate instance names: ['Alpha', 'Zeta']",
            )
            self.assertEqual(additions, [])

    def test_generate_scene_from_assets_rejects_occupied_paths_before_any_addition(self):
        from agent.scene.generation import scene_generator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generator = self._generator(root)
            asset = root / "asset.usd"
            asset.write_text("asset", encoding="utf-8")
            fake_usd, fake_usd_geom, fake_sdf = self._fake_usd_runtime(
                {"/World/Alpha", "/World/Zeta"}
            )
            additions = []

            with (
                patch.object(scene_generator, "Usd", fake_usd, create=True),
                patch.object(
                    scene_generator,
                    "UsdGeom",
                    fake_usd_geom,
                    create=True,
                ),
                patch.object(scene_generator, "Sdf", fake_sdf, create=True),
                patch.object(
                    generator,
                    "add_reference_to_stage",
                    side_effect=lambda *args, **kwargs: additions.append(args[2]) or True,
                ),
                patch.object(generator, "add_physics_properties"),
            ):
                with self.assertRaises(ValueError) as raised:
                    generator.generate_scene_from_assets(
                        [
                            ("Free", asset, True),
                            ("Zeta", asset, False),
                            ("Alpha", asset, True),
                        ],
                        output_filename="scene.usd",
                    )

            self.assertEqual(
                str(raised.exception),
                (
                    "scene instance conflicts: occupied prim paths: "
                    "['/World/Alpha', '/World/Zeta']"
                ),
            )
            self.assertEqual(additions, [])

    def test_generate_scene_from_assets_lists_every_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = self._generator(Path(tmp))
            first = Path(tmp) / "missing-a.usd"
            second = Path(tmp) / "missing-b.usd"

            with self.assertRaises(FileNotFoundError) as raised:
                generator.generate_scene_from_assets(
                    [("A", first, True), ("B", second, False)],
                    output_filename="scene.usd",
                )

            self.assertIn(str(first), str(raised.exception))
            self.assertIn(str(second), str(raised.exception))

    def test_generate_scene_remains_a_legacy_file_wrapper(self):
        from agent.scene.generation.scene_generator import SceneGenerator

        generator = SceneGenerator.__new__(SceneGenerator)
        generator.read_scene_information = lambda _: (["Flask1"], ["Flask"])
        generator.find_usd_file = lambda _: Path("resolved.usd")
        generator.is_interactive_asset = lambda _: True
        calls = []
        generator.generate_scene_from_assets = (
            lambda equipment, output_filename: calls.append((equipment, output_filename)) or Path("scene.usd")
        )

        result = generator.generate_scene("legacy.txt", "custom.usd")

        self.assertEqual(result, Path("scene.usd"))
        self.assertEqual(
            calls,
            [([("Flask1", Path("resolved.usd"), True)], "custom.usd")],
        )

    def test_module_convenience_defaults_are_repository_relative(self):
        from agent.scene.generation import scene_generator

        with patch.object(scene_generator, "SceneGenerator") as generator_type:
            generator_type.return_value.generate_scene.return_value = Path("scene.usd")
            scene_generator.generate_scene("legacy.txt")

        generator_type.assert_called_once_with(
            instruments_dir=str(ROOT / "Instruments"),
            base_usd_path=str(ROOT / "protocols/example_protocol/scene.usd"),
            scene_info_dir=str(ROOT / "agent/protocol/scene_information"),
            output_dir=str(ROOT / "agent/scene/scenes"),
        )


class PositionOptimizerProfileTests(unittest.TestCase):
    def test_from_profile_maps_all_layout_values(self):
        PositionOptimizer = load_position_optimizer_module().PositionOptimizer

        profile = {
            "surface_z": 1.25,
            "grid_resolution": 0.125,
            "minimum_spacing": 0.075,
            "reachable_region": {
                "type": "ellipse",
                "center": [2.0, 3.0],
                "semi_axes": [4.0, 5.0],
                "rotation": 0.75,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            scene_json = Path(tmp) / "scene.json"
            scene_json.write_text("{}", encoding="utf-8")

            optimizer = PositionOptimizer.from_profile(scene_json, profile)

        self.assertEqual(optimizer.z_height, 1.25)
        self.assertEqual(optimizer.minimum_spacing, 0.075)
        self.assertEqual(optimizer.ellipse_constraint.center_x, 2.0)
        self.assertEqual(optimizer.ellipse_constraint.center_y, 3.0)
        self.assertEqual(optimizer.ellipse_constraint.semi_major, 4.0)
        self.assertEqual(optimizer.ellipse_constraint.semi_minor, 5.0)
        self.assertEqual(optimizer.ellipse_constraint.rotation, 0.75)

    def test_layout_constructor_values_have_no_numeric_defaults(self):
        module = load_position_optimizer_module()
        EllipseConstraint = module.EllipseConstraint
        PositionOptimizer = module.PositionOptimizer

        optimizer_parameters = inspect.signature(PositionOptimizer).parameters
        ellipse_parameters = inspect.signature(EllipseConstraint).parameters
        optimize_parameters = inspect.signature(PositionOptimizer.optimize_milp).parameters

        self.assertIs(optimizer_parameters["z_height"].default, inspect.Parameter.empty)
        self.assertIs(optimizer_parameters["minimum_spacing"].default, inspect.Parameter.empty)
        self.assertIs(optimizer_parameters["ellipse_constraint"].default, inspect.Parameter.empty)
        for name in ("center_x", "center_y", "semi_major", "semi_minor", "rotation"):
            self.assertIs(ellipse_parameters[name].default, inspect.Parameter.empty)
        self.assertIs(optimize_parameters["grid_resolution"].default, inspect.Parameter.empty)

    def test_semantic_layout_requires_an_api_key(self):
        PositionOptimizer = load_position_optimizer_module().PositionOptimizer

        profile = {
            "surface_z": 1.0,
            "minimum_spacing": 0.08,
            "reachable_region": {
                "center": [0.0, 0.0],
                "semi_axes": [1.0, 1.0],
                "rotation": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            scene_json = Path(tmp) / "scene.json"
            scene_json.write_text("{}", encoding="utf-8")
            optimizer = PositionOptimizer.from_profile(scene_json, profile)

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    ValueError,
                    "OPENAI_API_KEY or api_key is required for semantic layout optimization",
                ):
                    optimizer.apply_semantic_constraints("keep objects separated")

    def test_semantic_layout_preserves_environment_api_key_fallback_without_logging(self):
        module = load_position_optimizer_module()
        PositionOptimizer = module.PositionOptimizer
        calls = {}
        secret = "task7-environment-api-token"

        class FakeCompletions:
            def create(self, **kwargs):
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content="{}")
                        )
                    ]
                )

        class FakeClient:
            def __init__(self, **kwargs):
                calls["client"] = kwargs
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        profile = {
            "surface_z": 1.0,
            "minimum_spacing": 0.08,
            "reachable_region": {
                "center": [0.0, 0.0],
                "semi_axes": [1.0, 1.0],
                "rotation": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            scene_json = Path(tmp) / "scene.json"
            scene_json.write_text("{}", encoding="utf-8")
            optimizer = PositionOptimizer.from_profile(scene_json, profile)

            output = io.StringIO()
            with (
                patch.object(module, "OpenAI", FakeClient),
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": secret,
                        "ACTION_AGENT_MODEL": "test-model",
                    },
                    clear=True,
                ),
                redirect_stdout(output),
            ):
                optimizer.apply_semantic_constraints(
                    "keep objects separated",
                    api_key=None,
                    base_url=None,
                )

        self.assertEqual(calls["client"], {"api_key": secret})
        self.assertNotIn(secret, output.getvalue())

    def test_layout_uses_profile_minimum_spacing_instead_of_fixed_twenty_cm(self):
        PositionOptimizer = load_position_optimizer_module().PositionOptimizer
        scene = {
            "/World/A": {
                "prim_name": "A",
                "position": [0.0, 0.0, 0.0],
                "bounding_box": {"size": [0.01, 0.01, 0.01]},
            },
            "/World/B": {
                "prim_name": "B",
                "position": [0.0, 0.0, 0.0],
                "bounding_box": {"size": [0.01, 0.01, 0.01]},
            },
        }
        profile = {
            "surface_z": 0.0,
            "minimum_spacing": 0.05,
            "reachable_region": {
                "center": [0.0, 0.0],
                "semi_axes": [0.4, 0.4],
                "rotation": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            scene_json = Path(tmp) / "scene.json"
            scene_json.write_text(json.dumps(scene), encoding="utf-8")
            optimizer = PositionOptimizer.from_profile(scene_json, profile)

            positions = optimizer.optimize(grid_resolution=0.05)

        distance = np.linalg.norm(
            np.asarray(positions["/World/A"][:2])
            - np.asarray(positions["/World/B"][:2])
        )
        self.assertGreaterEqual(distance, 0.05)
        self.assertLessEqual(distance, 0.10)

    def test_layout_minimum_spacing_is_a_hard_constraint_when_feasible(self):
        PositionOptimizer = load_position_optimizer_module().PositionOptimizer
        scene = {
            f"/World/{name}": {
                "prim_name": name,
                "position": [0.0, 0.0, 0.0],
                "bounding_box": {"size": [0.01, 0.01, 0.01]},
            }
            for name in ("A", "B")
        }
        profile = {
            "surface_z": 0.0,
            "minimum_spacing": 1.0,
            "reachable_region": {
                "center": [0.0, 0.0],
                "semi_axes": [2.0, 0.1],
                "rotation": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            scene_json = Path(tmp) / "scene.json"
            scene_json.write_text(json.dumps(scene), encoding="utf-8")
            optimizer = PositionOptimizer.from_profile(scene_json, profile)

            positions = optimizer.optimize(grid_resolution=0.05)

        distance = np.linalg.norm(
            np.asarray(positions["/World/A"][:2])
            - np.asarray(positions["/World/B"][:2])
        )
        self.assertGreaterEqual(distance, 1.0)

    def test_layout_rejects_nonpositive_or_nonfinite_minimum_spacing(self):
        PositionOptimizer = load_position_optimizer_module().PositionOptimizer
        with tempfile.TemporaryDirectory() as tmp:
            scene_json = Path(tmp) / "scene.json"
            scene_json.write_text("{}", encoding="utf-8")
            for value in (0.0, -0.1, float("nan"), float("inf")):
                with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "minimum_spacing"
                ):
                    PositionOptimizer.from_profile(
                        scene_json,
                        {
                            "surface_z": 0.0,
                            "minimum_spacing": value,
                            "reachable_region": {
                                "center": [0.0, 0.0],
                                "semi_axes": [1.0, 1.0],
                                "rotation": 0.0,
                            },
                        },
                    )


class SceneInitializerPlanTests(unittest.TestCase):
    def test_step4_forwards_explicit_credentials_without_logging_them(self):
        from agent.scene.scene_initializer import SceneInitializer

        calls = {}
        secret = "task7-explicit-api-token"
        secret_url = "https://task7-explicit.example.invalid/v1"

        class FakeOptimizer:
            @classmethod
            def from_profile(cls, json_path, profile):
                return cls()

            def optimize(self, **kwargs):
                return {"/World/Flask": [0.0, 0.0, 1.0]}

            def save_optimized_positions(self, positions, output_path):
                pass

            def apply_semantic_constraints(self, **kwargs):
                calls["semantic"] = kwargs
                return {"/World/Flask": [0.0, 0.0, 1.0]}

        modules = {
            "agent.scene.optimization": fake_package("agent.scene.optimization"),
            "agent.scene.optimization.position_optimizer": fake_module(
                "agent.scene.optimization.position_optimizer",
                PositionOptimizer=FakeOptimizer,
                OptimizationMethod=types.SimpleNamespace(MILP=object()),
            ),
        }

        with patch.dict(sys.modules, modules):
            output = io.StringIO()
            with redirect_stdout(output):
                result = SceneInitializer.__new__(
                    SceneInitializer
                ).step4_optimize_positions(
                    "scene.json",
                    semantic_prompt="keep objects separated",
                    api_key=secret,
                    base_url=secret_url,
                )

        self.assertEqual(result, "scene.json")
        self.assertEqual(calls["semantic"]["api_key"], secret)
        self.assertEqual(calls["semantic"]["base_url"], secret_url)
        self.assertNotIn(secret, output.getvalue())
        self.assertNotIn(secret_url, output.getvalue())

    def test_initialize_scene_forwards_credentials_to_step4_without_logging_them(self):
        from agent.scene.scene_initializer import SceneInitializer

        calls = {}
        secret = "task7-initialize-api-token"
        secret_url = "https://task7-initialize.example.invalid/v1"
        initializer = SceneInitializer.__new__(SceneInitializer)
        initializer.step1_extract_protocol = lambda *args, **kwargs: (
            "equipment.txt",
            "actions.txt",
        )
        initializer.step2_generate_scene = lambda equipment_file: "scene.usd"
        initializer.step3_extract_scene = lambda scene_usd: "scene.json"

        def optimize(scene_json, **kwargs):
            calls["step4"] = (scene_json, kwargs)
            return scene_json

        initializer.step4_optimize_positions = optimize
        initializer.step6_generate_yaml = lambda *args: "scene.yaml"

        output = io.StringIO()
        with redirect_stdout(output):
            result = initializer.initialize_scene(
                "mix the sample",
                api_key=secret,
                base_url=secret_url,
                skip_update=True,
                semantic_prompt="keep objects separated",
            )

        self.assertTrue(result.success)
        self.assertEqual(calls["step4"][0], "scene.json")
        self.assertEqual(calls["step4"][1]["api_key"], secret)
        self.assertEqual(calls["step4"][1]["base_url"], secret_url)
        self.assertNotIn(secret, output.getvalue())
        self.assertNotIn(secret_url, output.getvalue())

    def test_defaults_are_repository_relative_and_custom_paths_are_preserved(self):
        from agent.scene import scene_initializer

        source = Path(scene_initializer.__file__).read_text(encoding="utf-8")
        self.assertNotIn("/home/", source)

        initializer_parameters = inspect.signature(
            scene_initializer.SceneInitializer
        ).parameters
        convenience_parameters = inspect.signature(
            scene_initializer.initialize_scene
        ).parameters
        for name in (
            "protocol_dir",
            "scene_info_dir",
            "action_info_dir",
            "scenes_dir",
            "instruments_dir",
            "base_usd_path",
            "config_dir",
        ):
            self.assertIs(initializer_parameters[name].default, None)
        for name in ("protocol_dir", "scenes_dir", "config_dir"):
            self.assertIs(convenience_parameters[name].default, None)

        with patch.object(Path, "mkdir"):
            initializer = scene_initializer.SceneInitializer(
                manage_simulation_app=False
            )
        self.assertEqual(initializer.protocol_dir, ROOT / "agent/protocol")
        self.assertEqual(
            initializer.scene_info_dir,
            ROOT / "agent/protocol/scene_information",
        )
        self.assertEqual(
            initializer.action_info_dir,
            ROOT / "agent/protocol/action_information",
        )
        self.assertEqual(initializer.scenes_dir, ROOT / "agent/scene/scenes")
        self.assertEqual(initializer.instruments_dir, ROOT / "Instruments")
        self.assertEqual(
            initializer.base_usd_path,
            ROOT / "protocols/example_protocol/scene.usd",
        )
        self.assertEqual(initializer.config_dir, ROOT / "config")

        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp)
            with patch.object(Path, "mkdir"):
                overridden = scene_initializer.SceneInitializer(
                    protocol_dir=str(custom / "protocol"),
                    scene_info_dir=str(custom / "scene_info"),
                    action_info_dir=str(custom / "action_info"),
                    scenes_dir=str(custom / "scenes"),
                    instruments_dir=str(custom / "instruments"),
                    base_usd_path=str(custom / "base.usd"),
                    config_dir=str(custom / "config"),
                    manage_simulation_app=False,
                )
            self.assertEqual(overridden.protocol_dir, custom / "protocol")
            self.assertEqual(overridden.scene_info_dir, custom / "scene_info")
            self.assertEqual(overridden.action_info_dir, custom / "action_info")
            self.assertEqual(overridden.scenes_dir, custom / "scenes")
            self.assertEqual(overridden.instruments_dir, custom / "instruments")
            self.assertEqual(overridden.base_usd_path, custom / "base.usd")
            self.assertEqual(overridden.config_dir, custom / "config")

    def test_initialize_plan_wires_backend_compiler_and_compile_arguments(self):
        from agent.scene.scene_initializer import SceneInitializer

        calls = {}

        class FakeBackend:
            def __init__(self, root):
                calls["backend_root"] = root

        class FakeCompiler:
            def __init__(self, registry, backend, root):
                calls["compiler_init"] = (registry, backend, root)

            def compile(self, plan, artifacts):
                calls["compile"] = (plan, artifacts)
                return "compiled"

        modules = {
            "agent.scene.legacy_scene_backend": fake_module(
                "agent.scene.legacy_scene_backend",
                LegacySceneBackend=FakeBackend,
            ),
            "agent.scene.scene_compiler": fake_module(
                "agent.scene.scene_compiler",
                SceneCompiler=FakeCompiler,
            ),
        }
        plan = object()
        artifacts = object()
        registry = object()

        with patch.dict(sys.modules, modules):
            result = SceneInitializer.__new__(SceneInitializer).initialize_plan(
                plan,
                artifacts,
                registry,
            )

        self.assertEqual(result, "compiled")
        self.assertEqual(calls["backend_root"], ROOT)
        compiler_registry, backend, compiler_root = calls["compiler_init"]
        self.assertIs(compiler_registry, registry)
        self.assertIsInstance(backend, FakeBackend)
        self.assertEqual(compiler_root, ROOT)
        self.assertEqual(calls["compile"], (plan, artifacts))


if __name__ == "__main__":
    unittest.main()
