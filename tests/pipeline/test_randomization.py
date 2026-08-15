import copy
import unittest

from pydantic import ValidationError

from pipeline.contracts import RandomizationConfig
from pipeline.randomization import derive_episode_seed, resolve_randomization


def valid_config():
    return {
        "seed": 20260813,
        "episodes": 3,
        "reachable_workspace": {
            "center": [0.0, 0.0],
            "semi_axes": [1.0, 1.0],
            "z": [0.7, 0.9],
        },
        "scene": {
            "object_position": {
                "/World/Flask": {
                    "x": [-0.1, 0.1],
                    "y": [-0.1, 0.1],
                    "z": [0.8, 0.8],
                }
            },
            "object_yaw": {},
            "camera_pose": {},
            "lighting": {
                "/World/Light": {
                    "intensity": [800.0, 1200.0],
                    "color_temperature": [4500.0, 6500.0],
                }
            },
            "material": {
                "/World/Table": {
                    "candidates": ["/World/Looks/A", "/World/Looks/B"]
                }
            },
        },
        "physics": {"friction": {}, "mass_scale": {}},
    }


class RandomizationTests(unittest.TestCase):
    def test_randomization_is_deterministic_and_episode_scoped(self):
        config = RandomizationConfig.model_validate(valid_config())
        first = resolve_randomization(config, 1)
        repeated = resolve_randomization(config, 1)
        other = resolve_randomization(config, 2)

        self.assertEqual(first, repeated)
        self.assertEqual(first["episode_seed"], derive_episode_seed(config.seed, 1))
        self.assertNotEqual(first["resolved"], other["resolved"])
        self.assertIn(
            first["resolved"]["material"]["/World/Table"]["material_path"],
            config.scene.material["/World/Table"].candidates,
        )
        self.assertTrue(800.0 <= first["resolved"]["lighting"]["/World/Light"]["intensity"] <= 1200.0)

    def test_randomization_rejects_invalid_configuration(self):
        for mutation in ("unknown", "reversed", "unreachable"):
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(valid_config())
                if mutation == "unknown":
                    data["scene"]["surprise"] = {}
                elif mutation == "reversed":
                    data["scene"]["object_position"]["/World/Flask"]["x"] = [
                        0.2,
                        -0.2,
                    ]
                else:
                    data["scene"]["object_position"]["/World/Flask"]["x"] = [
                        0.9,
                        1.1,
                    ]

                with self.assertRaises(ValidationError):
                    RandomizationConfig.model_validate(data)


if __name__ == "__main__":
    unittest.main()
