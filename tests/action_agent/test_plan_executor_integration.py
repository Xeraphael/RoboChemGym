import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXTERNAL_STUB_BOOTSTRAP = r'''
import importlib.abc
import importlib.util
import sys
import types

PREFIXES = ("omni", "pxr", "isaacsim", "carb")


class StubMeta(type):
    def __getattr__(cls, name):
        return cls


class ExternalStub(metaclass=StubMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return type(self)()

    def __getattr__(self, name):
        return type(self)

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False

    def __rtruediv__(self, other):
        return other


class ExternalModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return ExternalStub


class ExternalLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = ExternalModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        return None


class ExternalFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in PREFIXES or fullname.startswith(
            tuple(prefix + "." for prefix in PREFIXES)
        ):
            return importlib.util.spec_from_loader(
                fullname,
                ExternalLoader(),
                is_package=True,
            )
        return None


sys.meta_path.insert(0, ExternalFinder())
'''


def run_with_external_stubs(body):
    script = EXTERNAL_STUB_BOOTSTRAP + "\n" + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class PlanExecutorPackageIntegrationTests(unittest.TestCase):
    def test_real_factory_import_has_no_plan_executor_rating_cycle(self):
        result = run_with_external_stubs(
            """
            import factories.controller_factory as factory

            assert "plan_executor" in factory._controller_registry
            assert factory._controller_registry["plan_executor"].__name__ == (
                "PlanExecutorController"
            )
            """
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runner_cleanup_failure_does_not_mutate_real_base_lifecycle(self):
        result = run_with_external_stubs(
            """
            from controllers.plan_executor import PlanExecutorController


            class Runner:
                def __init__(self):
                    self.calls = 0

                def reset(self):
                    self.calls += 1
                    if self.calls == 1:
                        raise RuntimeError("adapter cleanup failed")


            class Recorder:
                def __init__(self):
                    self.calls = 0

                def reset(self):
                    self.calls += 1


            class Collector:
                def __init__(self):
                    self.clear_calls = 0

                def clear_cache(self):
                    self.clear_calls += 1


            controller = PlanExecutorController.__new__(PlanExecutorController)
            controller.runner = Runner()
            controller.trajectory_recorder = Recorder()
            controller.data_collector = Collector()
            controller.mode = "collect"
            controller._episode_num = 3
            controller.success_count = 1
            controller._last_success = True
            controller.check_success_counter = 7
            controller.reset_needed = True
            controller._terminal_persisted = True

            try:
                controller.reset()
            except RuntimeError as exc:
                assert str(exc) == "adapter cleanup failed"
            else:
                raise AssertionError("runner reset failure must propagate")

            assert controller._episode_num == 3
            assert controller.success_count == 1
            assert controller._last_success is True
            assert controller.check_success_counter == 7
            assert controller.reset_needed is True
            assert controller.data_collector.clear_calls == 0
            assert controller.trajectory_recorder.calls == 0
            assert controller._terminal_persisted is True

            controller.reset()

            assert controller.runner.calls == 2
            assert controller._episode_num == 4
            assert controller.success_count == 2
            assert controller._last_success is False
            assert controller.check_success_counter == 0
            assert controller.reset_needed is False
            assert controller.data_collector.clear_calls == 1
            assert controller.trajectory_recorder.calls == 1
            assert controller._terminal_persisted is False
            """
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_registered_controller_constructs_with_live_internal_signatures(self):
        result = run_with_external_stubs(
            """
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from types import SimpleNamespace

            from agent.planning.models import (
                AgentPlan,
                AnnotationStatus,
                ScenePlan,
                SemanticAnnotation,
            )
            from agent.planning.registry import CapabilityRegistry
            from agent.planning.validator import PlanValidator
            import factories.controller_factory as factory
            from utils.object_utils import ObjectUtils

            root = Path.cwd()
            ObjectUtils.get_instance(stage=object())
            plan = AgentPlan(
                plan_id="live_constructor",
                scene=ScenePlan(objects=[]),
                actions=[],
                semantic_annotations=[
                    SemanticAnnotation(
                        source_text="observe an unavailable property",
                        status=AnnotationStatus.NOT_OBSERVABLE,
                        reason="state is not measurable",
                    )
                ],
            )
            report = PlanValidator(
                CapabilityRegistry.load_default(root)
            ).validate(plan)
            assert report.valid

            with TemporaryDirectory() as temporary_directory:
                directory = Path(temporary_directory)
                plan_path = directory / "plan.json"
                validation_path = directory / "validation.json"
                plan_path.write_text(
                    plan.model_dump_json(indent=2), encoding="utf-8"
                )
                validation_path.write_text(
                    report.model_dump_json(indent=2), encoding="utf-8"
                )
                cfg = SimpleNamespace(
                    agent=SimpleNamespace(
                        plan_path=str(plan_path),
                        validation_report_path=str(validation_path),
                        execution_report_path=str(directory / "execution.json"),
                        trajectory_path=str(directory / "trajectory.json"),
                    )
                )
                robot = SimpleNamespace(gripper=object())

                controller = factory.create_controller(
                    "plan_executor",
                    cfg=cfg,
                    robot=robot,
                )

            assert type(controller).__module__ == "controllers.plan_executor"
            assert type(controller).__mro__[1].__module__ == (
                "controllers.base_controller"
            )
            expected_atomic_modules = {
                "pick": "controllers.atomic_actions.pick_controller",
                "place": "controllers.atomic_actions.place_controller",
                "pour": "controllers.atomic_actions.pour_controller",
                "press": "controllers.atomic_actions.press_controller",
                "press_z": "controllers.atomic_actions.pressZ_controller",
                "shake": "controllers.atomic_actions.shake_controller",
                "open": "controllers.atomic_actions.open_controller",
                "close": "controllers.atomic_actions.close_controller",
            }
            assert set(controller.runner.adapters) == set(expected_atomic_modules)
            for action_name, module_name in expected_atomic_modules.items():
                adapter = controller.runner.adapters[action_name]
                assert type(adapter.controller).__module__ == module_name
            assert type(controller.trajectory_recorder).__module__ == (
                "agent.action.rating.trajectory_recorder"
            )
            """
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
