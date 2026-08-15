import argparse
import ast
import subprocess
import sys
import unittest
from pathlib import Path

from utils.isaacsim_runtime import (
    DRIVER_CHECK_BYPASS_ARG,
    prepare_isaacsim_argv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_LAUNCHERS = (
    "main.py",
)
SIMULATION_APP_CONSTRUCTOR_FILES = {
    *ROOT_LAUNCHERS,
    "agent/action/rating/trajectory_evaluator.py",
    "agent/scene/extractor/scene_extractor.py",
    "agent/scene/generation/scene_generator.py",
    "agent/scene/isaac_scene_worker_main.py",
    "agent/scene/optimization/position_updater.py",
    "agent/scene/scene_initializer.py",
}
SPAWN_ENTRYPOINTS = (
    "_mp_initialize_scene",
    "_mp_optimize_scene",
    "_mp_update_usd",
)
DIRECT_CLI_CASES = {
    "agent/action/rating/trajectory_evaluator.py": (
        [
            "--config-name",
            "demo",
            "--headless",
            "--portable-root",
            "/tmp/kit",
        ],
        {"config_name": "demo", "headless": True},
    ),
    "agent/scene/extractor/scene_extractor.py": (
        [
            "scene.usd",
            "--scenes-dir",
            "/tmp/scenes",
            "--portable-root",
            "/tmp/kit",
        ],
        {"usd_file": "scene.usd", "scenes_dir": "/tmp/scenes"},
    ),
    "agent/scene/generation/scene_generator.py": (
        [
            "protocol_scene.txt",
            "--output",
            "scene.usd",
            "--portable-root",
            "/tmp/kit",
        ],
        {"scene_info_file": "protocol_scene.txt", "output": "scene.usd"},
    ),
    "agent/scene/optimization/position_updater.py": (
        [
            "scene.json",
            "--usd-file",
            "scene.usd",
            "--output-file",
            "updated.usd",
            "--portable-root",
            "/tmp/kit",
        ],
        {
            "json_file": "scene.json",
            "usd_file": "scene.usd",
            "output_file": "updated.usd",
        },
    ),
}


def _parse(relative_path: str) -> ast.Module:
    path = PROJECT_ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]


def _simulation_app_imports(tree: ast.AST) -> list[ast.ImportFrom]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "isaacsim"
        and any(alias.name == "SimulationApp" for alias in node.names)
    ]


def _constructor_files() -> set[str]:
    candidates = list(PROJECT_ROOT.glob("*.py"))
    candidates.extend((PROJECT_ROOT / "agent").rglob("*.py"))

    constructor_files = set()
    for path in candidates:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _calls_named(tree, "SimulationApp"):
            constructor_files.add(str(path.relative_to(PROJECT_ROOT)))
    return constructor_files


def _execution_scope(tree: ast.Module, node: ast.AST) -> ast.AST:
    function_scopes = [
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        and candidate.lineno <= node.lineno <= candidate.end_lineno
    ]
    if not function_scopes:
        return tree
    return min(
        function_scopes,
        key=lambda candidate: candidate.end_lineno - candidate.lineno,
    )


def _parent_links(tree: ast.AST):
    links = {}
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for index, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        links[child] = (parent, field, index)
            elif isinstance(value, ast.AST):
                links[value] = (parent, field, None)
    return links


def _statement_path(tree: ast.Module, scope: ast.AST, node: ast.AST):
    links = _parent_links(tree)
    path = []
    current = node
    while current is not scope:
        parent, field, index = links[current]
        if isinstance(current, ast.stmt):
            path.append((parent, field, index, current))
        current = parent
    path.reverse()
    return path


def _dominates_in_scope(tree: ast.Module, candidate: ast.AST, guarded: ast.AST) -> bool:
    scope = _execution_scope(tree, guarded)
    if _execution_scope(tree, candidate) is not scope:
        return False

    candidate_path = _statement_path(tree, scope, candidate)
    guarded_path = _statement_path(tree, scope, guarded)
    common = 0
    while (
        common < len(candidate_path)
        and common < len(guarded_path)
        and candidate_path[common][:3] == guarded_path[common][:3]
    ):
        common += 1
    if common >= len(candidate_path) or common >= len(guarded_path):
        return False

    candidate_entry = candidate_path[common]
    guarded_entry = guarded_path[common]
    return (
        common == len(candidate_path) - 1
        and candidate_entry[0] is guarded_entry[0]
        and candidate_entry[1] == guarded_entry[1]
        and candidate_entry[2] < guarded_entry[2]
    )


def _load_cli_parser(relative_path: str):
    tree = _parse(relative_path)
    parser_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_cli_args"
    ]
    if len(parser_functions) != 1:
        raise AssertionError(
            f"{relative_path} must define exactly one parse_cli_args function"
        )
    namespace = {
        "__file__": str(PROJECT_ROOT / relative_path),
        "_project_root": PROJECT_ROOT,
        "argparse": argparse,
        "Path": Path,
    }
    parser_module = ast.Module(body=parser_functions, type_ignores=[])
    ast.fix_missing_locations(parser_module)
    exec(compile(parser_module, relative_path, "exec"), namespace)
    return namespace["parse_cli_args"]


class PrepareIsaacSimArgvTests(unittest.TestCase):
    def setUp(self):
        self.original_argv = sys.argv[:]

    def tearDown(self):
        sys.argv[:] = self.original_argv

    def test_adds_driver_check_bypass_by_default(self):
        sys.argv[:] = ["main.py", "--existing-kit-arg"]

        prepared = prepare_isaacsim_argv()

        self.assertEqual(
            prepared,
            ["main.py", "--existing-kit-arg", DRIVER_CHECK_BYPASS_ARG],
        )
        self.assertEqual(sys.argv, prepared)

    def test_is_idempotent(self):
        sys.argv[:] = ["main.py", "--portable-root", "/tmp/kit"]

        prepare_isaacsim_argv()
        prepared = prepare_isaacsim_argv()

        self.assertEqual(prepared.count(DRIVER_CHECK_BYPASS_ARG), 1)

    def test_preserves_explicit_driver_setting_without_duplication(self):
        for value in ("true", "false"):
            with self.subTest(value=value):
                explicit = f"--/rtx/verifyDriverVersion/enabled={value}"
                sys.argv[:] = ["main.py", explicit]

                prepare_isaacsim_argv()
                prepared = prepare_isaacsim_argv()

                self.assertEqual(prepared, ["main.py", explicit])

    def test_replaces_project_args_with_supplied_kit_args(self):
        sys.argv[:] = ["main.py", "--config-name", "test"]

        prepared = prepare_isaacsim_argv(["--portable-root", "/tmp/kit"])

        self.assertEqual(
            prepared,
            ["main.py", "--portable-root", "/tmp/kit", DRIVER_CHECK_BYPASS_ARG],
        )

    def test_accepts_one_shot_iterable_kit_args(self):
        sys.argv[:] = ["main.py", "--config-name", "test"]

        prepared = prepare_isaacsim_argv(
            arg for arg in ("--portable-root", "/tmp/kit")
        )

        self.assertEqual(
            prepared,
            ["main.py", "--portable-root", "/tmp/kit", DRIVER_CHECK_BYPASS_ARG],
        )

    def test_rejects_string_like_kit_args_without_mutating_argv(self):
        for kit_args in ("--portable-root", b"--portable-root"):
            with self.subTest(type=type(kit_args).__name__):
                sys.argv[:] = ["main.py", "--config-name", "test"]
                original = sys.argv[:]

                try:
                    prepare_isaacsim_argv(kit_args)
                except Exception as exc:
                    self.assertIsInstance(exc, TypeError)
                else:
                    self.fail("string-like kit_args must raise TypeError")

                self.assertEqual(sys.argv, original)


class IsaacSimLaunchIntegrationTests(unittest.TestCase):
    def test_all_simulation_app_constructor_files_are_covered(self):
        self.assertEqual(_constructor_files(), SIMULATION_APP_CONSTRUCTOR_FILES)

    def test_prepare_call_precedes_every_simulation_app_import_and_constructor(self):
        for relative_path in sorted(SIMULATION_APP_CONSTRUCTOR_FILES):
            with self.subTest(path=relative_path):
                tree = _parse(relative_path)
                prepare_calls = _calls_named(tree, "prepare_isaacsim_argv")
                imports = _simulation_app_imports(tree)
                constructors = _calls_named(tree, "SimulationApp")

                self.assertTrue(prepare_calls, "missing prepare_isaacsim_argv call")
                self.assertTrue(imports, "missing SimulationApp import")
                self.assertTrue(constructors, "missing SimulationApp constructor")
                for guarded_node in [*imports, *constructors]:
                    self.assertTrue(
                        any(
                            _dominates_in_scope(tree, call, guarded_node)
                            for call in prepare_calls
                        ),
                        "prepare_isaacsim_argv must dominate the guarded node "
                        f"on the same execution path before line {guarded_node.lineno}",
                    )

    def test_execution_path_check_rejects_prepare_in_an_unrelated_branch(self):
        tree = ast.parse(
            "if enabled:\n"
            "    prepare_isaacsim_argv()\n"
            "SimulationApp({})\n"
        )
        prepare_call = _calls_named(tree, "prepare_isaacsim_argv")[0]
        constructor = _calls_named(tree, "SimulationApp")[0]

        self.assertFalse(_dominates_in_scope(tree, prepare_call, constructor))

    def test_root_launchers_forward_unknown_kit_args_before_import(self):
        for relative_path in ROOT_LAUNCHERS:
            with self.subTest(path=relative_path):
                tree = _parse(relative_path)
                parse_function = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
                )
                parser_methods = [
                    node.func.attr
                    for node in ast.walk(parse_function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "parser"
                ]
                self.assertIn("parse_known_args", parser_methods)
                self.assertNotIn("parse_args", parser_methods)

                args_assignment = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Tuple)
                        and [
                            item.id for item in target.elts if isinstance(item, ast.Name)
                        ]
                        == ["args", "kit_args"]
                        for target in node.targets
                    )
                )
                prepare_call = next(
                    call
                    for call in _calls_named(tree, "prepare_isaacsim_argv")
                    if len(call.args) == 1
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id == "kit_args"
                )
                import_line = min(node.lineno for node in _simulation_app_imports(tree))
                constructor_line = min(
                    call.lineno for call in _calls_named(tree, "SimulationApp")
                )

                self.assertLess(args_assignment.lineno, prepare_call.lineno)
                self.assertLess(prepare_call.lineno, import_line)
                self.assertLess(import_line, constructor_line)

    def test_direct_cli_parsers_preserve_project_args_and_isolate_kit_args(self):
        original_argv = sys.argv[:]
        try:
            for relative_path, (argv, expected_project_args) in DIRECT_CLI_CASES.items():
                with self.subTest(path=relative_path):
                    project_args, kit_args = _load_cli_parser(relative_path)(argv)
                    for name, expected in expected_project_args.items():
                        self.assertEqual(getattr(project_args, name), expected)
                    self.assertEqual(
                        kit_args,
                        ["--portable-root", "/tmp/kit"],
                    )

                    sys.argv[:] = [relative_path, *argv]
                    prepared = prepare_isaacsim_argv(kit_args)
                    self.assertEqual(
                        prepared,
                        [
                            relative_path,
                            "--portable-root",
                            "/tmp/kit",
                            DRIVER_CHECK_BYPASS_ARG,
                        ],
                    )
        finally:
            sys.argv[:] = original_argv

    def test_direct_cli_modules_do_not_mutate_argv_at_import_or_parse_twice(self):
        for relative_path in DIRECT_CLI_CASES:
            with self.subTest(path=relative_path):
                tree = _parse(relative_path)
                top_level_prepare_calls = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and _call_name(node.value) == "prepare_isaacsim_argv"
                ]
                strict_parse_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "parse_args"
                ]

                self.assertEqual(top_level_prepare_calls, [])
                self.assertEqual(strict_parse_calls, [])

    def test_direct_cli_main_paths_parse_then_prepare_then_launch(self):
        launch_calls = {
            "agent/action/rating/trajectory_evaluator.py": "evaluate_trajectory",
            "agent/scene/extractor/scene_extractor.py": "_get_or_create_simulation_app",
            "agent/scene/generation/scene_generator.py": "_get_or_create_simulation_app",
            "agent/scene/optimization/position_updater.py": "_get_or_create_simulation_app",
        }
        for relative_path, launch_name in launch_calls.items():
            with self.subTest(path=relative_path):
                tree = _parse(relative_path)
                main_guards = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"
                ]
                parse_call = next(
                    call
                    for guard in main_guards
                    for call in _calls_named(guard, "parse_cli_args")
                )
                prepare_call = next(
                    call
                    for guard in main_guards
                    for call in _calls_named(guard, "prepare_isaacsim_argv")
                )
                launch_call = next(
                    call
                    for guard in main_guards
                    for call in _calls_named(guard, launch_name)
                )

                self.assertTrue(_dominates_in_scope(tree, parse_call, prepare_call))
                self.assertTrue(_dominates_in_scope(tree, prepare_call, launch_call))
                self.assertEqual(len(prepare_call.args), 1)

    def test_trajectory_evaluator_import_preserves_host_argv_and_is_lazy(self):
        script = """
import sys

sys.argv[:] = ["host.py", "--host-project-arg"]
before = sys.argv[:]
import agent.action.rating.trajectory_evaluator
assert sys.argv == before
assert "isaacsim" not in sys.modules
assert not any(name == "omni" or name.startswith("omni.") for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_spawn_entrypoints_prepare_argv_after_reset_and_before_agent_import(self):
        tree = _parse("agent/main.py")

        for function_name in SPAWN_ENTRYPOINTS:
            with self.subTest(function=function_name):
                function = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == function_name
                )
                reset_lines = [
                    node.lineno
                    for node in ast.walk(function)
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "sys"
                        and target.attr == "argv"
                        for target in node.targets
                    )
                ]
                prepare_lines = [
                    call.lineno
                    for call in _calls_named(function, "prepare_isaacsim_argv")
                ]
                agent_import_lines = [
                    node.lineno
                    for node in ast.walk(function)
                    if isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and node.module.startswith("agent.")
                ]

                self.assertEqual(len(reset_lines), 1)
                self.assertEqual(len(prepare_lines), 1)
                self.assertTrue(agent_import_lines)
                self.assertLess(reset_lines[0], prepare_lines[0])
                self.assertLess(prepare_lines[0], min(agent_import_lines))


if __name__ == "__main__":
    unittest.main()
