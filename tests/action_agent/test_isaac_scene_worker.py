import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import call, patch

from agent.scene.isaac_scene_worker import (
    IsaacSubprocessSceneBackend,
    SceneWorkerError,
)
from agent.scene import isaac_scene_worker_main
from agent.scene.scene_compiler import ResolvedSceneObject
from agent.scene.scene_preflight import ScenePreflightIssue, ScenePreflightReport


ROOT = Path(__file__).resolve().parents[2]


def make_object():
    return ResolvedSceneObject(
        id="flask",
        asset_id="ErlenmeyerFlask",
        instance_name="Flask",
        category="container",
        usd_path="Instruments/flask.usd",
        supported_actions=["pick", "place", "pour"],
        required_anchors={"pick": ["grisp_position"]},
        required_capabilities=["pick"],
    )


class RecordingPopen:
    def __init__(
        self,
        response=None,
        *,
        response_text=None,
        returncode=0,
        launch_error=None,
        wait_timeout=False,
        wait_error=None,
        poll_result=None,
    ):
        self.response = response
        self.response_text = response_text
        self.target_returncode = returncode
        self.returncode = None
        self.launch_error = launch_error
        self.wait_timeout = wait_timeout
        self.wait_error = wait_error
        self.poll_result = poll_result
        self.calls = []
        self.wait_calls = []
        self.poll_calls = 0
        self.pid = 424242

    def __call__(self, command, **kwargs):
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        with request_path.open("r", encoding="utf-8") as request_file:
            request = json.load(request_file)
        self.calls.append(
            {
                "command": command,
                "kwargs": kwargs,
                "request": request,
                "request_path": request_path,
                "response_path": response_path,
                "stdout_path": Path(kwargs["stdout"].name),
                "stderr_path": Path(kwargs["stderr"].name),
            }
        )
        if self.launch_error is not None:
            raise self.launch_error
        if self.response_text is not None:
            with response_path.open("w", encoding="utf-8") as response_file:
                response_file.write(self.response_text)
        elif self.response is not None:
            with response_path.open("w", encoding="utf-8") as response_file:
                json.dump(self.response, response_file)
        kwargs["stdout"].write(b"worker stdout")
        kwargs["stdout"].flush()
        kwargs["stderr"].write(b"worker stderr")
        kwargs["stderr"].flush()
        return self

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.wait_error is not None and len(self.wait_calls) == 1:
            raise self.wait_error
        if self.wait_timeout and len(self.wait_calls) == 1:
            raise subprocess.TimeoutExpired(["worker"], timeout)
        self.returncode = self.target_returncode
        return self.returncode

    def poll(self):
        self.poll_calls += 1
        if self.poll_result is not None:
            self.returncode = self.poll_result
        return self.poll_result


class IsaacSubprocessSceneBackendTests(unittest.TestCase):
    def make_backend(self, runner, **overrides):
        values = {
            "root": ROOT,
            "python_executable": "/test/python",
            "timeout": 19,
        }
        values.update(overrides)
        backend = IsaacSubprocessSceneBackend(**values)
        contexts = ExitStack()
        contexts.enter_context(patch(
            "agent.scene.isaac_scene_worker.subprocess.Popen",
            side_effect=runner,
        ))
        contexts.enter_context(patch("os.killpg"))
        return contexts, backend

    def assert_ipc_cleaned(self, runner):
        self.assertTrue(runner.calls)
        for call in runner.calls:
            self.assertFalse(call["request_path"].exists())
            self.assertFalse(call["response_path"].exists())
            self.assertFalse(call["stdout_path"].exists())
            self.assertFalse(call["stderr_path"].exists())
            self.assertFalse(call["request_path"].parent.exists())

    def test_build_sends_resolved_objects_and_absolute_output_paths(self):
        runner = RecordingPopen({"ok": True, "result": None})
        context, backend = self.make_backend(runner)
        with tempfile.TemporaryDirectory() as tmp, context:
            output_dir = Path(tmp) / "run"
            output_dir.mkdir()
            backend.build(
                [make_object()],
                output_usd=output_dir / "scene.usd",
                output_json=output_dir / "scene.json",
                layout_profile={"surface_z": 0.775},
            )

        call = runner.calls[0]
        self.assertEqual(
            call["command"][:3],
            ["/test/python", "-m", "agent.scene.isaac_scene_worker_main"],
        )
        self.assertEqual(call["kwargs"]["cwd"], str(ROOT.resolve()))
        self.assertTrue(call["kwargs"]["start_new_session"])
        self.assertNotEqual(call["kwargs"]["stdout"], subprocess.PIPE)
        self.assertNotEqual(call["kwargs"]["stderr"], subprocess.PIPE)
        self.assertEqual(runner.wait_calls, [19])
        request = call["request"]
        self.assertEqual(set(request), {
            "version",
            "operation",
            "root",
            "objects",
            "output_usd",
            "output_json",
            "layout_profile",
        })
        self.assertEqual(request["version"], 1)
        self.assertEqual(request["operation"], "build")
        self.assertEqual(request["root"], str(ROOT.resolve()))
        self.assertTrue(Path(request["output_usd"]).is_absolute())
        self.assertTrue(Path(request["output_json"]).is_absolute())
        self.assertEqual(request["objects"], [make_object().model_dump(mode="json")])
        self.assertEqual(request["layout_profile"], {"surface_z": 0.775})
        self.assert_ipc_cleaned(runner)

    def test_preflight_returns_a_strict_report(self):
        runner = RecordingPopen({
            "ok": True,
            "result": {"passed": True, "issues": []},
        })
        context, backend = self.make_backend(runner)
        with tempfile.TemporaryDirectory() as tmp, context:
            run_dir = Path(tmp)
            report = backend.preflight(
                [make_object()],
                usd_path=run_dir / "scene.usd",
                scene_json_path=run_dir / "scene.json",
                layout_profile={"surface_z": 0.775},
            )

        self.assertEqual(report, ScenePreflightReport(passed=True))
        self.assertEqual(runner.calls[0]["request"]["operation"], "preflight")
        self.assertEqual(set(runner.calls[0]["request"]), {
            "version",
            "operation",
            "root",
            "objects",
            "usd_path",
            "scene_json_path",
            "layout_profile",
        })
        self.assert_ipc_cleaned(runner)

    def test_updater_facade_delegates_all_update_arguments(self):
        runner = RecordingPopen({"ok": True, "result": True})
        context, backend = self.make_backend(runner)
        with tempfile.TemporaryDirectory() as tmp, context:
            scenes_dir = Path(tmp) / "scenes"
            scenes_dir.mkdir()
            updater = backend.position_updater_factory(
                scenes_dir=str(scenes_dir),
            )
            updated = updater.apply_positions_to_usd(
                scenes_dir / "candidate.json",
                scenes_dir / "candidate.usd",
                output_usd_path=scenes_dir / "published.usd",
                in_place=False,
                required_prim_paths={"/World/B", "/World/A"},
            )

        self.assertTrue(updated)
        request = runner.calls[0]["request"]
        self.assertEqual(request, {
            "version": 1,
            "operation": "apply_positions",
            "root": str(ROOT.resolve()),
            "scenes_dir": str(scenes_dir.resolve()),
            "json_file_path": str((scenes_dir / "candidate.json").resolve()),
            "usd_file_path": str((scenes_dir / "candidate.usd").resolve()),
            "output_usd_path": str((scenes_dir / "published.usd").resolve()),
            "in_place": False,
            "required_prim_paths": ["/World/A", "/World/B"],
        })
        self.assert_ipc_cleaned(runner)

    def test_timeout_is_typed_and_ipc_is_cleaned(self):
        runner = RecordingPopen(wait_timeout=True)
        context, backend = self.make_backend(runner, termination_grace=0)
        with context, patch("os.killpg") as killpg, self.assertRaises(
            SceneWorkerError
        ) as raised:
            backend.build(
                [make_object()],
                output_usd=Path("relative.usd"),
                output_json=Path("relative.json"),
                layout_profile={},
            )

        self.assertEqual(raised.exception.code, "SCENE_WORKER_TIMEOUT")
        self.assertEqual(killpg.call_args_list, [
            call(runner.pid, signal.SIGTERM),
            call(runner.pid, signal.SIGKILL),
        ])
        self.assertEqual(runner.wait_calls, [19, 0.1])
        self.assert_ipc_cleaned(runner)

    def test_cleanup_failure_is_not_hidden_by_a_valid_response(self):
        runner = RecordingPopen(
            {"ok": True, "result": None},
            wait_timeout=True,
        )
        context, backend = self.make_backend(
            runner,
            termination_grace=0,
        )
        with context, patch(
            "os.killpg",
            side_effect=PermissionError("cannot signal process group"),
        ), self.assertRaises(SceneWorkerError) as raised:
            backend.build(
                [],
                output_usd=Path("scene.usd"),
                output_json=Path("scene.json"),
                layout_profile={},
            )

        self.assertEqual(
            raised.exception.code,
            "SCENE_WORKER_TERMINATION_FAILED",
        )
        self.assertIn("cannot signal process group", str(raised.exception))
        self.assert_ipc_cleaned(runner)

    def test_wait_base_exceptions_cleanup_running_group_and_propagate_identity(self):
        exceptions = [
            KeyboardInterrupt("keyboard interrupt"),
            SystemExit(17),
            RuntimeError("unexpected wait failure"),
        ]
        for original in exceptions:
            with self.subTest(exception=type(original).__name__):
                runner = RecordingPopen(
                    wait_error=original,
                    poll_result=None,
                )
                context, backend = self.make_backend(runner)
                with context, patch.object(
                    backend,
                    "_terminate_process_group",
                    return_value=None,
                ) as terminate, self.assertRaises(type(original)) as raised:
                    backend.build(
                        [],
                        output_usd=Path("scene.usd"),
                        output_json=Path("scene.json"),
                        layout_profile={},
                    )

                self.assertIs(raised.exception, original)
                terminate.assert_called_once_with(runner)
                self.assertEqual(runner.poll_calls, 1)
                self.assert_ipc_cleaned(runner)

    def test_wait_exception_does_not_signal_an_already_exited_process(self):
        original = KeyboardInterrupt("after natural exit")
        runner = RecordingPopen(
            wait_error=original,
            poll_result=0,
        )
        context, backend = self.make_backend(runner)
        with context, patch.object(
            backend,
            "_terminate_process_group",
            return_value=None,
        ) as terminate, self.assertRaises(KeyboardInterrupt) as raised:
            backend.build(
                [],
                output_usd=Path("scene.usd"),
                output_json=Path("scene.json"),
                layout_profile={},
            )

        self.assertIs(raised.exception, original)
        terminate.assert_not_called()
        self.assertEqual(runner.poll_calls, 1)
        self.assert_ipc_cleaned(runner)

    def test_launch_error_is_typed_and_ipc_is_cleaned(self):
        runner = RecordingPopen(launch_error=OSError("cannot execute"))
        context, backend = self.make_backend(runner)
        with context, self.assertRaises(SceneWorkerError) as raised:
            backend.preflight(
                [make_object()],
                usd_path=Path("scene.usd"),
                scene_json_path=Path("scene.json"),
                layout_profile={},
            )

        self.assertEqual(raised.exception.code, "SCENE_WORKER_LAUNCH_FAILED")
        self.assertIn("cannot execute", str(raised.exception))
        self.assert_ipc_cleaned(runner)

    def test_nonzero_exit_is_typed_and_ipc_is_cleaned(self):
        runner = RecordingPopen(returncode=7)
        context, backend = self.make_backend(runner)
        with context, self.assertRaises(SceneWorkerError) as raised:
            backend.preflight(
                [make_object()],
                usd_path=Path("scene.usd"),
                scene_json_path=Path("scene.json"),
                layout_profile={},
            )

        self.assertEqual(raised.exception.code, "SCENE_WORKER_PROCESS_FAILED")
        self.assertEqual(raised.exception.returncode, 7)
        self.assertEqual(raised.exception.stderr, "worker stderr")
        self.assert_ipc_cleaned(runner)

    def test_missing_response_is_typed_and_ipc_is_cleaned(self):
        runner = RecordingPopen()
        context, backend = self.make_backend(runner)
        with context, self.assertRaises(SceneWorkerError) as raised:
            backend.build(
                [make_object()],
                output_usd=Path("scene.usd"),
                output_json=Path("scene.json"),
                layout_profile={},
            )

        self.assertEqual(raised.exception.code, "SCENE_WORKER_RESPONSE_MISSING")
        self.assert_ipc_cleaned(runner)

    def test_malformed_or_schema_invalid_response_is_rejected(self):
        cases = [
            ("not json", "SCENE_WORKER_RESPONSE_INVALID"),
            (json.dumps([]), "SCENE_WORKER_RESPONSE_INVALID"),
            (json.dumps({"ok": True}), "SCENE_WORKER_RESPONSE_INVALID"),
            (
                json.dumps({"ok": True, "result": None, "extra": True}),
                "SCENE_WORKER_RESPONSE_INVALID",
            ),
            (
                json.dumps({"ok": False, "error": {"code": 1, "message": "bad"}}),
                "SCENE_WORKER_RESPONSE_INVALID",
            ),
        ]
        for response_text, code in cases:
            with self.subTest(response_text=response_text):
                runner = RecordingPopen(response_text=response_text)
                context, backend = self.make_backend(runner)
                with context, self.assertRaises(SceneWorkerError) as raised:
                    backend.build(
                        [make_object()],
                        output_usd=Path("scene.usd"),
                        output_json=Path("scene.json"),
                        layout_profile={},
                    )
                self.assertEqual(raised.exception.code, code)
                self.assert_ipc_cleaned(runner)

    def test_worker_declared_error_preserves_code_and_message(self):
        runner = RecordingPopen({
            "ok": False,
            "error": {
                "code": "USD_STAGE_OPEN_FAILED",
                "message": "cannot open candidate.usd",
            },
        })
        context, backend = self.make_backend(runner)
        with context, self.assertRaises(SceneWorkerError) as raised:
            backend.position_updater_factory(scenes_dir=".").apply_positions_to_usd(
                Path("candidate.json"),
                Path("candidate.usd"),
            )

        self.assertEqual(raised.exception.code, "USD_STAGE_OPEN_FAILED")
        self.assertEqual(str(raised.exception), "cannot open candidate.usd")
        self.assert_ipc_cleaned(runner)

    def test_temporary_directory_creation_error_is_typed(self):
        backend = IsaacSubprocessSceneBackend(ROOT)
        with patch(
            "agent.scene.isaac_scene_worker.tempfile.TemporaryDirectory",
            side_effect=OSError("temporary storage unavailable"),
        ), self.assertRaises(SceneWorkerError) as raised:
            backend.build(
                [],
                output_usd=Path("scene.usd"),
                output_json=Path("scene.json"),
                layout_profile={},
            )

        self.assertEqual(raised.exception.code, "SCENE_WORKER_IPC_CREATE_FAILED")
        self.assertIn("temporary storage unavailable", str(raised.exception))

    def test_request_write_error_is_typed(self):
        runner = RecordingPopen({"ok": True, "result": None})
        context, backend = self.make_backend(runner)
        with context, patch.object(
            Path,
            "write_text",
            side_effect=OSError("request storage unavailable"),
        ), self.assertRaises(SceneWorkerError) as raised:
            backend.build(
                [],
                output_usd=Path("scene.usd"),
                output_json=Path("scene.json"),
                layout_profile={},
            )

        self.assertEqual(raised.exception.code, "SCENE_WORKER_REQUEST_WRITE_FAILED")
        self.assertIn("request storage unavailable", str(raised.exception))
        self.assertEqual(runner.calls, [])

    def test_response_read_error_is_typed(self):
        runner = RecordingPopen({"ok": True, "result": None})
        context, backend = self.make_backend(runner)
        original_read_text = Path.read_text

        def fail_response_read(path, *args, **kwargs):
            if path.name == "response.json":
                raise OSError("response storage unavailable")
            return original_read_text(path, *args, **kwargs)

        with context, patch.object(
            Path,
            "read_text",
            fail_response_read,
        ), self.assertRaises(SceneWorkerError) as raised:
            backend.build(
                [],
                output_usd=Path("scene.usd"),
                output_json=Path("scene.json"),
                layout_profile={},
            )

        self.assertEqual(raised.exception.code, "SCENE_WORKER_RESPONSE_READ_FAILED")
        self.assertIn("response storage unavailable", str(raised.exception))
        self.assert_ipc_cleaned(runner)


FAKE_OS_WORKER_SOURCE = r'''
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--response", required=True)
args, _ = parser.parse_known_args()
mode = os.environ["LABUTOPIA_FAKE_WORKER_MODE"]
response_path = Path(args.response)


def write_response(payload):
    temporary = response_path.with_name(f".{response_path.name}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(response_path)


if mode == "large_nonzero":
    sys.stdout.write("o" * 20000 + "STDOUT_END")
    sys.stderr.write("e" * 20000 + "STDERR_END")
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(9)
if mode == "response_nonzero":
    write_response({"ok": True, "result": None})
    raise SystemExit(7)
if mode == "worker_error_nonzero":
    write_response({
        "ok": False,
        "error": {"code": "FAKE_WORKER_ERROR", "message": "declared failure"},
    })
    raise SystemExit(8)
if mode == "response_hang_descendant":
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    Path(os.environ["LABUTOPIA_FAKE_CHILD_PID"]).write_text(
        str(child.pid),
        encoding="utf-8",
    )
    write_response({"ok": True, "result": None})
    time.sleep(60)
if mode == "hang_no_response":
    sys.stdout.write("timeout stdout")
    sys.stderr.write("timeout stderr")
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(60)
if mode == "nonzero_no_response":
    sys.stdout.write("nonzero stdout")
    sys.stderr.write("nonzero stderr")
    raise SystemExit(6)
if mode == "missing_response":
    raise SystemExit(0)
if mode == "malformed_response":
    response_path.write_text("not json", encoding="utf-8")
    raise SystemExit(0)
raise RuntimeError(f"unknown mode: {mode}")
'''


class IsaacSubprocessSceneBackendOSTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        (self.root / "fake_scene_worker.py").write_text(
            FAKE_OS_WORKER_SOURCE,
            encoding="utf-8",
        )

    def make_backend(self, **overrides):
        values = {
            "root": self.root,
            "python_executable": sys.executable,
            "worker_module": "fake_scene_worker",
            "timeout": 1,
        }
        values.update(overrides)
        return IsaacSubprocessSceneBackend(**values)

    def build(self, mode, **backend_overrides):
        backend = self.make_backend(**backend_overrides)
        with patch.dict(
            os.environ,
            {"LABUTOPIA_FAKE_WORKER_MODE": mode},
        ):
            return backend.build(
                [],
                output_usd=self.root / "scene.usd",
                output_json=self.root / "scene.json",
                layout_profile={},
            )

    @staticmethod
    def process_is_running(pid):
        stat_path = Path(f"/proc/{pid}/stat")
        try:
            fields = stat_path.read_text(encoding="utf-8").split()
        except FileNotFoundError:
            return False
        return len(fields) > 2 and fields[2] != "Z"

    def test_large_worker_logs_are_returned_only_as_bounded_tails(self):
        with self.assertRaises(SceneWorkerError) as raised:
            self.build("large_nonzero")

        self.assertEqual(raised.exception.code, "SCENE_WORKER_PROCESS_FAILED")
        self.assertLessEqual(len(raised.exception.stdout), 4000)
        self.assertLessEqual(len(raised.exception.stderr), 4000)
        self.assertTrue(raised.exception.stdout.endswith("STDOUT_END"))
        self.assertTrue(raised.exception.stderr.endswith("STDERR_END"))

    def test_valid_response_wins_over_nonzero_shutdown_status(self):
        self.assertIsNone(self.build("response_nonzero"))

    def test_worker_declared_error_wins_over_nonzero_shutdown_status(self):
        with self.assertRaises(SceneWorkerError) as raised:
            self.build("worker_error_nonzero")

        self.assertEqual(raised.exception.code, "FAKE_WORKER_ERROR")
        self.assertEqual(str(raised.exception), "declared failure")
        self.assertEqual(raised.exception.returncode, 8)

    def test_response_before_timeout_wins_after_process_group_cleanup(self):
        child_pid_path = self.root / "child.pid"
        backend = self.make_backend(timeout=0.2, termination_grace=0.1)
        child_pid = None
        try:
            with patch.dict(
                os.environ,
                {
                    "LABUTOPIA_FAKE_WORKER_MODE": "response_hang_descendant",
                    "LABUTOPIA_FAKE_CHILD_PID": str(child_pid_path),
                },
            ):
                self.assertIsNone(backend.build(
                    [],
                    output_usd=self.root / "scene.usd",
                    output_json=self.root / "scene.json",
                    layout_profile={},
                ))
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2
            while self.process_is_running(child_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(self.process_is_running(child_pid))
        finally:
            if child_pid is None and child_pid_path.is_file():
                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            if child_pid is not None and self.process_is_running(child_pid):
                os.kill(child_pid, signal.SIGKILL)

    def test_timeout_without_response_is_typed_after_cleanup(self):
        with self.assertRaises(SceneWorkerError) as raised:
            self.build("hang_no_response", timeout=0.2, termination_grace=0.1)

        self.assertEqual(raised.exception.code, "SCENE_WORKER_TIMEOUT")
        self.assertIn("timeout stdout", raised.exception.stdout)
        self.assertIn("timeout stderr", raised.exception.stderr)

    def test_nonzero_and_missing_response_are_distinct(self):
        cases = [
            ("nonzero_no_response", "SCENE_WORKER_PROCESS_FAILED"),
            ("missing_response", "SCENE_WORKER_RESPONSE_MISSING"),
            ("malformed_response", "SCENE_WORKER_RESPONSE_INVALID"),
        ]
        for mode, expected_code in cases:
            with self.subTest(mode=mode), self.assertRaises(
                SceneWorkerError
            ) as raised:
                self.build(mode)
            self.assertEqual(raised.exception.code, expected_code)

    def test_launch_failure_is_typed(self):
        backend = self.make_backend(python_executable="/missing/python")
        with self.assertRaises(SceneWorkerError) as raised:
            with patch.dict(
                os.environ,
                {"LABUTOPIA_FAKE_WORKER_MODE": "missing_response"},
            ):
                backend.build(
                    [],
                    output_usd=self.root / "scene.usd",
                    output_json=self.root / "scene.json",
                    layout_profile={},
                )

        self.assertEqual(raised.exception.code, "SCENE_WORKER_LAUNCH_FAILED")


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


class LazyConsumerModule(types.ModuleType):
    def __init__(self, name, exported_name, exported_value, events):
        super().__init__(name)
        self._exported_name = exported_name
        self._exported_value = exported_value
        self._events = events

    def __getattr__(self, name):
        if name == self._exported_name:
            self._events.append(f"import:{name}")
            return self._exported_value
        raise AttributeError(name)


class NoopWorkerBackend:
    def __init__(self, root):
        pass

    def build(self, objects, **kwargs):
        pass


class NoopWorkerUpdater:
    pass


def make_build_worker_request():
    return {
        "version": 1,
        "operation": "build",
        "root": str(ROOT.resolve()),
        "objects": [],
        "output_usd": "/tmp/run/scene.usd",
        "output_json": "/tmp/run/scene.json",
        "layout_profile": {},
    }


class IsaacSceneWorkerMainTests(unittest.TestCase):
    def test_parser_separates_worker_arguments_from_kit_arguments(self):
        args, kit_args = isaac_scene_worker_main.parse_cli_args([
            "--request",
            "/tmp/request.json",
            "--response",
            "/tmp/response.json",
            "--no-headless",
            "--/rtx/example=true",
        ])

        self.assertEqual(args.request, Path("/tmp/request.json"))
        self.assertEqual(args.response, Path("/tmp/response.json"))
        self.assertFalse(args.headless)
        self.assertEqual(kit_args, ["--/rtx/example=true"])

    def run_worker(
        self,
        request,
        backend_class,
        updater_class,
        *,
        extra_args=(),
        close_assertion=None,
        close_error=None,
        response_writer=None,
        request_text=None,
    ):
        events = []
        apps = []
        original_parse = isaac_scene_worker_main.parse_cli_args

        class FakeSimulationApp:
            def __init__(self, config):
                events.append(("simulation_app", config))
                apps.append(self)

            def close(self):
                events.append("close")
                if close_assertion is not None:
                    close_assertion(response_path)
                if close_error is not None:
                    raise close_error

        def recording_parse(argv=None):
            events.append("parse")
            return original_parse(argv)

        def recording_prepare(kit_args):
            events.append(("prepare", list(kit_args)))

        modules = {
            "isaacsim": fake_module(
                "isaacsim",
                SimulationApp=FakeSimulationApp,
            ),
            "agent.scene.legacy_scene_backend": LazyConsumerModule(
                "agent.scene.legacy_scene_backend",
                "LegacySceneBackend",
                backend_class,
                events,
            ),
            "agent.scene.optimization.position_updater": LazyConsumerModule(
                "agent.scene.optimization.position_updater",
                "PositionUpdater",
                updater_class,
                events,
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            request_path = Path(tmp) / "request.json"
            response_path = Path(tmp) / "response.json"
            request_path.write_text(
                json.dumps(request) if request_text is None else request_text,
                encoding="utf-8",
            )
            argv = [
                "--request",
                str(request_path),
                "--response",
                str(response_path),
                *extra_args,
            ]
            writer_context = (
                nullcontext()
                if response_writer is None
                else patch.object(
                    isaac_scene_worker_main,
                    "_write_response",
                    side_effect=response_writer,
                )
            )
            with patch.dict(sys.modules, modules), patch.object(
                isaac_scene_worker_main,
                "parse_cli_args",
                side_effect=recording_parse,
            ), patch.object(
                isaac_scene_worker_main,
                "prepare_isaacsim_argv",
                side_effect=recording_prepare,
            ), writer_context:
                returncode = isaac_scene_worker_main.main(argv)
            response = (
                json.loads(response_path.read_text(encoding="utf-8"))
                if response_path.is_file()
                else None
            )
        return returncode, response, events, apps

    def test_response_is_atomically_persisted_before_isaac_shutdown(self):
        observed_responses = []

        def assert_persisted_response(response_path):
            temporary_path = response_path.with_name(
                f".{response_path.name}.tmp"
            )
            self.assertTrue(response_path.is_file())
            self.assertFalse(temporary_path.exists())
            observed_responses.append(
                json.loads(response_path.read_text(encoding="utf-8"))
            )

        returncode, response, events, _ = self.run_worker(
            make_build_worker_request(),
            NoopWorkerBackend,
            NoopWorkerUpdater,
            close_assertion=assert_persisted_response,
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(response, {"ok": True, "result": None})
        self.assertEqual(
            observed_responses,
            [{"ok": True, "result": None}],
        )
        self.assertEqual(events[-1], "close")

    def test_close_error_atomically_overwrites_a_success_response(self):
        returncode, response, events, _ = self.run_worker(
            make_build_worker_request(),
            NoopWorkerBackend,
            NoopWorkerUpdater,
            close_error=RuntimeError("Kit close failed"),
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(response, {
            "ok": False,
            "error": {
                "code": "SCENE_WORKER_CLOSE_FAILED",
                "message": "Kit close failed",
            },
        })
        self.assertEqual(events[-1], "close")

    def test_response_write_failure_still_closes_and_returns_two(self):
        def fail_write(path, response):
            raise OSError("disk unavailable")

        returncode, response, events, _ = self.run_worker(
            make_build_worker_request(),
            NoopWorkerBackend,
            NoopWorkerUpdater,
            response_writer=fail_write,
        )

        self.assertEqual(returncode, 2)
        self.assertIsNone(response)
        self.assertEqual(events[-1], "close")

    def test_build_initializes_isaac_before_importing_consumers_and_closes(self):
        calls = {}

        class FakeBackend:
            def __init__(self, root):
                calls["root"] = root

            def build(self, objects, **kwargs):
                calls["build"] = (objects, kwargs)

        class FakeUpdater:
            pass

        request = {
            "version": 1,
            "operation": "build",
            "root": str(ROOT.resolve()),
            "objects": [make_object().model_dump(mode="json")],
            "output_usd": "/tmp/run/scene.usd",
            "output_json": "/tmp/run/scene.json",
            "layout_profile": {"surface_z": 0.775},
        }

        returncode, response, events, apps = self.run_worker(
            request,
            FakeBackend,
            FakeUpdater,
            extra_args=("--/rtx/example=true",),
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(response, {"ok": True, "result": None})
        self.assertEqual(len(apps), 1)
        self.assertEqual(events[0], "parse")
        self.assertEqual(events[1], ("prepare", ["--/rtx/example=true"]))
        self.assertEqual(events[2], ("simulation_app", {"headless": True}))
        self.assertGreater(
            events.index("import:LegacySceneBackend"),
            events.index(("simulation_app", {"headless": True})),
        )
        self.assertGreater(
            events.index("import:PositionUpdater"),
            events.index(("simulation_app", {"headless": True})),
        )
        self.assertEqual(events[-1], "close")
        self.assertEqual(calls["root"], ROOT.resolve())
        objects, kwargs = calls["build"]
        self.assertEqual(objects, [make_object()])
        self.assertEqual(kwargs, {
            "output_usd": Path("/tmp/run/scene.usd"),
            "output_json": Path("/tmp/run/scene.json"),
            "layout_profile": {"surface_z": 0.775},
        })

    def test_preflight_dispatches_and_serializes_the_strict_report(self):
        calls = {}
        expected = ScenePreflightReport(
            passed=False,
            issues=(
                ScenePreflightIssue(
                    code="MISSING_PRIM",
                    message="missing /World/Flask",
                    object_id="flask",
                ),
            ),
        )

        class FakeBackend:
            def __init__(self, root):
                calls["root"] = root

            def preflight(self, objects, **kwargs):
                calls["preflight"] = (objects, kwargs)
                return expected

        class FakeUpdater:
            pass

        request = {
            "version": 1,
            "operation": "preflight",
            "root": str(ROOT.resolve()),
            "objects": [make_object().model_dump(mode="json")],
            "usd_path": "/tmp/run/scene.usd",
            "scene_json_path": "/tmp/run/scene.json",
            "layout_profile": {"surface_z": 0.775},
        }

        returncode, response, events, _ = self.run_worker(
            request,
            FakeBackend,
            FakeUpdater,
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(response, {
            "ok": True,
            "result": expected.model_dump(mode="json"),
        })
        self.assertEqual(events[-1], "close")
        objects, kwargs = calls["preflight"]
        self.assertEqual(objects, [make_object()])
        self.assertEqual(kwargs, {
            "usd_path": Path("/tmp/run/scene.usd"),
            "scene_json_path": Path("/tmp/run/scene.json"),
            "layout_profile": {"surface_z": 0.775},
        })

    def test_apply_positions_dispatches_every_argument(self):
        calls = {}

        class FakeBackend:
            def __init__(self, root):
                calls["root"] = root

        class FakeUpdater:
            def __init__(self, scenes_dir):
                calls["scenes_dir"] = scenes_dir

            def apply_positions_to_usd(self, *args, **kwargs):
                calls["apply"] = (args, kwargs)
                return False

        request = {
            "version": 1,
            "operation": "apply_positions",
            "root": str(ROOT.resolve()),
            "scenes_dir": "/tmp/run",
            "json_file_path": "/tmp/run/candidate.json",
            "usd_file_path": "/tmp/run/candidate.usd",
            "output_usd_path": "/tmp/run/output.usd",
            "in_place": False,
            "required_prim_paths": ["/World/A", "/World/B"],
        }

        returncode, response, events, _ = self.run_worker(
            request,
            FakeBackend,
            FakeUpdater,
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(response, {"ok": True, "result": False})
        self.assertEqual(events[-1], "close")
        self.assertEqual(calls["scenes_dir"], "/tmp/run")
        self.assertEqual(calls["apply"], (
            (
                Path("/tmp/run/candidate.json"),
                Path("/tmp/run/candidate.usd"),
            ),
            {
                "output_usd_path": Path("/tmp/run/output.usd"),
                "in_place": False,
                "required_prim_paths": {"/World/A", "/World/B"},
            },
        ))

    def test_worker_failure_writes_a_typed_response_and_closes(self):
        class FailingBackend:
            def __init__(self, root):
                pass

            def build(self, objects, **kwargs):
                raise RuntimeError("generation exploded")

        class FakeUpdater:
            pass

        request = {
            "version": 1,
            "operation": "build",
            "root": str(ROOT.resolve()),
            "objects": [],
            "output_usd": "/tmp/run/scene.usd",
            "output_json": "/tmp/run/scene.json",
            "layout_profile": {},
        }

        returncode, response, events, _ = self.run_worker(
            request,
            FailingBackend,
            FakeUpdater,
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(response, {
            "ok": False,
            "error": {
                "code": "SCENE_WORKER_BUILD_FAILED",
                "message": "generation exploded",
            },
        })
        self.assertEqual(events[-1], "close")

    def test_invalid_request_is_rejected_before_isaac_initialization(self):
        class FakeBackend:
            pass

        class FakeUpdater:
            pass

        cases = [
            ({"version": 1, "operation": "unknown"}, None),
            (None, "not json"),
            (
                {
                    **make_build_worker_request(),
                    "objects": [{"id": "only-id"}],
                },
                None,
            ),
        ]
        for request, request_text in cases:
            with self.subTest(request_text=request_text):
                returncode, response, events, apps = self.run_worker(
                    request,
                    FakeBackend,
                    FakeUpdater,
                    request_text=request_text,
                )

                self.assertEqual(returncode, 0)
                self.assertEqual(response["ok"], False)
                self.assertEqual(
                    response["error"]["code"],
                    "SCENE_WORKER_REQUEST_INVALID",
                )
                self.assertEqual(set(response), {"ok", "error"})
                self.assertEqual(set(response["error"]), {"code", "message"})
                self.assertEqual(events, ["parse"])
                self.assertEqual(apps, [])


if __name__ == "__main__":
    unittest.main()
