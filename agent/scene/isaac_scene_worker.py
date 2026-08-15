from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from agent.scene.scene_preflight import ScenePreflightReport


_DIAGNOSTIC_TAIL_CHARS = 4000


class SceneWorkerError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ):
        self.code = code
        self.returncode = returncode
        self.stdout = str(stdout or "")[-_DIAGNOSTIC_TAIL_CHARS:]
        self.stderr = str(stderr or "")[-_DIAGNOSTIC_TAIL_CHARS:]
        super().__init__(message)


class IsaacSubprocessPositionUpdater:
    def __init__(self, backend: "IsaacSubprocessSceneBackend", scenes_dir: str):
        self._backend = backend
        self.scenes_dir = Path(scenes_dir).resolve()

    def apply_positions_to_usd(
        self,
        json_file_path: Path,
        usd_file_path: Path,
        output_usd_path: Path | None = None,
        in_place: bool = True,
        required_prim_paths: set[str] | None = None,
    ) -> bool:
        request = {
            "scenes_dir": str(self.scenes_dir),
            "json_file_path": str(Path(json_file_path).resolve()),
            "usd_file_path": str(Path(usd_file_path).resolve()),
            "output_usd_path": (
                None
                if output_usd_path is None
                else str(Path(output_usd_path).resolve())
            ),
            "in_place": in_place,
            "required_prim_paths": (
                None
                if required_prim_paths is None
                else sorted(required_prim_paths)
            ),
        }
        result = self._backend._invoke("apply_positions", request)
        if type(result) is not bool:
            raise SceneWorkerError(
                "SCENE_WORKER_RESPONSE_INVALID",
                "apply_positions worker result must be a boolean",
            )
        return result


class IsaacSubprocessSceneBackend:
    def __init__(
        self,
        root: Path,
        *,
        python_executable: str | None = None,
        timeout: int | float = 600,
        termination_grace: int | float = 2,
        worker_module: str = "agent.scene.isaac_scene_worker_main",
    ):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number")
        if (
            isinstance(termination_grace, bool)
            or not isinstance(termination_grace, (int, float))
            or termination_grace < 0
        ):
            raise ValueError("termination_grace must be a nonnegative number")
        self.root = Path(root).resolve()
        self.python_executable = python_executable or sys.executable
        self.timeout = timeout
        self.termination_grace = termination_grace
        self.worker_module = worker_module

    def build(
        self,
        objects,
        *,
        output_usd: Path,
        output_json: Path,
        layout_profile: dict,
    ) -> None:
        result = self._invoke(
            "build",
            {
                "objects": self._serialize_objects(objects),
                "output_usd": str(Path(output_usd).resolve()),
                "output_json": str(Path(output_json).resolve()),
                "layout_profile": layout_profile,
            },
        )
        if result is not None:
            raise SceneWorkerError(
                "SCENE_WORKER_RESPONSE_INVALID",
                "build worker result must be null",
            )

    def preflight(
        self,
        objects,
        *,
        usd_path: Path,
        scene_json_path: Path,
        layout_profile: dict,
    ) -> ScenePreflightReport:
        result = self._invoke(
            "preflight",
            {
                "objects": self._serialize_objects(objects),
                "usd_path": str(Path(usd_path).resolve()),
                "scene_json_path": str(Path(scene_json_path).resolve()),
                "layout_profile": layout_profile,
            },
        )
        if not isinstance(result, dict):
            raise SceneWorkerError(
                "SCENE_WORKER_RESPONSE_INVALID",
                "preflight worker result must be an object",
            )
        try:
            return ScenePreflightReport(**result)
        except (TypeError, ValueError, ValidationError) as exc:
            raise SceneWorkerError(
                "SCENE_WORKER_RESPONSE_INVALID",
                "preflight worker returned an invalid report",
            ) from exc

    def position_updater_factory(
        self,
        *,
        scenes_dir: str,
    ) -> IsaacSubprocessPositionUpdater:
        return IsaacSubprocessPositionUpdater(self, scenes_dir)

    @staticmethod
    def _serialize_objects(objects: Sequence) -> list[dict]:
        serialized = []
        for obj in objects:
            try:
                data = obj.model_dump(mode="json", warnings="error")
            except (AttributeError, TypeError, ValueError) as exc:
                raise SceneWorkerError(
                    "SCENE_WORKER_REQUEST_INVALID",
                    "scene objects must support strict model serialization",
                ) from exc
            if not isinstance(data, dict):
                raise SceneWorkerError(
                    "SCENE_WORKER_REQUEST_INVALID",
                    "serialized scene object must be an object",
                )
            serialized.append(data)
        return serialized

    def _invoke(self, operation: str, payload: dict):
        request = {
            "version": 1,
            "operation": operation,
            "root": str(self.root),
            **payload,
        }
        try:
            request_text = json.dumps(
                request,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise SceneWorkerError(
                "SCENE_WORKER_REQUEST_INVALID",
                "scene worker request is not strict JSON",
            ) from exc

        try:
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="labutopia-isaac-scene-worker-",
            )
        except OSError as exc:
            raise SceneWorkerError(
                "SCENE_WORKER_IPC_CREATE_FAILED",
                f"failed to create scene worker IPC directory: {exc}",
            ) from exc

        try:
            result = self._invoke_in_directory(
                request_text,
                Path(temporary_directory.name),
            )
        except BaseException:
            try:
                temporary_directory.cleanup()
            except OSError:
                pass
            raise
        try:
            temporary_directory.cleanup()
        except OSError as exc:
            raise SceneWorkerError(
                "SCENE_WORKER_IPC_CLEANUP_FAILED",
                f"failed to clean scene worker IPC directory: {exc}",
            ) from exc
        return result

    def _invoke_in_directory(
        self,
        request_text: str,
        ipc_path: Path,
    ):
        request_path = ipc_path / "request.json"
        response_path = ipc_path / "response.json"
        stdout_path = ipc_path / "stdout.log"
        stderr_path = ipc_path / "stderr.log"
        try:
            request_path.write_text(request_text, encoding="utf-8")
        except OSError as exc:
            raise SceneWorkerError(
                "SCENE_WORKER_REQUEST_WRITE_FAILED",
                f"failed to write scene worker request: {exc}",
            ) from exc

        command = [
            self.python_executable,
            "-m",
            self.worker_module,
            "--request",
            str(request_path),
            "--response",
            str(response_path),
            "--headless",
        ]
        # Regular files avoid pipe deadlocks and RAM growth from verbose Kit logs.
        # They remain private to this short-lived IPC directory; only bounded tails
        # are read before cleanup.
        try:
            stdout_file = stdout_path.open("wb")
        except OSError as exc:
            raise SceneWorkerError(
                "SCENE_WORKER_LOG_OPEN_FAILED",
                f"failed to open scene worker stdout log: {exc}",
            ) from exc
        try:
            stderr_file = stderr_path.open("wb")
        except OSError as exc:
            stdout_file.close()
            raise SceneWorkerError(
                "SCENE_WORKER_LOG_OPEN_FAILED",
                f"failed to open scene worker stderr log: {exc}",
            ) from exc

        process = None
        timed_out = False
        termination_error = None
        with stdout_file, stderr_file:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.root),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
            except OSError as exc:
                raise SceneWorkerError(
                    "SCENE_WORKER_LAUNCH_FAILED",
                    f"failed to launch scene worker: {exc}",
                ) from exc
            try:
                process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                termination_error = self._terminate_process_group(process)
            except BaseException:
                try:
                    process_is_running = process.poll() is None
                except BaseException:
                    process_is_running = True
                if process_is_running:
                    try:
                        self._terminate_process_group(process)
                    except BaseException:
                        pass
                raise

        returncode = process.returncode if process is not None else None
        stdout = self._read_log_tail(stdout_path)
        stderr = self._read_log_tail(stderr_path)

        response = None
        response_error = None
        try:
            response_is_regular = (
                response_path.is_file() and not response_path.is_symlink()
            )
        except OSError as exc:
            response_is_regular = False
            response_error = SceneWorkerError(
                "SCENE_WORKER_RESPONSE_READ_FAILED",
                f"failed to inspect scene worker response: {exc}",
            )
        if response_is_regular:
            try:
                response = self._read_response(response_path)
            except SceneWorkerError as exc:
                response_error = exc

        if termination_error is not None:
            raise SceneWorkerError(
                "SCENE_WORKER_TERMINATION_FAILED",
                termination_error,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )
        if response is not None:
            if response["ok"] is False:
                error = response["error"]
                raise SceneWorkerError(
                    error["code"],
                    error["message"],
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
            return response["result"]

        if timed_out:
            raise SceneWorkerError(
                "SCENE_WORKER_TIMEOUT",
                f"scene worker exceeded {self.timeout} seconds",
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )
        if returncode != 0:
            raise SceneWorkerError(
                "SCENE_WORKER_PROCESS_FAILED",
                f"scene worker exited with status {returncode}",
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )
        if response_error is not None:
            raise SceneWorkerError(
                response_error.code,
                str(response_error),
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            ) from response_error
        raise SceneWorkerError(
            "SCENE_WORKER_RESPONSE_MISSING",
            "scene worker did not produce a regular response file",
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _terminate_process_group(self, process) -> str | None:
        errors = []
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f"SIGTERM failed: {exc}")

        if self.termination_grace > 0:
            time.sleep(self.termination_grace)

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f"SIGKILL failed: {exc}")

        try:
            process.wait(timeout=max(self.termination_grace, 0.1))
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=max(self.termination_grace, 0.1))
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"worker reap failed: {exc}")
        except OSError as exc:
            errors.append(f"wait after SIGKILL failed: {exc}")
        if process.returncode is None:
            errors.append("scene worker could not be reaped")
        return "; ".join(errors) if errors else None

    @staticmethod
    def _read_log_tail(path: Path) -> str:
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(
                    max(0, size - (_DIAGNOSTIC_TAIL_CHARS * 4)),
                    os.SEEK_SET,
                )
                data = stream.read()
        except OSError as exc:
            return f"[worker log unavailable: {exc}]"[-_DIAGNOSTIC_TAIL_CHARS:]
        return data.decode("utf-8", errors="replace")[-_DIAGNOSTIC_TAIL_CHARS:]

    @classmethod
    def _read_response(cls, response_path: Path) -> dict:
        try:
            response = json.loads(
                response_path.read_text(encoding="utf-8"),
                parse_constant=cls._reject_nonfinite_constant,
            )
        except OSError as exc:
            raise SceneWorkerError(
                "SCENE_WORKER_RESPONSE_READ_FAILED",
                f"failed to read scene worker response: {exc}",
            ) from exc
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise SceneWorkerError(
                "SCENE_WORKER_RESPONSE_INVALID",
                "scene worker response is not strict JSON",
            ) from exc
        if not isinstance(response, dict) or type(response.get("ok")) is not bool:
            raise SceneWorkerError(
                "SCENE_WORKER_RESPONSE_INVALID",
                "scene worker response has an invalid schema",
            )
        if response["ok"] is True:
            if set(response) != {"ok", "result"}:
                raise SceneWorkerError(
                    "SCENE_WORKER_RESPONSE_INVALID",
                    "successful scene worker response has an invalid schema",
                )
        else:
            if set(response) != {"ok", "error"}:
                raise SceneWorkerError(
                    "SCENE_WORKER_RESPONSE_INVALID",
                    "failed scene worker response has an invalid schema",
                )
            error = response["error"]
            if (
                not isinstance(error, dict)
                or set(error) != {"code", "message"}
                or not isinstance(error["code"], str)
                or not error["code"]
                or not isinstance(error["message"], str)
                or not error["message"]
            ):
                raise SceneWorkerError(
                    "SCENE_WORKER_RESPONSE_INVALID",
                    "scene worker error has an invalid schema",
                )
        return response

    @staticmethod
    def _reject_nonfinite_constant(value: str):
        raise ValueError(f"non-finite JSON number is not allowed: {value}")
