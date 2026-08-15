from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.isaacsim_runtime import prepare_isaacsim_argv


_OPERATION_ERROR_CODES = {
    "build": "SCENE_WORKER_BUILD_FAILED",
    "preflight": "SCENE_WORKER_PREFLIGHT_FAILED",
    "apply_positions": "SCENE_WORKER_APPLY_POSITIONS_FAILED",
}


class _RequestError(ValueError):
    pass


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one isolated Isaac USD operation")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_known_args(argv)


def _reject_nonfinite_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _read_request(path: Path) -> dict:
    try:
        request = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise _RequestError("request is not strict JSON") from exc
    if not isinstance(request, dict):
        raise _RequestError("request root must be an object")
    return request


def _require_absolute_path(request: dict, key: str, *, nullable=False) -> None:
    value = request.get(key)
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise _RequestError(f"{key} must be a nonempty absolute path")


def _validate_request(request: dict) -> str:
    if type(request.get("version")) is not int or request["version"] != 1:
        raise _RequestError("unsupported request version")
    operation = request.get("operation")
    if operation not in _OPERATION_ERROR_CODES:
        raise _RequestError("unsupported scene worker operation")

    common = {"version", "operation", "root"}
    operation_keys = {
        "build": {
            "objects",
            "output_usd",
            "output_json",
            "layout_profile",
        },
        "preflight": {
            "objects",
            "usd_path",
            "scene_json_path",
            "layout_profile",
        },
        "apply_positions": {
            "scenes_dir",
            "json_file_path",
            "usd_file_path",
            "output_usd_path",
            "in_place",
            "required_prim_paths",
        },
    }
    if set(request) != common | operation_keys[operation]:
        raise _RequestError(f"{operation} request has an invalid schema")
    _require_absolute_path(request, "root")

    if operation in {"build", "preflight"}:
        objects = request["objects"]
        if not isinstance(objects, list) or not all(
            isinstance(item, dict) for item in objects
        ):
            raise _RequestError("objects must be a list of objects")
        if not isinstance(request["layout_profile"], dict):
            raise _RequestError("layout_profile must be an object")
        if operation == "build":
            _require_absolute_path(request, "output_usd")
            _require_absolute_path(request, "output_json")
        else:
            _require_absolute_path(request, "usd_path")
            _require_absolute_path(request, "scene_json_path")
        return operation

    _require_absolute_path(request, "scenes_dir")
    _require_absolute_path(request, "json_file_path")
    _require_absolute_path(request, "usd_file_path")
    _require_absolute_path(request, "output_usd_path", nullable=True)
    if type(request["in_place"]) is not bool:
        raise _RequestError("in_place must be a boolean")
    required_paths = request["required_prim_paths"]
    if required_paths is not None:
        if (
            not isinstance(required_paths, list)
            or any(
                not isinstance(path, str) or not path
                for path in required_paths
            )
            or len(set(required_paths)) != len(required_paths)
        ):
            raise _RequestError(
                "required_prim_paths must be null or unique nonempty strings"
            )
    return operation


def _resolve_objects(items, resolved_scene_object_class):
    objects = []
    for item in items:
        try:
            objects.append(resolved_scene_object_class(**item))
        except (TypeError, ValueError) as exc:
            raise _RequestError("request contains an invalid scene object") from exc
    return objects


def _dispatch(
    request,
    operation,
    legacy_scene_backend_class,
    position_updater_class,
    resolved_objects,
):
    if operation in {"build", "preflight"}:
        backend = legacy_scene_backend_class(Path(request["root"]))
        if operation == "build":
            backend.build(
                resolved_objects,
                output_usd=Path(request["output_usd"]),
                output_json=Path(request["output_json"]),
                layout_profile=request["layout_profile"],
            )
            return None

        report = backend.preflight(
            resolved_objects,
            usd_path=Path(request["usd_path"]),
            scene_json_path=Path(request["scene_json_path"]),
            layout_profile=request["layout_profile"],
        )
        try:
            return report.model_dump(mode="json", warnings="error")
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("preflight returned an invalid report") from exc

    updater = position_updater_class(scenes_dir=request["scenes_dir"])
    required_paths = request["required_prim_paths"]
    return updater.apply_positions_to_usd(
        Path(request["json_file_path"]),
        Path(request["usd_file_path"]),
        output_usd_path=(
            None
            if request["output_usd_path"] is None
            else Path(request["output_usd_path"])
        ),
        in_place=request["in_place"],
        required_prim_paths=(
            None if required_paths is None else set(required_paths)
        ),
    )


def _error_response(code: str, exc: BaseException) -> dict:
    message = str(exc) or type(exc).__name__
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _write_response(path: Path, response: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(
                response,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    args, kit_args = parse_cli_args(argv)
    response = None
    operation = None
    resolved_objects = None
    simulation_app = None
    try:
        request = _read_request(args.request)
        operation = _validate_request(request)
        if operation in {"build", "preflight"}:
            from agent.scene.scene_compiler import ResolvedSceneObject

            resolved_objects = _resolve_objects(
                request["objects"],
                ResolvedSceneObject,
            )
    except _RequestError as exc:
        response = _error_response("SCENE_WORKER_REQUEST_INVALID", exc)

    if response is None:
        try:
            prepare_isaacsim_argv(kit_args)
            from isaacsim import SimulationApp

            simulation_app = SimulationApp({"headless": args.headless})

            from agent.scene.legacy_scene_backend import LegacySceneBackend
            from agent.scene.optimization.position_updater import PositionUpdater

            result = _dispatch(
                request,
                operation,
                LegacySceneBackend,
                PositionUpdater,
                resolved_objects,
            )
            response = {"ok": True, "result": result}
        except _RequestError as exc:
            response = _error_response("SCENE_WORKER_REQUEST_INVALID", exc)
        except Exception as exc:
            code = _OPERATION_ERROR_CODES.get(
                operation,
                "SCENE_WORKER_STARTUP_FAILED",
            )
            response = _error_response(code, exc)
    if response is None:
        response = {
            "ok": False,
            "error": {
                "code": "SCENE_WORKER_STARTUP_FAILED",
                "message": "scene worker did not produce a response",
            },
        }
    response_write_failed = False
    try:
        _write_response(args.response, response)
    except (OSError, TypeError, ValueError):
        response_write_failed = True

    if simulation_app is not None:
        try:
            simulation_app.close()
        except Exception as exc:
            if response.get("ok") is True:
                response = _error_response("SCENE_WORKER_CLOSE_FAILED", exc)
                try:
                    _write_response(args.response, response)
                except (OSError, TypeError, ValueError):
                    response_write_failed = True

    if response_write_failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
