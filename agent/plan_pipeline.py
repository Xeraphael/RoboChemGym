from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from agent.action.plan_orchestrator import SimulationProcessError
from agent.runtime.run_artifacts import RunArtifacts
from agent.scene.scene_compiler import SceneCompileError


_ASSET_ERROR_CODES = {
    "UNKNOWN_ASSET",
    "UNKNOWN_ASSET_VARIANT",
    "ASSET_FILE_MISSING",
}


class ActionAgentPipeline:
    def __init__(
        self,
        planner,
        scene_compiler,
        orchestrator,
        legacy_codegen,
        run_root,
    ) -> None:
        self.planner = planner
        self.scene_compiler = scene_compiler
        self.orchestrator = orchestrator
        self.legacy_codegen = legacy_codegen
        self.run_root = Path(run_root)

    def run(
        self,
        protocol_text,
        *,
        execution_backend="plan_executor",
        max_plan_attempts=3,
    ):
        if execution_backend not in {"plan_executor", "legacy_codegen"}:
            raise ValueError(
                f"unsupported execution backend: {execution_backend}"
            )

        planning = self.planner.create_plan(
            protocol_text,
            max_attempts=max_plan_attempts,
        )
        plan_id = planning.plan.plan_id if planning.plan is not None else planning.status
        artifacts = RunArtifacts.create(self.run_root, plan_id)
        artifacts.write_protocol(protocol_text)
        artifacts.write_json(
            artifacts.repair_history_path,
            [
                attempt.model_dump(mode="json")
                for attempt in planning.attempts
            ],
        )
        if planning.plan is None:
            validation = planning.final_report or {
                "valid": False,
                "status": planning.status,
            }
            artifacts.write_json(artifacts.validation_path, validation)
            failure = self._planning_failure(planning)
            return self._terminal_report(
                artifacts,
                {
                    "execution_success": False,
                    "status": planning.status,
                    **failure,
                    "failed_step": None,
                    "steps": [],
                },
            )

        plan = planning.plan
        artifacts.write_legacy_exports(plan)

        try:
            compiled = self.scene_compiler.compile(plan, artifacts)
        except SceneCompileError as exc:
            status = (
                "scene_preflight_failed"
                if exc.code == "SCENE_PREFLIGHT_FAILED"
                else "scene_compile_failed"
            )
            return self._terminal_report(
                artifacts,
                {
                    "execution_success": False,
                    "status": status,
                    "failure_type": "scene_generation_failed",
                    "error_code": exc.code,
                    "message": str(exc),
                    "returncode": exc.returncode,
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                    "failed_step": None,
                    "steps": [],
                },
            )
        except Exception as exc:
            return self._terminal_exception_report(
                artifacts,
                status="scene_compile_failed",
                error_code="SCENE_COMPILE_ERROR",
                failure_type="scene_generation_failed",
                exc=exc,
            )
        if execution_backend == "legacy_codegen":
            try:
                self.legacy_codegen.generate(
                    plan,
                    artifacts,
                    compiled.config_path,
                )
            except Exception as exc:
                return self._terminal_exception_report(
                    artifacts,
                    status="legacy_codegen_failed",
                    error_code="LEGACY_CODEGEN_ERROR",
                    exc=exc,
                )
            try:
                raw_report = self.legacy_codegen.execute(
                    plan,
                    artifacts,
                    compiled.config_path,
                )
            except Exception as exc:
                return self._terminal_exception_report(
                    artifacts,
                    status="legacy_execution_failed",
                    error_code="LEGACY_EXECUTION_ERROR",
                    exc=exc,
                )
        else:
            try:
                raw_report = self.orchestrator.run(
                    plan,
                    compiled.config_path,
                    artifacts,
                    scene_compile_result=compiled,
                )
            except SimulationProcessError as exc:
                raw_report = exc.to_report()
            except Exception as exc:
                return self._terminal_exception_report(
                    artifacts,
                    status="execution_failed",
                    error_code="PLAN_EXECUTION_ERROR",
                    failure_type="action_execution_failed",
                    exc=exc,
                )

        try:
            report = self._report_payload(raw_report)
            report.setdefault(
                "status",
                "completed"
                if report.get("execution_success")
                else "execution_failed",
            )
            validation = planning.final_report
            report.update(
                {
                    "plan_valid": bool(validation and validation.valid),
                    "scene_preflight_passed": compiled.preflight.passed,
                    "plan_status": (
                        "executable_with_degradations"
                        if validation and validation.degraded_count
                        else "executable"
                    ),
                    "protocol_coverage": {
                        "supported": (
                            validation.supported_count if validation else 0
                        ),
                        "degraded": (
                            validation.degraded_count if validation else 0
                        ),
                        "blocked": validation.blocked_count if validation else 0,
                    },
                    "semantic_annotations": [
                        annotation.model_dump(mode="json")
                        for annotation in plan.semantic_annotations
                    ],
                    "unresolved_capabilities": [
                        capability.model_dump(mode="json")
                        for capability in plan.unresolved_capabilities
                    ],
                }
            )
            if not report.get("execution_success"):
                report.setdefault("failure_type", "action_execution_failed")
                report.setdefault(
                    "error_code",
                    self._execution_error_code(report),
                )
        except Exception as exc:
            return self._terminal_exception_report(
                artifacts,
                status="execution_report_failed",
                error_code="EXECUTION_REPORT_ERROR",
                exc=exc,
            )
        return self._terminal_report(artifacts, report)

    @staticmethod
    def _planning_failure(planning) -> dict[str, str]:
        if planning.status == "client_failed":
            return {
                "failure_type": "llm_request_failed",
                "error_code": "LLM_REQUEST_FAILED",
            }
        report = planning.final_report
        issue_codes = {
            issue.code for issue in report.issues
        } if report is not None else set()
        if issue_codes & _ASSET_ERROR_CODES:
            return {
                "failure_type": "asset_unavailable",
                "error_code": "ASSET_UNAVAILABLE",
            }
        if report is not None and not report.valid:
            return {
                "failure_type": "plan_validation_blocked",
                "error_code": "PLAN_VALIDATION_BLOCKED",
            }
        return {
            "failure_type": "llm_output_invalid",
            "error_code": "LLM_OUTPUT_INVALID",
        }

    @staticmethod
    def _execution_error_code(report: dict[str, Any]) -> str:
        failed_step = report.get("failed_step")
        for step in reversed(report.get("steps", [])):
            if step.get("step_id") == failed_step:
                verification = step.get("verification", {})
                code = verification.get("code")
                if isinstance(code, str) and code:
                    return code
        return "ACTION_EXECUTION_FAILED"

    @staticmethod
    def _report_payload(report: Any) -> dict[str, Any]:
        if hasattr(report, "model_dump"):
            report = report.model_dump(mode="json")
        if not isinstance(report, dict):
            raise ValueError("execution backend must return a report object")
        payload = deepcopy(report)
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        return payload

    def _terminal_report(
        self,
        artifacts: RunArtifacts,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        terminal = deepcopy(report)
        terminal["run_dir"] = str(artifacts.run_dir)
        artifacts.write_json(artifacts.execution_report_path, terminal)
        return deepcopy(terminal)

    def _terminal_exception_report(
        self,
        artifacts: RunArtifacts,
        *,
        status: str,
        error_code: str,
        failure_type: str | None = None,
        exc: Exception,
    ) -> dict[str, Any]:
        report = {
            "execution_success": False,
            "status": status,
            "error_code": error_code,
            "message": f"{type(exc).__name__}: {exc}",
            "failed_step": None,
            "steps": [],
        }
        if failure_type:
            report["failure_type"] = failure_type
        return self._terminal_report(artifacts, report)


class LegacyCodegenBackend:
    def __init__(
        self,
        controllers_dir,
        *,
        api_key,
        model,
        base_url=None,
        max_iterations=5,
        python_executable=None,
        project_root=None,
        headless=True,
    ) -> None:
        self.controllers_dir = Path(controllers_dir)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_iterations = max_iterations
        self.python_executable = python_executable
        self.project_root = Path(project_root) if project_root is not None else None
        self.headless = headless
        self.controller_path: Path | None = None

    def generate(self, plan, artifacts, config_path):
        from agent.action.generation.code_generator import generate_controller_code

        generated = generate_controller_code(
            action_info_path=str(artifacts.legacy_actions_path),
            controllers_dir=str(self.controllers_dir),
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
        )
        controller_path, class_name, register_name = generated
        config_path = Path(config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["controller_type"] = register_name
        config.setdefault("agent", {})["execution_backend"] = "legacy_codegen"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self.controller_path = Path(controller_path)
        return controller_path, class_name, register_name

    def execute(self, plan, artifacts, config_path):
        if self.controller_path is None:
            raise RuntimeError("legacy controller has not been generated")
        from agent.action.action_orchestrator import AgentOrchestrator

        config_path = Path(config_path)
        success = AgentOrchestrator(
            controller_file=str(self.controller_path),
            config_name=config_path.stem,
            config_dir=str(config_path.parent),
            max_iterations=self.max_iterations,
            python_executable=self.python_executable,
            project_root=(
                str(self.project_root) if self.project_root is not None else None
            ),
            run_dir=str(Path(artifacts.run_dir) / "legacy_optimization"),
            headless=self.headless,
        ).run()
        return {
            "execution_success": bool(success),
            "failed_step": None,
            "steps": [],
            "verification_level": "legacy_log_monitor",
        }
