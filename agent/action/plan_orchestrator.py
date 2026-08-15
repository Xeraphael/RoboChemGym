from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from pydantic import ValidationError

from agent.action.optimization.plan_parameter_optimizer import (
    ParameterPatch,
    apply_parameter_patch,
)
from agent.action.plan_execution.models import ExecutionReport
from agent.planning.models import AgentPlan
from agent.planning.validator import plan_fingerprint, registry_fingerprint
from agent.scene.scene_preflight import ScenePreflightReport


class SimulationProcessError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        returncode: int | None = None,
        stderr: str = "",
        stdout: str = "",
    ) -> None:
        self.code = code
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout
        super().__init__(message)

    def to_report(self) -> dict[str, Any]:
        return {
            "execution_success": False,
            "status": "simulation_process_failed",
            "error_code": self.code,
            "message": str(self),
            "returncode": self.returncode,
            "stderr": self.stderr,
            "stdout": self.stdout,
            "failed_step": None,
            "steps": [],
        }


class SubprocessSimulationRunner:
    def __init__(
        self,
        project_root,
        *,
        python_executable=None,
        timeout=600,
        headless=True,
    ) -> None:
        self.project_root = Path(project_root)
        self.python_executable = python_executable or sys.executable
        self.timeout = timeout
        self.headless = headless

    @staticmethod
    def _output_tail(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return str(value)[-4000:]

    def run(self, config_path) -> dict[str, Any]:
        config_path = Path(config_path).resolve()
        report_path = config_path.parent / "execution_report.json"
        try:
            report_path.unlink(missing_ok=True)
        except OSError as exc:
            raise SimulationProcessError(
                "SIMULATION_REPORT_CLEANUP_FAILED",
                f"could not remove stale execution report: {exc}",
            ) from exc

        cmd = [
            self.python_executable,
            "main.py",
            "--config-dir",
            str(config_path.parent),
            "--config-name",
            config_path.stem,
        ]
        if self.headless:
            cmd.append("--headless")
        cmd.append("--/rtx/verifyDriverVersion/enabled=false")

        try:
            completed = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise SimulationProcessError(
                "SIMULATION_TIMEOUT",
                f"simulation exceeded timeout of {self.timeout} seconds",
                stderr=self._output_tail(exc.stderr),
                stdout=self._output_tail(exc.stdout),
            ) from exc
        except OSError as exc:
            raise SimulationProcessError(
                "SIMULATION_LAUNCH_FAILED",
                f"could not launch simulation: {exc}",
            ) from exc

        stderr = self._output_tail(completed.stderr)
        stdout = self._output_tail(completed.stdout)
        if completed.returncode != 0:
            raise SimulationProcessError(
                "SIMULATION_PROCESS_FAILED",
                f"simulation exited with status {completed.returncode}",
                returncode=completed.returncode,
                stderr=stderr,
                stdout=stdout,
            )
        if not report_path.is_file():
            raise SimulationProcessError(
                "SIMULATION_REPORT_MISSING",
                "simulation completed without execution_report.json",
                returncode=completed.returncode,
                stderr=stderr,
                stdout=stdout,
            )

        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            report = ExecutionReport.model_validate(payload)
            if not report.execution_success:
                if report.failed_step is None:
                    raise ValueError("failed report requires failed_step")
                matching = [
                    record
                    for record in report.steps
                    if record.step_id == report.failed_step
                ]
                if not matching or matching[-1].success:
                    raise ValueError(
                        "failed report requires a failed-step record"
                    )
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise SimulationProcessError(
                "SIMULATION_REPORT_MALFORMED",
                f"execution report is malformed: {exc}",
                returncode=completed.returncode,
                stderr=stderr,
                stdout=stdout,
            ) from exc
        return report.model_dump(mode="json")


class PlanOrchestrator:
    def __init__(
        self,
        simulation_runner,
        parameter_optimizer,
        scene_optimizer_factory,
        *,
        registry,
        validator,
        scene_preflight,
        max_parameter_iterations=2,
        max_scene_iterations=2,
    ) -> None:
        for name, value in (
            ("max_parameter_iterations", max_parameter_iterations),
            ("max_scene_iterations", max_scene_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.simulation_runner = simulation_runner
        self.parameter_optimizer = parameter_optimizer
        self.scene_optimizer_factory = scene_optimizer_factory
        self.registry = registry
        self.validator = validator
        self.scene_preflight = scene_preflight
        self.max_parameter_iterations = max_parameter_iterations
        self.max_scene_iterations = max_scene_iterations

    def run(
        self,
        plan: AgentPlan,
        config_path,
        artifacts,
        *,
        scene_compile_result=None,
    ) -> dict[str, Any]:
        current_plan = plan.model_copy(deep=True)
        initial_validation = self._validate_plan(current_plan, "initial plan is invalid")
        if not initial_validation.valid:
            raise ValueError("initial plan is invalid")

        report = self._run_once(config_path, current_plan)
        if report["execution_success"]:
            return report

        for _ in range(self.max_parameter_iterations):
            if self.parameter_optimizer is None:
                break
            patch = self.parameter_optimizer.propose(current_plan, report)
            if patch is None:
                break
            if not isinstance(patch, ParameterPatch):
                raise ValueError("parameter optimizer returned a malformed patch")
            if patch.step_id != report["failed_step"]:
                raise ValueError("parameter patch must target the failed step")

            current_plan = apply_parameter_patch(
                current_plan,
                patch,
                self.registry,
            )
            validation = self._validate_plan(
                current_plan,
                "parameter patch produced an invalid plan",
            )
            if not validation.valid:
                raise ValueError("parameter patch produced an invalid plan")
            artifacts.write_plan(current_plan)
            artifacts.write_json(artifacts.validation_path, validation)

            report = self._run_once(config_path, current_plan)
            if report["execution_success"]:
                return report

        if self.max_scene_iterations == 0:
            return report

        if self.scene_optimizer_factory is None:
            scene_optimizer = None
        elif scene_compile_result is None:
            scene_optimizer = self.scene_optimizer_factory(artifacts)
        else:
            scene_optimizer = self.scene_optimizer_factory(
                artifacts,
                scene_compile_result,
            )
        for _ in range(self.max_scene_iterations):
            if scene_optimizer is None:
                return report
            changed = scene_optimizer.optimize_from_execution_report(report)
            if not isinstance(changed, bool):
                raise ValueError("scene optimizer must return a boolean")
            if not changed:
                return report

            if self.scene_preflight is not None:
                preflight = self.scene_preflight(current_plan, artifacts)
                if hasattr(preflight, "model_dump"):
                    preflight_payload = preflight.model_dump(mode="python")
                else:
                    preflight_payload = preflight
                try:
                    preflight = ScenePreflightReport.model_validate(
                        preflight_payload
                    )
                except ValidationError as exc:
                    raise ValueError("scene preflight returned a malformed report") from exc
                artifacts.write_json(artifacts.scene_preflight_path, preflight)
                if not preflight.passed:
                    failed = dict(report)
                    failed["error_code"] = (
                        "SCENE_PREFLIGHT_FAILED_AFTER_OPTIMIZATION"
                    )
                    failed["scene_preflight"] = preflight.model_dump(mode="json")
                    return failed

            report = self._run_once(config_path, current_plan)
            if report["execution_success"]:
                return report
        return report

    def _validate_plan(self, plan: AgentPlan, message: str):
        validation = self.validator.validate(plan)
        if (
            validation.plan_fingerprint != plan_fingerprint(plan)
            or validation.registry_fingerprint
            != registry_fingerprint(self.registry)
        ):
            raise ValueError(f"{message}: validator fingerprints are stale")
        return validation

    def _run_once(self, config_path, plan: AgentPlan) -> dict[str, Any]:
        raw_report = self.simulation_runner.run(config_path)
        if hasattr(raw_report, "model_dump"):
            raw_report = raw_report.model_dump(mode="python")
        try:
            report = ExecutionReport.model_validate(raw_report)
        except ValidationError as exc:
            raise ValueError("simulation runner returned a malformed report") from exc

        if report.execution_success:
            if len(report.steps) != len(plan.actions):
                raise ValueError(
                    "successful execution report must contain every plan step exactly once"
                )
            for record, plan_step in zip(report.steps, plan.actions):
                if (
                    not record.success
                    or not record.controller_completed
                    or record.step_id != plan_step.id
                    or record.action != plan_step.type.value
                    or record.object_id != plan_step.object
                    or record.target_id != plan_step.target
                ):
                    raise ValueError(
                        "successful execution report does not match the ordered plan steps"
                    )
        else:
            if report.failed_step is None:
                raise ValueError("failed execution report requires failed_step")
            matching = [
                record
                for record in report.steps
                if record.step_id == report.failed_step
            ]
            if not matching or matching[-1].success:
                raise ValueError(
                    "failed execution report requires a failed-step verification record"
                )
            plan_step = next(
                (
                    step
                    for step in plan.actions
                    if step.id == report.failed_step
                ),
                None,
            )
            if plan_step is None:
                raise ValueError("execution report references an unknown failed step")
            failed_record = matching[-1]
            if (
                failed_record.action != plan_step.type.value
                or failed_record.object_id != plan_step.object
                or failed_record.target_id != plan_step.target
            ):
                raise ValueError(
                    "failed execution record does not match the current plan step"
                )
        return report.model_dump(mode="json")
