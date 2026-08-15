from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, Mapping

from agent.action.plan_execution.interfaces import ActionAdapter, ActionVerifier
from agent.action.plan_execution.models import (
    ExecutionReport,
    StepExecutionRecord,
    VerificationRequest,
    VerificationResult,
)
from agent.planning.models import ActionStep, AgentPlan, CoverageLevel


_FILTERED_STATE_KEYS = frozenset({"camera_data", "camera_display"})
_DEFAULT_ACTION_TIMEOUT = 10000


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


class SequentialPlanExecutor:
    def __init__(
        self,
        plan: AgentPlan,
        adapters: Mapping[str, ActionAdapter],
        verifiers: Mapping[str, ActionVerifier],
        *,
        coverage_by_step: Mapping[str, CoverageLevel | str] | None = None,
        action_timeouts: Mapping[str, int] | None = None,
        max_retries: int = 1,
    ) -> None:
        self._plan = deepcopy(plan)

        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be a nonnegative integer")
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")

        normalized_timeouts: dict[str, int] = {}
        for action_type, timeout in (action_timeouts or {}).items():
            key = str(_enum_value(action_type))
            if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
                raise ValueError(f"action timeout for {key} must be positive")
            normalized_timeouts[key] = timeout

        normalized_adapters = {
            str(_enum_value(action_type)): adapter
            for action_type, adapter in adapters.items()
        }
        normalized_verifiers = {
            str(_enum_value(action_type)): verifier
            for action_type, verifier in verifiers.items()
        }
        required_types = {
            str(_enum_value(step.type))
            for step in self._plan.actions
        }
        missing_adapters = sorted(required_types - normalized_adapters.keys())
        if missing_adapters:
            raise ValueError(f"missing adapters for action types: {', '.join(missing_adapters)}")
        missing_verifiers = sorted(required_types - normalized_verifiers.keys())
        if missing_verifiers:
            raise ValueError(f"missing verifiers for action types: {', '.join(missing_verifiers)}")

        normalized_coverage: dict[str, CoverageLevel] = {}
        for step_id, coverage in (coverage_by_step or {}).items():
            try:
                normalized_coverage[step_id] = CoverageLevel(_enum_value(coverage))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid coverage level for {step_id}: {coverage!r}") from exc

        self.adapters = normalized_adapters
        self.verifiers = normalized_verifiers
        self.coverage_by_step = normalized_coverage
        self.max_retries = max_retries
        self.action_timeouts = normalized_timeouts
        self._instance_names = {
            scene_object.id: scene_object.instance_name
            for scene_object in self._plan.scene.objects
        }
        self._pending_cleanup_adapter: ActionAdapter | None = None
        self._initialize_runtime_state()

    def step(self, state: Mapping[str, Any]) -> Any:
        if self.done or self._pending_cleanup_adapter is not None:
            return None

        self.frame += 1
        if not self._prepared:
            if not self._prepare_current_attempt():
                return None

        try:
            snapshot = self._snapshot_state(state)
        except Exception as exc:
            self._terminalize_external_error(
                "STATE_SNAPSHOT_ERROR",
                exc,
                controller_completed=False,
            )
            return None
        if self._episode_initial_state is None:
            self._episode_initial_state = snapshot
        if self._attempt_pre_state is None:
            self._attempt_pre_state = snapshot
        self._attempt_state_history.append(snapshot)

        adapter = self._current_adapter
        try:
            action = adapter.step(deepcopy(snapshot))
        except Exception as exc:
            self._terminalize_external_error(
                "ADAPTER_STEP_ERROR",
                exc,
                controller_completed=False,
            )
            return None
        self._attempt_elapsed_frames += 1

        try:
            controller_completed = adapter.is_done()
        except Exception as exc:
            self._terminalize_external_error(
                "ADAPTER_STATUS_ERROR",
                exc,
                controller_completed=False,
            )
            return action

        if controller_completed:
            try:
                request = self._build_verification_request(snapshot)
            except Exception as exc:
                self._terminalize_external_error(
                    "VERIFICATION_REQUEST_INVALID",
                    exc,
                    controller_completed=True,
                )
                return action
            try:
                raw_result = self._current_verifier.verify(request)
            except Exception as exc:
                self._terminalize_external_error(
                    "VERIFIER_ERROR",
                    exc,
                    controller_completed=True,
                )
                return action
            try:
                result = self._coerce_verification_result(raw_result)
            except Exception as exc:
                self._terminalize_external_error(
                    "VERIFICATION_RESULT_INVALID",
                    exc,
                    controller_completed=True,
                )
                return action
            self._resolve_attempt(
                controller_completed=True,
                result=result,
                post_state=snapshot,
            )
        elif self._attempt_elapsed_frames >= self._current_timeout:
            result = VerificationResult(
                success=False,
                code="ACTION_TIMEOUT",
                message="action adapter did not complete before its frame limit",
                verification_level="controller_state",
                measurements={"elapsed_frames": self._attempt_elapsed_frames},
            )
            self._resolve_attempt(
                controller_completed=False,
                result=result,
                post_state=snapshot,
            )

        return action

    def reset(self) -> None:
        if self._pending_cleanup_adapter is not None:
            pending_adapter = self._pending_cleanup_adapter
            try:
                pending_adapter.reset()
            except Exception:
                raise
            self._pending_cleanup_adapter = None
            self._initialize_runtime_state()
            return

        if self._adapter_active:
            reset_error = self._reset_active_adapter()
            if reset_error is not None:
                if not self.done:
                    self._terminalize(
                        self._adapter_reset_result(reset_error),
                        controller_completed=False,
                    )
                raise reset_error

        self._initialize_runtime_state()

    def fail(self, code: str, message: str = "") -> None:
        if self.done:
            return
        reset_error = self._reset_active_adapter()
        measurements = {}
        if reset_error is not None:
            measurements["cleanup_exception_type"] = type(reset_error).__name__
        self._terminalize(
            VerificationResult(
                success=False,
                code=code,
                message=message,
                measurements=measurements,
                verification_level="simulation_state",
            ),
            controller_completed=False,
        )

    @property
    def plan(self) -> AgentPlan:
        return deepcopy(self._plan)

    def _initialize_runtime_state(self) -> None:
        self.index = 0
        self.frame = 0
        self.done = not self._plan.actions
        self.success = self.done
        self.report = ExecutionReport(execution_success=self.done)
        self._attempt_count = 1
        self._attempt_elapsed_frames = 0
        self._prepared = False
        self._adapter_active = False
        self._attempt_pre_state = None
        self._attempt_state_history = []
        self._episode_initial_state = None
        self._step_start_frame = None

    @property
    def current_step(self) -> ActionStep | None:
        return None if self.done else deepcopy(self._current_step)

    @property
    def current_adapter(self) -> ActionAdapter | None:
        return None if self.done else self._current_adapter

    @property
    def current_verifier(self) -> ActionVerifier | None:
        return None if self.done else self._current_verifier

    @property
    def current_action(self) -> str | None:
        return None if self.done else self._current_action_type

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    @property
    def attempt_elapsed_frames(self) -> int:
        return self._attempt_elapsed_frames

    @property
    def plan_instance_names(self) -> dict[str, str]:
        return deepcopy(self._instance_names)

    @property
    def _current_step(self) -> ActionStep:
        return self._plan.actions[self.index]

    @property
    def _current_action_type(self) -> str:
        return str(_enum_value(self._current_step.type))

    @property
    def _current_adapter(self) -> ActionAdapter:
        return self.adapters[self._current_action_type]

    @property
    def _current_verifier(self) -> ActionVerifier:
        return self.verifiers[self._current_action_type]

    @property
    def _current_timeout(self) -> int:
        return self.action_timeouts.get(
            self._current_action_type,
            _DEFAULT_ACTION_TIMEOUT,
        )

    def _snapshot_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = {
            key: value
            for key, value in state.items()
            if key not in _FILTERED_STATE_KEYS
        }
        snapshot["plan_instance_names"] = self._instance_names
        return deepcopy(snapshot)

    def _prepare_current_attempt(self) -> bool:
        if self._step_start_frame is None:
            self._step_start_frame = self.frame
        self._adapter_active = True
        try:
            self._current_adapter.prepare(deepcopy(self._current_step), self)
        except Exception as exc:
            self._terminalize_external_error(
                "ADAPTER_PREPARE_ERROR",
                exc,
                controller_completed=False,
            )
            return False
        self._prepared = True
        return True

    def _resolve_attempt(
        self,
        *,
        controller_completed: bool,
        result: VerificationResult,
        post_state: Mapping[str, Any],
    ) -> None:
        reset_error = self._reset_active_adapter()
        if reset_error is not None:
            self._terminalize(
                self._adapter_reset_result(reset_error, result),
                controller_completed=controller_completed,
            )
            return

        if not result.success and self._attempt_count <= self.max_retries:
            self._attempt_count += 1
            self._reset_attempt_state(reset_count=False)
            self._prepare_current_attempt()
            return

        if not result.success:
            self._terminalize(result, controller_completed=controller_completed)
            return

        record = self._build_record(controller_completed, result)
        completed_steps = [*self.report.steps, record]
        self.index += 1
        if self.index == len(self._plan.actions):
            self.done = True
            self.success = True
            self.report = ExecutionReport(
                execution_success=True,
                steps=completed_steps,
            )
        else:
            self.report = ExecutionReport(steps=completed_steps)
            self._reset_attempt_state(reset_step_start=True)

    def _build_record(
        self,
        controller_completed: bool,
        result: VerificationResult,
    ) -> StepExecutionRecord:
        step = self._current_step
        return StepExecutionRecord(
            step_id=step.id,
            action=self._current_action_type,
            object_id=step.object,
            target_id=step.target,
            coverage_level=self._coverage_for(step).value,
            adapter=type(self._current_adapter).__name__,
            verifier=type(self._current_verifier).__name__,
            attempt_count=self._attempt_count,
            success=result.success,
            start_frame=self._step_start_frame if self._step_start_frame is not None else self.frame,
            end_frame=self.frame,
            controller_completed=controller_completed,
            semantic_requirements=self._semantic_requirements(step),
            verification=self._detach_verification_result(result),
        )

    def _build_verification_request(
        self,
        post_state: Mapping[str, Any],
    ) -> VerificationRequest:
        return VerificationRequest(
            step=deepcopy(self._current_step),
            pre_state=deepcopy(self._attempt_pre_state),
            post_state=deepcopy(dict(post_state)),
            state_history=deepcopy(self._attempt_state_history),
            episode_initial_state=deepcopy(self._episode_initial_state),
        )

    def _coerce_verification_result(self, raw_result: Any) -> VerificationResult:
        if isinstance(raw_result, VerificationResult):
            payload = deepcopy(raw_result.model_dump(mode="python"))
        else:
            payload = deepcopy(raw_result)
        return VerificationResult.model_validate(payload)

    def _detach_verification_result(
        self,
        result: VerificationResult,
    ) -> VerificationResult:
        return VerificationResult.model_validate(
            deepcopy(result.model_dump(mode="python"))
        )

    def _reset_active_adapter(self) -> Exception | None:
        if not self._adapter_active:
            return None
        adapter = self._current_adapter
        try:
            adapter.reset()
        except Exception as exc:
            self._pending_cleanup_adapter = adapter
            return exc
        finally:
            self._adapter_active = False
            self._prepared = False
        return None

    def _terminalize_external_error(
        self,
        code: str,
        error: Exception,
        *,
        controller_completed: bool,
    ) -> None:
        cleanup_error = self._reset_active_adapter()
        measurements = {"exception_type": type(error).__name__}
        if cleanup_error is not None:
            measurements["cleanup_exception_type"] = type(cleanup_error).__name__
        self._terminalize(
            VerificationResult(
                success=False,
                code=code,
                measurements=measurements,
            ),
            controller_completed=controller_completed,
        )

    def _adapter_reset_result(
        self,
        error: Exception,
        prior_result: VerificationResult | None = None,
    ) -> VerificationResult:
        measurements = {"exception_type": type(error).__name__}
        if prior_result is not None:
            measurements["prior_verification_code"] = prior_result.code
        return VerificationResult(
            success=False,
            code="ADAPTER_RESET_ERROR",
            measurements=measurements,
        )

    def _terminalize(
        self,
        result: VerificationResult,
        *,
        controller_completed: bool,
    ) -> None:
        if self.done:
            return
        failed_step = self._current_step.id
        record = self._build_record(controller_completed, result)
        self._adapter_active = False
        self._prepared = False
        self.done = True
        self.success = False
        self.report = ExecutionReport(
            failed_step=failed_step,
            steps=[*self.report.steps, record],
        )

    def _coverage_for(self, step: ActionStep) -> CoverageLevel:
        return self.coverage_by_step.get(step.id, CoverageLevel.SUPPORTED)

    def _semantic_requirements(self, step: ActionStep) -> list[str]:
        return [
            annotation.source_text
            for annotation in self._plan.semantic_annotations
            if step.id in annotation.step_ids
        ]

    def _reset_attempt_state(
        self,
        *,
        reset_count: bool = True,
        reset_step_start: bool = False,
    ) -> None:
        if reset_count:
            self._attempt_count = 1
        self._attempt_elapsed_frames = 0
        self._prepared = False
        self._attempt_pre_state = None
        self._attempt_state_history = []
        if reset_step_start:
            self._step_start_frame = None
