from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Protocol

from agent.action.plan_execution.models import VerificationRequest, VerificationResult
from agent.planning.models import ActionStep

if TYPE_CHECKING:
    from agent.action.plan_execution.executor import SequentialPlanExecutor


class ActionAdapter(Protocol):
    def prepare(
        self,
        step: ActionStep,
        context: "SequentialPlanExecutor",
    ) -> None:
        ...

    def step(self, state: Mapping[str, Any]) -> Any:
        ...

    def is_done(self) -> bool:
        ...

    def reset(self) -> None:
        ...


class ActionVerifier(Protocol):
    def verify(
        self,
        request: VerificationRequest,
    ) -> VerificationResult | Mapping[str, Any]:
        ...


StepVerifier = ActionVerifier
Verifier = ActionVerifier
