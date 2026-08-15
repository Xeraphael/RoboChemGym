import os
import sys
from pathlib import Path
from typing import Dict, Type

from controllers.base_controller import BaseController
from controllers.plan_executor import PlanExecutorController
from controllers.policy_controller import PolicyController


_controller_registry: Dict[str, Type[BaseController]] = {}


def register_controller(name: str, controller_class: Type[BaseController]):
    _controller_registry[name] = controller_class


def create_controller(controller_name: str, *args, **kwargs) -> BaseController:
    if controller_name not in _controller_registry:
        raise ValueError(f"unknown controller: {controller_name}")

    controller = _controller_registry[controller_name](*args, **kwargs)
    if os.getenv("AGENT_MONITOR_MODE") != "true":
        return controller

    log_file = os.getenv("AGENT_LOG_FILE")
    if not log_file:
        return controller

    try:
        agent_action_path = Path(__file__).parent.parent / "agent" / "action"
        if str(agent_action_path) not in sys.path:
            sys.path.insert(0, str(agent_action_path))
        from monitoring.execution_monitor import ExecutionMonitor

        return ExecutionMonitor(
            controller=controller,
            log_file=log_file,
            frame_interval=10,
            enable_verification=True,
            strict_mode=True,
        )
    except Exception as exc:
        print(f"[ControllerFactory] monitor disabled: {exc}")
        return controller


register_controller("plan_executor", PlanExecutorController)
register_controller("policy", PolicyController)
