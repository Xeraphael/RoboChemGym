from typing import Dict, Type

from tasks.example_protocol_task import ExampleProtocolTask
from tasks.all_task import AllTask
from tasks.base_task import BaseTask


_task_registry: Dict[str, Type[BaseTask]] = {}


def register_task(name: str, task_class: Type[BaseTask]):
    _task_registry[name] = task_class


def create_task(task_name: str, *args, **kwargs) -> BaseTask:
    if task_name not in _task_registry:
        raise ValueError(f"unknown task: {task_name}")
    return _task_registry[task_name](*args, **kwargs)


register_task("all", AllTask)
register_task("example_protocol", ExampleProtocolTask)
