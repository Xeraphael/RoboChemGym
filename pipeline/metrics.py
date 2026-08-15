from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np


def summarize_evaluation(
    reports: Sequence[dict[str, Any]], *, percentiles=(50, 90, 95)
) -> dict[str, Any]:
    attempted = len(reports)
    completed = sum(report.get("status") == "completed" for report in reports)
    successful = sum(bool(report.get("success")) for report in reports)
    failed = attempted - successful
    lengths = [int(report.get("length", 0)) for report in reports]
    step_totals: Counter[str] = Counter()
    step_successes: Counter[str] = Counter()
    action_totals: Counter[str] = Counter()
    action_successes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    distances = []
    for report in reports:
        if not report.get("success"):
            failures[str(report.get("failure_code", "UNKNOWN"))] += 1
        for step in report.get("steps", []):
            step_id = str(step["step_id"])
            action = str(step.get("action", "unknown"))
            step_totals[step_id] += 1
            step_successes[step_id] += bool(step.get("success"))
            action_totals[action] += 1
            action_successes[action] += bool(step.get("success"))
            distance = step.get("measurements", {}).get("object_target_distance")
            if distance is not None:
                distances.append(float(distance))
    return {
        "attempted": attempted,
        "completed": completed,
        "successful": successful,
        "failed": failed,
        "task_success_rate": successful / attempted if attempted else 0.0,
        "per_action_step_success_rate": {
            step_id: step_successes[step_id] / total
            for step_id, total in sorted(step_totals.items())
        },
        "per_action_type_success_rate": {
            action: action_successes[action] / total
            for action, total in sorted(action_totals.items())
        },
        "failure_code_distribution": dict(sorted(failures.items())),
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "episode_length_percentiles": {
            str(value): float(np.percentile(lengths, value)) if lengths else 0.0
            for value in percentiles
        },
        "placement_terminal_distance_mean": (
            float(np.mean(distances)) if distances else None
        ),
    }
