import numpy as np


DEFAULT_MAX_JOINT_STEP = 0.05


def bound_joint_position_step(
    action,
    current_joint_positions,
    max_step=DEFAULT_MAX_JOINT_STEP,
):
    targets = getattr(action, "joint_positions", None)
    if targets is None:
        return action
    try:
        current = np.asarray(current_joint_positions, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        current = np.array([], dtype=float)

    bounded = []
    for index, target in enumerate(targets):
        if target is None:
            bounded.append(None)
            continue
        if index >= current.size or not np.isfinite(current[index]):
            bounded.append(None)
            continue
        try:
            target = float(target)
        except (TypeError, ValueError):
            target = current[index]
        if not np.isfinite(target):
            target = current[index]
        bounded.append(
            float(
                np.clip(
                    target,
                    current[index] - max_step,
                    current[index] + max_step,
                )
            )
        )
    action.joint_positions = bounded
    return action
