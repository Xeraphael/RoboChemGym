def anchor_state_suffix(anchor: str) -> str:
    if not isinstance(anchor, str):
        raise TypeError("anchor must be a string")
    if anchor == "RevoluteJoint":
        return "revolute_joint_position"
    return anchor if anchor.endswith("_position") else f"{anchor}_position"
