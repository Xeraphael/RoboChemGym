def matching_anchor_prims(Usd, instance_prim, instance_path: str, anchor: str):
    if not isinstance(anchor, str):
        raise TypeError("anchor must be a string")
    if not instance_prim or not instance_prim.IsValid():
        return ()

    descendant_prefix = f"{instance_path}/"
    matches = []
    for descendant in Usd.PrimRange(
        instance_prim,
        Usd.TraverseInstanceProxies(),
    ):
        descendant_path = descendant.GetPath().pathString
        if not descendant_path.startswith(descendant_prefix):
            continue
        relative_path = descendant_path[len(descendant_prefix):]
        if "/" in anchor:
            matched = relative_path == anchor
        else:
            matched = descendant.GetName() == anchor
        if matched:
            matches.append(descendant)
    return tuple(matches)
