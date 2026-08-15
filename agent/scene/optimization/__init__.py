"""Optimization helpers with lazy Isaac-dependent exports."""

__all__ = ["PositionUpdater", "update_positions_from_json"]


def __getattr__(name):
    if name in __all__:
        from .position_updater import PositionUpdater, update_positions_from_json

        return {
            "PositionUpdater": PositionUpdater,
            "update_positions_from_json": update_positions_from_json,
        }[name]
    raise AttributeError(name)
