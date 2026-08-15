from importlib import import_module


__all__ = [
    "PickController",
    "PlaceController",
    "StirController",
    "PourController",
    "OpenController",
    "CloseController",
    "ShakeController",
    "PressController",
    "PressZController",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".atomic_actions", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *__all__])
