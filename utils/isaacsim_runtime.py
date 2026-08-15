import sys
from collections.abc import Iterable


DRIVER_CHECK_SETTING_PREFIX = "--/rtx/verifyDriverVersion/enabled="
DRIVER_CHECK_BYPASS_ARG = f"{DRIVER_CHECK_SETTING_PREFIX}false"


def prepare_isaacsim_argv(kit_args: Iterable[str] | None = None) -> list[str]:
    if isinstance(kit_args, (str, bytes)):
        raise TypeError("kit_args must be an iterable of strings, not str or bytes")

    program = sys.argv[0] if sys.argv else ""
    forwarded_args = list(sys.argv[1:] if kit_args is None else kit_args)
    prepared = [program, *forwarded_args]

    if not any(
        arg.startswith(DRIVER_CHECK_SETTING_PREFIX) for arg in prepared[1:]
    ):
        prepared.append(DRIVER_CHECK_BYPASS_ARG)

    sys.argv[:] = prepared
    return prepared
