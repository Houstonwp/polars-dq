from dataclasses import dataclass
from enum import Enum


class ActionLevel(str, Enum):
    WARN = "warning"
    STOP = "stop"
    NOTIFY = "notify"


@dataclass
class Action:
    warn_atol: float | None = None
    warn_rtol: float | None = None
    stop_atol: float | None = None
    stop_rtol: float | None = None
    notify_atol: float | None = None
    notify_rtol: float | None = None
