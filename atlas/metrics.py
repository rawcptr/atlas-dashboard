from enum import Enum
from typing import Optional


class AutoMetric(Enum):
    THROUGHPUT = "throughput"  # samples / sec
    STEP_TIME = "step_time"  # sec / step (averaged over delta)
    MFU = "mfu"  # percent


class Metric(Enum):
    MFU = "mfu"
    THROUGHPUT = "throughput"
    STEP_TIME = "step_time"


def _ema(prev: Optional[float], value: float, alpha: float) -> float:
    """Exponential Moving Average."""
    if prev is None:
        return value
    return alpha * value + (1.0 - alpha) * prev
