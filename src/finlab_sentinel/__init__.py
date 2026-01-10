"""finlab-sentinel: Defensive monitoring layer for finlab data.get API."""

from datetime import datetime

from finlab_sentinel.core.hooks import (
    clear_preprocess_hooks,
    register_preprocess_hook,
    unregister_preprocess_hook,
)
from finlab_sentinel.core.patcher import disable, enable, is_enabled
from finlab_sentinel.core.time_travel import TimeTravelContext
from finlab_sentinel.exceptions import (
    DataAnomalyError,
    NoHistoricalDataError,
    SentinelError,
    TimeTravelError,
)

__version__ = "0.1.7"

__all__ = [
    "__version__",
    "enable",
    "disable",
    "is_enabled",
    "SentinelError",
    "DataAnomalyError",
    "TimeTravelError",
    "NoHistoricalDataError",
    "register_preprocess_hook",
    "unregister_preprocess_hook",
    "clear_preprocess_hooks",
    "set_time_travel",
    "exit_time_travel",
    "get_time_travel_status",
]


def set_time_travel(target_time: datetime) -> None:
    """Set time travel mode to specified datetime.

    After calling this, all data.get() calls will return historical backup data
    from the specified time point instead of fetching fresh data.

    Args:
        target_time: Target datetime to travel to

    Example:
        >>> import finlab_sentinel as fs
        >>> from datetime import datetime
        >>> fs.enable()
        >>> fs.set_time_travel(datetime(2024, 1, 5, 14, 30))
        >>> data.get("price:收盤價")  # Returns historical data
        >>> fs.exit_time_travel()
    """
    ctx = TimeTravelContext.get_instance()
    ctx.set_target_time(target_time)


def exit_time_travel() -> None:
    """Exit time travel mode and return to normal data retrieval."""
    ctx = TimeTravelContext.get_instance()
    ctx.clear()


def get_time_travel_status() -> dict:
    """Get current time travel status.

    Returns:
        Dictionary with 'enabled' (bool) and 'target_time' (str | None) keys

    Example:
        >>> fs.get_time_travel_status()
        {'enabled': True, 'target_time': '2024-01-05T14:30:00'}
    """
    ctx = TimeTravelContext.get_instance()
    return {
        "enabled": ctx.is_active(),
        "target_time": ctx.target_time.isoformat() if ctx.target_time else None,
    }
