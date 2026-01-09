"""Time travel context for historical data retrieval."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TimeTravelContext:
    """Global time travel context (thread-safe singleton).

    This context allows data.get() calls to retrieve historical backups
    instead of fetching fresh data from the API.
    """

    _instance: TimeTravelContext | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize time travel context."""
        self._target_time: datetime | None = None

    @classmethod
    def get_instance(cls) -> TimeTravelContext:
        """Get singleton instance.

        Returns:
            The singleton TimeTravelContext instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_target_time(self, dt: datetime) -> None:
        """Set time travel target time.

        Args:
            dt: Target datetime to travel to
        """
        self._target_time = dt

    def clear(self) -> None:
        """Exit time travel mode and return to normal operation."""
        self._target_time = None

    def is_active(self) -> bool:
        """Check if time travel mode is active.

        Returns:
            True if in time travel mode, False otherwise
        """
        return self._target_time is not None

    @property
    def target_time(self) -> datetime | None:
        """Get current target time.

        Returns:
            Target datetime or None if not in time travel mode
        """
        return self._target_time

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset singleton instance (for testing only)."""
        with cls._lock:
            cls._instance = None
