"""Custom exceptions for finlab-sentinel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finlab_sentinel.comparison.report import AnomalyReport


class SentinelError(Exception):
    """Base exception for finlab-sentinel."""


class SentinelAlreadyEnabledError(SentinelError):
    """Raised when trying to enable sentinel when it's already enabled."""


class SentinelNotEnabledError(SentinelError):
    """Raised when trying to disable sentinel when it's not enabled."""


class ConfigurationError(SentinelError):
    """Raised when there's a configuration error."""


class StorageError(SentinelError):
    """Raised when there's a storage-related error."""


class BaselineChangedError(StorageError):
    """Raised when a dataset's latest backup changed before a write.

    Writes that must not supersede a baseline they did not read (e.g. a
    concurrent accept or patch restore) raise this; nothing is written.
    """


class ComparisonError(SentinelError):
    """Raised when there's an error during data comparison."""


class DataAnomalyError(SentinelError):
    """Raised when a data anomaly is detected."""

    def __init__(self, report: AnomalyReport) -> None:
        self.report = report
        super().__init__(str(report))

    def __str__(self) -> str:
        return f"Data anomaly detected: {self.report.summary}"


class PatchNotFoundError(SentinelError):
    """Raised when a requested patch does not exist."""


class PatchRestoreError(SentinelError):
    """Raised when a patch cannot be restored.

    The dataset's baseline is left unchanged when this is raised.
    """


class TimeTravelError(SentinelError):
    """Base exception for time travel related errors."""


class NoHistoricalDataError(TimeTravelError):
    """Raised when no historical backup exists for time travel request."""
