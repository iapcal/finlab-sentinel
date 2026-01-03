"""Callback-based anomaly handler."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

from finlab_sentinel.handlers.base import AnomalyHandler

if TYPE_CHECKING:
    from finlab_sentinel.comparison.report import AnomalyReport

logger = logging.getLogger(__name__)


class CallbackHandler(AnomalyHandler):
    """Handler that executes callback function on anomaly.

    The callback is called with the AnomalyReport, then the fallback
    handler determines what data to return.
    """

    def __init__(
        self,
        callback: Callable[[AnomalyReport], None],
        fallback_handler: AnomalyHandler,
    ) -> None:
        """Initialize callback handler.

        Args:
            callback: Function to call with anomaly report
            fallback_handler: Handler to use for data return decision
        """
        self.callback = callback
        self.fallback_handler = fallback_handler

    @property
    def name(self) -> str:
        return f"callback+{self.fallback_handler.name}"

    def handle(
        self,
        report: AnomalyReport,
        cached_data: pd.DataFrame,
        new_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Execute callback and delegate to fallback handler.

        Args:
            report: Anomaly report
            cached_data: Previously cached DataFrame
            new_data: New DataFrame

        Returns:
            DataFrame from fallback handler
        """
        # Execute callback
        try:
            self.callback(report)
        except Exception as e:
            # Log but don't fail on callback errors
            logger.error(
                f"Callback failed for anomaly in {report.dataset}: {e}",
                exc_info=True,
            )

        # Delegate to fallback handler
        return self.fallback_handler.handle(report, cached_data, new_data)


def create_handler_from_config(
    behavior: str,
    callback: Callable[[AnomalyReport], None] | None = None,
) -> AnomalyHandler:
    """Create handler from configuration.

    Args:
        behavior: Behavior string from config
        callback: Optional callback function

    Returns:
        Configured handler
    """
    from finlab_sentinel.handlers.exception import RaiseExceptionHandler
    from finlab_sentinel.handlers.warning import (
        WarnReturnCachedHandler,
        WarnReturnNewHandler,
    )

    # Create base handler
    base_handler: AnomalyHandler
    if behavior == "raise":
        base_handler = RaiseExceptionHandler()
    elif behavior == "warn_return_cached":
        base_handler = WarnReturnCachedHandler()
    elif behavior == "warn_return_new":
        base_handler = WarnReturnNewHandler()
    else:
        # Default to raise
        logger.warning(f"Unknown behavior '{behavior}', defaulting to 'raise'")
        base_handler = RaiseExceptionHandler()

    # Wrap with callback if provided
    if callback is not None:
        return CallbackHandler(callback, base_handler)

    return base_handler
