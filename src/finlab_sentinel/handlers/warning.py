"""Warning-based anomaly handlers."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import pandas as pd

from finlab_sentinel.handlers.base import AnomalyHandler

if TYPE_CHECKING:
    from finlab_sentinel.comparison.report import AnomalyReport

logger = logging.getLogger(__name__)


class DataAnomalyWarning(UserWarning):
    """Warning issued when data anomaly is detected."""

    pass


class WarnReturnCachedHandler(AnomalyHandler):
    """Handler that warns and returns cached data."""

    @property
    def name(self) -> str:
        return "warn_return_cached"

    def handle(
        self,
        report: AnomalyReport,
        cached_data: pd.DataFrame,
        new_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Issue warning and return cached data.

        Args:
            report: Anomaly report
            cached_data: Previously cached DataFrame (returned)
            new_data: New DataFrame (discarded)

        Returns:
            The cached DataFrame
        """
        warning_msg = (
            f"Data anomaly detected for {report.dataset}: "
            f"{report.comparison_result.summary()}. "
            "Using cached data instead of new data."
        )

        warnings.warn(warning_msg, DataAnomalyWarning, stacklevel=4)
        logger.warning(warning_msg)

        return cached_data


class WarnReturnNewHandler(AnomalyHandler):
    """Handler that warns but returns new data."""

    @property
    def name(self) -> str:
        return "warn_return_new"

    def handle(
        self,
        report: AnomalyReport,
        cached_data: pd.DataFrame,
        new_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Issue warning but return new data.

        Args:
            report: Anomaly report
            cached_data: Previously cached DataFrame (discarded)
            new_data: New DataFrame (returned)

        Returns:
            The new DataFrame
        """
        warning_msg = (
            f"Data anomaly detected for {report.dataset}: "
            f"{report.comparison_result.summary()}. "
            "Returning new data despite anomaly."
        )

        warnings.warn(warning_msg, DataAnomalyWarning, stacklevel=4)
        logger.warning(warning_msg)

        return new_data
