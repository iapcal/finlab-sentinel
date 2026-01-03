"""Exception-raising anomaly handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from finlab_sentinel.exceptions import DataAnomalyError
from finlab_sentinel.handlers.base import AnomalyHandler

if TYPE_CHECKING:
    from finlab_sentinel.comparison.report import AnomalyReport


class RaiseExceptionHandler(AnomalyHandler):
    """Handler that raises exception on anomaly detection."""

    @property
    def name(self) -> str:
        return "raise"

    def handle(
        self,
        report: AnomalyReport,
        cached_data: pd.DataFrame,
        new_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Raise DataAnomalyError with report.

        Args:
            report: Anomaly report
            cached_data: Cached data (not used)
            new_data: New data (not used)

        Raises:
            DataAnomalyError: Always raised with the anomaly report
        """
        raise DataAnomalyError(report)
