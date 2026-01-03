"""Base class for anomaly handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from finlab_sentinel.comparison.report import AnomalyReport


class AnomalyHandler(ABC):
    """Base class for anomaly handlers.

    Handlers determine what happens when a data anomaly is detected.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name."""
        ...

    @abstractmethod
    def handle(
        self,
        report: AnomalyReport,
        cached_data: pd.DataFrame,
        new_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Handle anomaly and return DataFrame to use.

        Args:
            report: Detailed anomaly report
            cached_data: Previously cached DataFrame
            new_data: New DataFrame from finlab

        Returns:
            DataFrame to return to caller
        """
        ...
