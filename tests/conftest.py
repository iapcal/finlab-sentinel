"""Pytest fixtures for finlab-sentinel tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from finlab_sentinel.config.schema import SentinelConfig


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample DataFrame mimicking finlab data."""
    dates = pd.date_range("2025-01-01", periods=10)
    symbols = ["2330", "2317", "2454", "2412"]
    data = np.random.default_rng(42).random((10, 4)) * 100 + 500
    return pd.DataFrame(data, index=dates, columns=symbols)


@pytest.fixture
def sample_df_modified(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Create modified version of sample DataFrame."""
    df = sample_df.copy()
    # Modify some values
    df.iloc[0, 0] = 999.99
    df.iloc[5, 2] = 888.88
    return df


@pytest.fixture
def sample_df_appended(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Create sample DataFrame with appended rows."""
    new_dates = pd.date_range("2025-01-11", periods=3)
    new_data = np.random.default_rng(43).random((3, 4)) * 100 + 500
    new_df = pd.DataFrame(new_data, index=new_dates, columns=sample_df.columns)
    return pd.concat([sample_df, new_df])


@pytest.fixture
def sample_df_with_na() -> pd.DataFrame:
    """Create sample DataFrame with various NA types."""
    dates = pd.date_range("2025-01-01", periods=5)
    return pd.DataFrame(
        {
            "col1": [1.0, np.nan, 3.0, None, 5.0],
            "col2": [pd.NA, 2.0, 3.0, 4.0, pd.NA],
            "col3": ["a", "b", None, "d", "e"],
        },
        index=dates,
    )


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Path:
    """Create temporary storage directory for tests."""
    storage_path = tmp_path / ".finlab-sentinel"
    storage_path.mkdir(parents=True)
    return storage_path


@pytest.fixture
def sample_config(tmp_storage: Path) -> SentinelConfig:
    """Create sample configuration for tests."""
    from finlab_sentinel.config.schema import SentinelConfig, StorageConfig

    return SentinelConfig(
        storage=StorageConfig(path=tmp_storage),
    )


@pytest.fixture
def mock_finlab():
    """Mock finlab package for testing."""
    mock = MagicMock()
    mock.data = MagicMock()
    return mock


@pytest.fixture
def parquet_storage(tmp_storage: Path):
    """Create ParquetStorage instance for testing."""
    from finlab_sentinel.storage.parquet import ParquetStorage

    return ParquetStorage(base_path=tmp_storage)


@pytest.fixture
def cli_runner():
    """Create Typer CLI test runner."""
    from typer.testing import CliRunner

    return CliRunner()
