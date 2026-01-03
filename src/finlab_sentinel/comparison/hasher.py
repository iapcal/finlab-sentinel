"""Content hashing for DataFrames."""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import pandas as pd
import xxhash


class ContentHasher:
    """Generate content hashes for DataFrames.

    Uses xxhash for fast hashing. The hash considers:
    - Index values and dtype
    - Column names and order
    - All cell values
    - Dtypes of each column
    """

    def __init__(
        self,
        algorithm: Literal["xxhash", "sha256"] = "xxhash",
    ) -> None:
        """Initialize hasher.

        Args:
            algorithm: Hash algorithm to use
        """
        self.algorithm = algorithm

    def hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate content hash for DataFrame.

        Args:
            df: DataFrame to hash

        Returns:
            Hex string hash
        """
        hasher = self._get_hasher()

        # Hash index
        self._hash_index(df.index, hasher)

        # Hash columns (names and order)
        self._hash_columns(df.columns, hasher)

        # Hash dtypes
        for col in df.columns:
            hasher.update(str(df[col].dtype).encode())

        # Hash values efficiently
        self._hash_values(df, hasher)

        return hasher.hexdigest()

    def hash_series(self, series: pd.Series) -> str:
        """Generate content hash for Series.

        Args:
            series: Series to hash

        Returns:
            Hex string hash
        """
        hasher = self._get_hasher()

        # Hash index
        self._hash_index(series.index, hasher)

        # Hash name
        hasher.update(str(series.name).encode())

        # Hash dtype
        hasher.update(str(series.dtype).encode())

        # Hash values
        self._hash_series_values(series, hasher)

        return hasher.hexdigest()

    def _get_hasher(self):
        """Get appropriate hasher object."""
        if self.algorithm == "xxhash":
            return xxhash.xxh64()
        else:
            import hashlib

            return hashlib.sha256()

    def _hash_index(self, index: pd.Index, hasher) -> None:
        """Hash index values and properties."""
        hasher.update(str(index.dtype).encode())
        hasher.update(str(len(index)).encode())

        # For efficiency, hash the numpy array representation
        if hasattr(index, "to_numpy"):
            arr = index.to_numpy()
            if arr.dtype.kind in ("U", "O"):
                # String/object dtype - convert to bytes
                hasher.update(str(arr.tolist()).encode())
            else:
                hasher.update(arr.tobytes())
        else:
            hasher.update(str(index.tolist()).encode())

    def _hash_columns(self, columns: pd.Index, hasher) -> None:
        """Hash column names and order."""
        for col in columns:
            hasher.update(str(col).encode())

    def _hash_values(self, df: pd.DataFrame, hasher) -> None:
        """Hash DataFrame values efficiently."""
        for col in df.columns:
            self._hash_series_values(df[col], hasher)

    def _hash_series_values(self, series: pd.Series, hasher) -> None:
        """Hash Series values with NA handling."""
        # Create a buffer to capture the representation
        buf = io.BytesIO()

        # Handle different dtypes
        if pd.api.types.is_numeric_dtype(series.dtype):
            # Numeric: use numpy bytes representation
            arr = series.to_numpy()
            # Replace NaN with a sentinel for consistent hashing
            if np.issubdtype(arr.dtype, np.floating):
                nan_mask = pd.isna(series)
                if nan_mask.any():
                    # Hash NaN positions separately
                    hasher.update(nan_mask.to_numpy().tobytes())
                    # Replace NaN with 0 for value hashing
                    arr = np.where(nan_mask, 0.0, arr)
            buf.write(arr.tobytes())
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            # Datetime: use int64 representation
            arr = series.view("int64").to_numpy()
            buf.write(arr.tobytes())
        else:
            # Object/string: use string representation
            # Need to handle NA values carefully
            values = []
            for v in series:
                if pd.isna(v):
                    # Encode the type of NA for distinction
                    values.append(f"__NA__{_get_na_type(v)}__")
                else:
                    values.append(str(v))
            buf.write(str(values).encode())

        hasher.update(buf.getvalue())


def _get_na_type(val) -> str:
    """Identify the type of NA value.

    Args:
        val: Value to check

    Returns:
        String identifying NA type
    """
    if val is None:
        return "None"
    if val is pd.NA:
        return "pd.NA"
    if isinstance(val, float) and np.isnan(val):
        return "np.nan"
    return "unknown"
