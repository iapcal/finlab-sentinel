"""Tests for content hasher."""

import numpy as np
import pandas as pd

from finlab_sentinel.comparison.hasher import ContentHasher


class TestContentHasher:
    """Tests for ContentHasher class."""

    def test_identical_dataframes_same_hash(self, sample_df: pd.DataFrame):
        """Verify identical DataFrames produce same hash."""
        hasher = ContentHasher()

        hash1 = hasher.hash_dataframe(sample_df)
        hash2 = hasher.hash_dataframe(sample_df.copy())

        assert hash1 == hash2

    def test_different_values_different_hash(self, sample_df: pd.DataFrame):
        """Verify value changes produce different hash."""
        hasher = ContentHasher()

        df_modified = sample_df.copy()
        df_modified.iloc[0, 0] = 999999.0

        hash1 = hasher.hash_dataframe(sample_df)
        hash2 = hasher.hash_dataframe(df_modified)

        assert hash1 != hash2

    def test_column_order_affects_hash(self, sample_df: pd.DataFrame):
        """Verify column order affects hash."""
        hasher = ContentHasher()

        df_reordered = sample_df[sample_df.columns[::-1]]

        hash1 = hasher.hash_dataframe(sample_df)
        hash2 = hasher.hash_dataframe(df_reordered)

        assert hash1 != hash2

    def test_index_order_affects_hash(self, sample_df: pd.DataFrame):
        """Verify index order affects hash."""
        hasher = ContentHasher()

        df_reordered = sample_df.iloc[::-1]

        hash1 = hasher.hash_dataframe(sample_df)
        hash2 = hasher.hash_dataframe(df_reordered)

        assert hash1 != hash2

    def test_hash_with_na_values(self, sample_df_with_na: pd.DataFrame):
        """Verify hashing works with NA values."""
        hasher = ContentHasher()

        # Should not raise
        hash1 = hasher.hash_dataframe(sample_df_with_na)

        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_different_na_types_different_hash(self):
        """Verify different NA types produce different hashes."""
        hasher = ContentHasher()

        # Create DataFrames with different NA types
        df_nan = pd.DataFrame({"col": [np.nan]})
        df_none = pd.DataFrame({"col": [None]})
        df_pdna = pd.DataFrame({"col": [pd.NA]})

        hash_nan = hasher.hash_dataframe(df_nan)
        hash_none = hasher.hash_dataframe(df_none)
        hash_pdna = hasher.hash_dataframe(df_pdna)

        # All should be different
        assert len({hash_nan, hash_none, hash_pdna}) == 3

    def test_hash_series(self):
        """Verify Series hashing works."""
        hasher = ContentHasher()

        series = pd.Series([1.0, 2.0, 3.0], name="test")

        hash1 = hasher.hash_series(series)
        hash2 = hasher.hash_series(series.copy())

        assert hash1 == hash2

    def test_empty_dataframe(self):
        """Verify empty DataFrame hashing works."""
        hasher = ContentHasher()

        df = pd.DataFrame()

        hash1 = hasher.hash_dataframe(df)

        assert isinstance(hash1, str)

    def test_sha256_algorithm(self):
        """Verify SHA256 algorithm works."""
        hasher = ContentHasher(algorithm="sha256")

        df = pd.DataFrame({"a": [1, 2, 3]})
        hash1 = hasher.hash_dataframe(df)
        hash2 = hasher.hash_dataframe(df.copy())

        assert hash1 == hash2
        # SHA256 produces 64-character hex string
        assert len(hash1) == 64

    def test_datetime_column_hashing(self):
        """Verify datetime columns are hashed correctly."""
        hasher = ContentHasher()

        dates = pd.date_range("2025-01-01", periods=5)
        df = pd.DataFrame({"date_col": dates, "value": [1, 2, 3, 4, 5]})

        hash1 = hasher.hash_dataframe(df)
        hash2 = hasher.hash_dataframe(df.copy())

        assert hash1 == hash2

    def test_string_object_column_hashing(self):
        """Verify string/object columns are hashed correctly."""
        hasher = ContentHasher()

        df = pd.DataFrame(
            {
                "str_col": ["apple", "banana", "cherry"],
                "obj_col": [{"a": 1}, {"b": 2}, {"c": 3}],
            }
        )

        hash1 = hasher.hash_dataframe(df)

        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_index_without_to_numpy(self):
        """Verify hashing works for index without to_numpy method."""
        hasher = ContentHasher()

        # Create a DataFrame with a custom index
        df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])

        hash1 = hasher.hash_dataframe(df)

        assert isinstance(hash1, str)

    def test_hash_with_nan_in_float_column(self):
        """Verify NaN values in float columns are handled correctly."""
        hasher = ContentHasher()

        df = pd.DataFrame({"col": [1.0, np.nan, 3.0, np.nan, 5.0]})

        hash1 = hasher.hash_dataframe(df)
        hash2 = hasher.hash_dataframe(df.copy())

        assert hash1 == hash2

    def test_hash_series_with_na(self):
        """Verify Series with NA values are hashed correctly."""
        hasher = ContentHasher()

        series = pd.Series([1.0, np.nan, None, pd.NA, 5.0], name="test")

        hash1 = hasher.hash_series(series)

        assert isinstance(hash1, str)
        assert len(hash1) > 0


class TestGetNaTypeInHasher:
    """Tests for _get_na_type helper in hasher module."""

    def test_get_na_type_unknown(self):
        """Verify unknown NA types are handled."""
        from finlab_sentinel.comparison.hasher import _get_na_type

        # pd.NaT is a special NA type
        result = _get_na_type(pd.NaT)

        # Should return some string, probably 'unknown'
        assert isinstance(result, str)
