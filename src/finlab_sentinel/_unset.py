"""Marker for omitted optional arguments where None is a meaningful value."""

from __future__ import annotations

from enum import Enum
from typing import Final


class _Unset(Enum):
    """Type of UNSET."""

    UNSET = "UNSET"


# Marker for an omitted ``expected_latest`` argument (no compare-and-swap);
# distinct from None, which means "the key has no backups". Kept in its own
# module so the package can use it without importing the storage layer.
UNSET: Final = _Unset.UNSET
