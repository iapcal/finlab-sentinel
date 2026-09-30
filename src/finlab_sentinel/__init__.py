"""finlab-sentinel: Defensive monitoring layer for finlab data.get API."""

from datetime import datetime
from typing import TYPE_CHECKING

from finlab_sentinel.core.hooks import (
    clear_preprocess_hooks,
    register_preprocess_hook,
    unregister_preprocess_hook,
)
from finlab_sentinel.core.patcher import disable, enable, is_enabled
from finlab_sentinel.core.time_travel import TimeTravelContext
from finlab_sentinel.exceptions import (
    DataAnomalyError,
    NoHistoricalDataError,
    PatchNotFoundError,
    PatchRestoreError,
    SentinelError,
    TimeTravelError,
)

if TYPE_CHECKING:
    import pandas as pd

    from finlab_sentinel.config.schema import SentinelConfig
    from finlab_sentinel.storage.patches import (
        PatchMetadata,
        PatchStore,
        RestoreResult,
    )

__version__ = "0.1.9"

__all__ = [
    "__version__",
    "enable",
    "disable",
    "is_enabled",
    "SentinelError",
    "DataAnomalyError",
    "TimeTravelError",
    "NoHistoricalDataError",
    "PatchNotFoundError",
    "PatchRestoreError",
    "register_preprocess_hook",
    "unregister_preprocess_hook",
    "clear_preprocess_hooks",
    "set_time_travel",
    "exit_time_travel",
    "get_time_travel_status",
    "list_patches",
    "load_patch_data",
    "restore_patch",
]


def set_time_travel(target_time: datetime) -> None:
    """Set time travel mode to specified datetime.

    After calling this, all data.get() calls will return historical backup data
    from the specified time point instead of fetching fresh data.

    Args:
        target_time: Target datetime to travel to

    Example:
        >>> import finlab_sentinel as fs
        >>> from datetime import datetime
        >>> fs.enable()
        >>> fs.set_time_travel(datetime(2024, 1, 5, 14, 30))
        >>> data.get("price:收盤價")  # Returns historical data
        >>> fs.exit_time_travel()
    """
    ctx = TimeTravelContext.get_instance()
    ctx.set_target_time(target_time)


def exit_time_travel() -> None:
    """Exit time travel mode and return to normal data retrieval."""
    ctx = TimeTravelContext.get_instance()
    ctx.clear()


def get_time_travel_status() -> dict:
    """Get current time travel status.

    Returns:
        Dictionary with 'enabled' (bool) and 'target_time' (str | None) keys

    Example:
        >>> fs.get_time_travel_status()
        {'enabled': True, 'target_time': '2024-01-05T14:30:00'}
    """
    ctx = TimeTravelContext.get_instance()
    return {
        "enabled": ctx.is_active(),
        "target_time": ctx.target_time.isoformat() if ctx.target_time else None,
    }


def _get_patch_store() -> "PatchStore":
    """Create PatchStore from the default configuration."""
    from finlab_sentinel.config.loader import load_config
    from finlab_sentinel.storage.patches import PatchStore

    config = load_config()
    return PatchStore(
        base_path=config.get_storage_path(),
        compression=config.storage.compression,
    )


def list_patches(dataset: str | None = None) -> "list[PatchMetadata]":
    """List permanent patches created when accepting data.

    Args:
        dataset: Optional dataset name filter

    Returns:
        Patch metadata sorted by creation time (newest first)

    Example:
        >>> import finlab_sentinel as fs
        >>> for p in fs.list_patches("price:收盤價"):
        ...     print(p.patch_id, p.diff_summary["summary_text"])
    """
    return _get_patch_store().list_patches(dataset)


def load_patch_data(patch_id: str) -> "pd.DataFrame":
    """Load the old baseline DataFrame preserved by a patch.

    Args:
        patch_id: The patch identifier (see list_patches)

    Returns:
        The baseline DataFrame as it was before the accept

    Raises:
        PatchNotFoundError: If the patch does not exist

    Example:
        >>> import finlab_sentinel as fs
        >>> old_close = fs.load_patch_data(fs.list_patches()[0].patch_id)
    """
    return _get_patch_store().load_old_data(patch_id)


def restore_patch(
    patch_id: str,
    config: "SentinelConfig | None" = None,
    reason: str | None = None,
    dry_run: bool = False,
) -> "RestoreResult":
    """Restore the baseline preserved by a patch, undoing its accept.

    The patch's data and content hash become the dataset's baseline again,
    exactly as before the accept that created the patch. The replaced
    baseline is saved as a new patch (see ``RestoreResult.new_patch_id``), so
    the restore can itself be restored; the source patch is kept.

    Args:
        patch_id: The patch identifier (see list_patches)
        config: Optional configuration (uses default if not provided)
        reason: Reason recorded on the new patch (default: "restore of <id>")
        dry_run: Only report what would change, without writing anything

    Returns:
        RestoreResult describing the previous and restored baselines

    Raises:
        PatchNotFoundError: If the patch does not exist
        PatchRestoreError: If the restore fails; the baseline is unchanged

    Example:
        >>> import finlab_sentinel as fs
        >>> result = fs.restore_patch("price__收盤價__2026-07-11T10-30-00")
        >>> result.new_patch_id  # restore this to undo the restore
    """
    from finlab_sentinel.storage.patches import restore_patch as _restore_patch

    return _restore_patch(patch_id, config=config, reason=reason, dry_run=dry_run)
