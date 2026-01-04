"""Pydantic configuration schema for finlab-sentinel."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PolicyMode(str, Enum):
    """Comparison policy mode."""

    APPEND_ONLY = "append_only"
    THRESHOLD = "threshold"
    PERMISSIVE = "permissive"


class AnomalyBehavior(str, Enum):
    """Behavior when anomaly is detected."""

    RAISE = "raise"
    WARN_RETURN_CACHED = "warn_return_cached"
    WARN_RETURN_NEW = "warn_return_new"


class StorageConfig(BaseModel):
    """Storage configuration."""

    path: Path = Field(default=Path("~/.finlab-sentinel/"))
    retention_days: int = Field(default=7, ge=1, le=365)
    min_backups_per_dataset: int = Field(default=3, ge=1, le=100)
    format: Literal["parquet"] = "parquet"
    compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"

    @field_validator("path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand ~ and environment variables in path."""
        if isinstance(v, str):
            v = Path(v)
        return v.expanduser()


class ComparisonPoliciesConfig(BaseModel):
    """Comparison policies configuration."""

    default_mode: PolicyMode = PolicyMode.APPEND_ONLY
    history_modifiable: list[str] = Field(default_factory=list)
    allow_na_to_value: list[str] = Field(default_factory=list)


class ComparisonConfig(BaseModel):
    """Comparison configuration."""

    rtol: float = Field(default=1e-5, ge=0)
    atol: float = Field(default=1e-8, ge=0)
    check_dtype: bool = True
    check_na_type: bool = True
    change_threshold: float = Field(default=0.10, ge=0, le=1)
    policies: ComparisonPoliciesConfig = Field(default_factory=ComparisonPoliciesConfig)


class AnomalyConfig(BaseModel):
    """Anomaly handling configuration."""

    behavior: AnomalyBehavior = AnomalyBehavior.RAISE
    save_reports: bool = True
    reports_dir: str = "reports/"
    callback: str | None = None  # "module.path:function_name"

    # Runtime callback (not serializable, set programmatically)
    _callback_fn: Callable | None = None

    model_config = {"arbitrary_types_allowed": True}

    def set_callback(self, fn: Callable) -> None:
        """Set callback function programmatically."""
        self._callback_fn = fn

    def get_callback(self) -> Callable | None:
        """Get callback function."""
        return self._callback_fn


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path | None = None

    @field_validator("file", mode="before")
    @classmethod
    def expand_file_path(cls, v: str | Path | None) -> Path | None:
        """Expand ~ in log file path."""
        if v is None:
            return None
        if isinstance(v, str):
            v = Path(v)
        return v.expanduser()


class CLIConfig(BaseModel):
    """CLI configuration."""

    output_format: Literal["table", "json", "csv"] = "table"


class SentinelConfig(BaseModel):
    """Main sentinel configuration."""

    storage: StorageConfig = Field(default_factory=StorageConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)

    def get_storage_path(self) -> Path:
        """Get resolved storage path."""
        return self.storage.path

    def get_reports_path(self) -> Path:
        """Get resolved reports path."""
        return self.storage.path / self.anomaly.reports_dir

    def is_dataset_history_modifiable(self, dataset: str) -> bool:
        """Check if dataset is in the history_modifiable blacklist."""
        return dataset in self.comparison.policies.history_modifiable
