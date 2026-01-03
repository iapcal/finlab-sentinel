"""Main CLI entry point for finlab-sentinel."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="sentinel",
    help="finlab-sentinel: Defensive layer for finlab data.get API",
    add_completion=True,
)
console = Console()

# Global options
_config_path: Path | None = None
_verbose: bool = False


@app.callback()
def main(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """finlab-sentinel CLI - Monitor finlab data changes."""
    global _config_path, _verbose
    _config_path = config
    _verbose = verbose

    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)


@app.command("list")
def list_backups(
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Filter by dataset name",
    ),
    days: int | None = typer.Option(
        None,
        "--days",
        help="Show only last N days",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, csv",
    ),
) -> None:
    """List all backups."""
    from finlab_sentinel.config.loader import load_config
    from finlab_sentinel.storage.parquet import ParquetStorage, sanitize_backup_key

    config = load_config(_config_path)
    storage = ParquetStorage(
        base_path=config.get_storage_path(),
        compression=config.storage.compression,
    )

    # Get backup key if dataset specified
    backup_key = sanitize_backup_key(dataset) if dataset else None

    backups = storage.list_backups(backup_key)

    # Filter by days if specified
    if days:
        cutoff = datetime.now() - timedelta(days=days)
        backups = [b for b in backups if b.created_at >= cutoff]

    if not backups:
        console.print("[yellow]No backups found[/yellow]")
        return

    if format == "json":
        import json

        console.print(json.dumps([b.to_dict() for b in backups], indent=2))
        return

    if format == "csv":
        console.print("dataset,backup_key,created_at,rows,columns,size_bytes")
        for b in backups:
            console.print(
                f"{b.dataset},{b.backup_key},{b.created_at.isoformat()},"
                f"{b.row_count},{b.column_count},{b.file_size_bytes}"
            )
        return

    # Table format
    table = Table(title="Backups")
    table.add_column("Dataset", style="cyan")
    table.add_column("Created At", style="green")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    table.add_column("Size", justify="right")

    for b in backups:
        size_str = _format_size(b.file_size_bytes)
        table.add_row(
            b.dataset,
            b.created_at.strftime("%Y-%m-%d %H:%M"),
            str(b.row_count),
            str(b.column_count),
            size_str,
        )

    console.print(table)


@app.command("cleanup")
def cleanup(
    days: int | None = typer.Option(
        None,
        "--days",
        "-d",
        help="Retention days (overrides config)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without deleting",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Clean up expired backups."""
    from finlab_sentinel.config.loader import load_config
    from finlab_sentinel.storage.parquet import ParquetStorage

    config = load_config(_config_path)
    retention_days = days if days is not None else config.storage.retention_days

    storage = ParquetStorage(
        base_path=config.get_storage_path(),
        compression=config.storage.compression,
    )

    # Preview what would be deleted
    cutoff = datetime.now() - timedelta(days=retention_days)
    backups = storage.list_backups()
    to_delete = [b for b in backups if b.created_at < cutoff]

    if not to_delete:
        console.print(f"[green]No backups older than {retention_days} days[/green]")
        return

    # Show what will be deleted
    total_size = sum(b.file_size_bytes for b in to_delete)
    console.print(
        f"Found [bold]{len(to_delete)}[/bold] backups older than "
        f"{retention_days} days ({_format_size(total_size)})"
    )

    if dry_run:
        console.print("[yellow]Dry run - no files deleted[/yellow]")
        for b in to_delete:
            console.print(f"  Would delete: {b.dataset} ({b.created_at})")
        return

    # Confirm unless forced
    if not force:
        confirm = typer.confirm("Delete these backups?")
        if not confirm:
            console.print("[yellow]Aborted[/yellow]")
            return

    # Delete
    deleted = storage.cleanup_expired(retention_days)
    console.print(f"[green]Deleted {deleted} backups[/green]")


@app.command("export")
def export_backup(
    dataset: str = typer.Argument(..., help="Dataset name to export"),
    output: Path = typer.Option(
        Path("."),
        "--output",
        "-o",
        help="Output directory or file",
    ),
    format: str = typer.Option(
        "parquet",
        "--format",
        "-f",
        help="Export format: parquet, csv",
    ),
    date: str | None = typer.Option(
        None,
        "--date",
        help="Export specific date (YYYY-MM-DD)",
    ),
) -> None:
    """Export backup data."""
    from finlab_sentinel.config.loader import load_config
    from finlab_sentinel.storage.parquet import ParquetStorage, sanitize_backup_key

    config = load_config(_config_path)
    storage = ParquetStorage(
        base_path=config.get_storage_path(),
        compression=config.storage.compression,
    )

    backup_key = sanitize_backup_key(dataset)

    # Load backup
    if date:
        target_date = datetime.strptime(date, "%Y-%m-%d")
        result = storage.load_by_date(backup_key, target_date)
    else:
        result = storage.load_latest(backup_key)

    if result is None:
        console.print(f"[red]No backup found for: {dataset}[/red]")
        raise typer.Exit(1)

    df, metadata = result

    # Determine output path
    if output.is_dir():
        ext = "parquet" if format == "parquet" else "csv"
        output_file = output / f"{backup_key}.{ext}"
    else:
        output_file = output

    # Export
    if format == "csv":
        df.to_csv(output_file)
    else:
        df.to_parquet(output_file)

    console.print(f"[green]Exported to: {output_file}[/green]")


@app.command("accept")
def accept(
    dataset: str = typer.Argument(..., help="Dataset name to accept"),
    reason: str | None = typer.Option(
        None,
        "--reason",
        "-r",
        help="Reason for accepting the new data",
    ),
) -> None:
    """Accept current data as new baseline."""
    from finlab_sentinel.config.loader import load_config
    from finlab_sentinel.core.interceptor import accept_current_data

    config = load_config(_config_path)

    success = accept_current_data(dataset, config, reason)

    if success:
        console.print(f"[green]Accepted new data for: {dataset}[/green]")
        if reason:
            console.print(f"  Reason: {reason}")
    else:
        console.print(f"[red]Failed to accept data for: {dataset}[/red]")
        raise typer.Exit(1)


@app.command("diff")
def diff(
    dataset: str = typer.Argument(..., help="Dataset name to diff"),
    from_date: str | None = typer.Option(
        None,
        "--from",
        help="From date (YYYY-MM-DD)",
    ),
    to_date: str | None = typer.Option(
        None,
        "--to",
        help="To date (YYYY-MM-DD)",
    ),
) -> None:
    """Show differences for a dataset."""
    from finlab_sentinel.comparison.differ import DataFrameComparer
    from finlab_sentinel.config.loader import load_config
    from finlab_sentinel.storage.parquet import ParquetStorage, sanitize_backup_key

    config = load_config(_config_path)
    storage = ParquetStorage(
        base_path=config.get_storage_path(),
        compression=config.storage.compression,
    )

    backup_key = sanitize_backup_key(dataset)

    # Load backups
    if from_date and to_date:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")

        old_result = storage.load_by_date(backup_key, from_dt)
        new_result = storage.load_by_date(backup_key, to_dt)
    else:
        # Compare latest with current data from finlab
        old_result = storage.load_latest(backup_key)
        new_result = None  # Will fetch from finlab

    if old_result is None:
        console.print(f"[red]No backup found for: {dataset}[/red]")
        raise typer.Exit(1)

    old_df, old_metadata = old_result

    if new_result:
        new_df, _ = new_result
    else:
        # Fetch current from finlab
        try:
            from finlab import data

            new_df = data.get(dataset)
        except ImportError as e:
            console.print("[red]finlab not installed[/red]")
            raise typer.Exit(1) from e

    # Compare
    comparer = DataFrameComparer(
        rtol=config.comparison.rtol,
        atol=config.comparison.atol,
        check_dtype=config.comparison.check_dtype,
        check_na_type=config.comparison.check_na_type,
    )

    result = comparer.compare(old_df, new_df)

    # Display results
    if result.is_identical:
        console.print("[green]No differences found[/green]")
        return

    console.print(f"[bold]Changes: {result.summary()}[/bold]\n")

    if result.added_rows:
        console.print(f"[green]+ {len(result.added_rows)} rows added[/green]")
        for row in list(result.added_rows)[:5]:
            console.print(f"    {row}")
        if len(result.added_rows) > 5:
            console.print(f"    ... and {len(result.added_rows) - 5} more")

    if result.deleted_rows:
        console.print(f"[red]- {len(result.deleted_rows)} rows deleted[/red]")
        for row in list(result.deleted_rows)[:5]:
            console.print(f"    {row}")

    if result.added_columns:
        console.print(f"[green]+ {len(result.added_columns)} columns added[/green]")

    if result.deleted_columns:
        console.print(f"[red]- {len(result.deleted_columns)} columns deleted[/red]")

    if result.modified_cells:
        console.print(f"[yellow]~ {len(result.modified_cells)} cells modified[/yellow]")
        for change in result.modified_cells[:5]:
            console.print(f"    {change}")
        if len(result.modified_cells) > 5:
            console.print(f"    ... and {len(result.modified_cells) - 5} more")

    if result.dtype_changes:
        console.print(f"[yellow]~ {len(result.dtype_changes)} dtype changes[/yellow]")
        for dtype_change in result.dtype_changes:
            console.print(f"    {dtype_change}")


@app.command("config")
def config_cmd(
    action: str = typer.Argument(
        "show",
        help="Action: show, validate, init, paths",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for init",
    ),
) -> None:
    """Configuration management."""
    if action == "show":
        from finlab_sentinel.config.loader import load_config

        try:
            config = load_config(_config_path)
            console.print(config.model_dump_json(indent=2))
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            raise typer.Exit(1) from e

    elif action == "validate":
        from finlab_sentinel.config.loader import load_config

        try:
            load_config(_config_path)
            console.print("[green]Configuration is valid[/green]")
        except Exception as e:
            console.print(f"[red]Invalid configuration: {e}[/red]")
            raise typer.Exit(1) from e

    elif action == "init":
        from finlab_sentinel.config.loader import create_default_config_file

        path = output if output else Path("sentinel.toml")
        if path.exists():
            console.print(f"[yellow]File already exists: {path}[/yellow]")
            if not typer.confirm("Overwrite?"):
                raise typer.Exit(0)

        create_default_config_file(path)
        console.print(f"[green]Created config file: {path}[/green]")

    elif action == "paths":
        from finlab_sentinel.config.loader import CONFIG_SEARCH_PATHS

        console.print("Config file search paths:")
        for path in CONFIG_SEARCH_PATHS:
            expanded = path.expanduser()
            if expanded.exists():
                exists = "[green]exists[/green]"
            else:
                exists = "[dim]not found[/dim]"
            console.print(f"  {path} -> {exists}")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@app.command("info")
def info() -> None:
    """Show storage information and statistics."""
    from finlab_sentinel.config.loader import load_config
    from finlab_sentinel.storage.cleanup import get_storage_info

    config = load_config(_config_path)
    stats = get_storage_info(config)

    console.print("[bold]finlab-sentinel Storage Info[/bold]\n")
    console.print(f"Storage Path: {stats['storage_path']}")
    console.print(f"Total Backups: {stats['total_backups']}")
    console.print(f"Unique Datasets: {stats['unique_datasets']}")
    console.print(f"Total Size: {_format_size(stats['total_size_bytes'])}")

    if stats["oldest"]:
        console.print(f"Oldest Backup: {stats['oldest']}")
    if stats["newest"]:
        console.print(f"Newest Backup: {stats['newest']}")


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


if __name__ == "__main__":
    app()
