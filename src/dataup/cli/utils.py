"""Shared utilities for DataUp CLI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from dataup.cli.config import get_setting

if TYPE_CHECKING:
    from dataup.models.evaluations import EvaluationRead

console = Console()


def get_dataup_client():
    """Get DataUp client from config or environment variables."""
    from dataup import DataUpClient

    api_key = get_setting("dataup_api_key", "DATAUP_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] DataUp API key not configured.")
        console.print("Run: dataup configure")
        console.print("Or set: export DATAUP_API_KEY='your_key_id.your_key_secret'")
        sys.exit(1)

    return DataUpClient(api_key=api_key)


def get_async_dataup_client():
    """Get async DataUp client from config or environment variables."""
    from dataup import AsyncDataUpClient

    api_key = get_setting("dataup_api_key", "DATAUP_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] DataUp API key not configured.")
        console.print("Run: dataup configure")
        console.print("Or set: export DATAUP_API_KEY='your_key_id.your_key_secret'")
        sys.exit(1)

    return AsyncDataUpClient(api_key=api_key)


def get_cvat_client():
    """Get CVAT client from config or environment variables."""
    from dataup.cvat import CVATClient

    api_token = get_setting("cvat_api_token", "CVAT_API_TOKEN")
    if not api_token:
        console.print("[red]Error:[/red] CVAT API token not configured.")
        console.print("Run: dataup configure")
        console.print("Or set: export CVAT_API_TOKEN='your_cvat_token'")
        sys.exit(1)

    base_url = get_setting("cvat_base_url", "CVAT_BASE_URL") or "https://app.cvat.ai"
    return CVATClient(api_token=api_token, base_url=base_url)


def get_async_cvat_client():
    """Get async CVAT client from config or environment variables."""
    from dataup.cvat import AsyncCVATClient

    api_token = get_setting("cvat_api_token", "CVAT_API_TOKEN")
    if not api_token:
        console.print("[red]Error:[/red] CVAT API token not configured.")
        console.print("Run: dataup configure")
        console.print("Or set: export CVAT_API_TOKEN='your_cvat_token'")
        sys.exit(1)

    base_url = get_setting("cvat_base_url", "CVAT_BASE_URL") or "https://app.cvat.ai"
    return AsyncCVATClient(api_token=api_token, base_url=base_url)


def get_provider(provider_name: str):
    """Get inference provider by name."""
    if provider_name == "ultralytics":
        from dataup.evaluation.providers import UltralyticsProvider

        return UltralyticsProvider()
    elif provider_name == "roboflow":
        from dataup.evaluation.providers import RoboflowProvider

        return RoboflowProvider()
    else:
        console.print(f"[red]Error:[/red] Unknown provider: {provider_name}")
        sys.exit(1)


def display_evaluation_results(evaluation: EvaluationRead) -> None:
    """Display evaluation results in a formatted table."""
    console.print()
    console.print("[bold green]Evaluation Complete[/bold green]")
    console.print(f"Evaluation ID: [cyan]{evaluation.id}[/cyan]")
    console.print(f"Status: [cyan]{evaluation.status}[/cyan]")
    console.print()

    # Summary metrics table
    if evaluation.summary_metrics:
        sm = evaluation.summary_metrics
        summary_table = Table(title="COCO Summary Metrics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("AP @ IoU=0.50:0.95", f"{sm.ap:.4f}")
        summary_table.add_row("AP @ IoU=0.50", f"{sm.ap_50:.4f}")
        summary_table.add_row("AP @ IoU=0.75", f"{sm.ap_75:.4f}")
        if sm.ap_small is not None:
            summary_table.add_row("AP (small)", f"{sm.ap_small:.4f}")
        if sm.ap_medium is not None:
            summary_table.add_row("AP (medium)", f"{sm.ap_medium:.4f}")
        if sm.ap_large is not None:
            summary_table.add_row("AP (large)", f"{sm.ap_large:.4f}")
        summary_table.add_row("AR @ max=1", f"{sm.ar_1:.4f}")
        summary_table.add_row("AR @ max=10", f"{sm.ar_10:.4f}")
        summary_table.add_row("AR @ max=100", f"{sm.ar_100:.4f}")

        console.print(summary_table)
        console.print()

    # Per-class metrics table
    if evaluation.per_class_summary_metrics:
        class_table = Table(title="Per-Class Metrics")
        class_table.add_column("Class", style="cyan")
        class_table.add_column("AP", justify="right")
        class_table.add_column("AP@50", justify="right")
        class_table.add_column("AP@75", justify="right")
        class_table.add_column("AR@100", justify="right")
        class_table.add_column("GT Count", justify="right")
        class_table.add_column("Det Count", justify="right")

        for cm in evaluation.per_class_summary_metrics:
            class_table.add_row(
                cm.class_name,
                f"{cm.ap:.4f}",
                f"{cm.ap_50:.4f}",
                f"{cm.ap_75:.4f}",
                f"{cm.ar_100:.4f}",
                str(cm.ground_truth_count),
                str(cm.detection_count),
            )

        console.print(class_table)
        console.print()

    # Summary info
    console.print(f"Total Frames: {evaluation.received_frames}")
    if evaluation.progress is not None:
        console.print(f"Progress: {evaluation.progress * 100:.1f}%")
