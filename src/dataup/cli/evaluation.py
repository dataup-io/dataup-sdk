"""Evaluation commands for DataUp CLI."""

from __future__ import annotations

import sys

import click
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from dataup.cli.utils import (
    console,
    display_evaluation_results,
    get_async_cvat_client,
    get_async_dataup_client,
    get_cvat_client,
    get_dataup_client,
    get_provider,
)


@click.group()
def evaluation():
    """Evaluation commands for evaluating models against CVAT datasets."""
    pass


# Alias 'eval' for 'evaluation'
@click.group(name="eval")
def eval_alias():
    """Evaluation commands (alias for 'evaluation')."""
    pass


@evaluation.command("run")
@click.option(
    "--provider",
    type=click.Choice(["ultralytics", "roboflow"]),
    required=True,
    help="Inference provider to use.",
)
@click.option(
    "--task",
    type=int,
    required=True,
    help="CVAT task ID to evaluate against.",
)
@click.option(
    "--weights",
    required=True,
    help="Path to model weights or model identifier.",
)
@click.option(
    "--agent-id",
    required=False,
    default=None,
    help="Optional DataUp agent ID for this evaluation.",
)
@click.option(
    "--agent-name",
    required=False,
    default=None,
    help="Name of the model being evaluated.",
)
@click.option(
    "--agent-version",
    default="1.0.0",
    help="Agent version string.",
)
@click.option(
    "--conf",
    type=float,
    default=0.25,
    help="Confidence threshold for detections.",
)
@click.option(
    "--iou",
    type=float,
    default=0.5,
    help="IoU threshold for NMS.",
)
@click.option(
    "--batch-size",
    type=int,
    default=10,
    help="Number of frames per batch.",
)
def evaluation_run(
    provider: str,
    task: int,
    weights: str,
    agent_id: str | None,
    agent_name: str | None,
    agent_version: str,
    conf: float,
    iou: float,
    batch_size: int,
):
    """Run an evaluation by running inference on a CVAT task.

    This command:
    1. Creates an evaluation on the DataUp API
    2. Runs inference on each frame using the specified provider
    3. Submits batches of predictions to the API
    4. Finalizes the evaluation to compute COCO metrics

    Example:

        dataup evaluation run --provider ultralytics --task 121 --weights yolov8n.pt
    """
    from dataup.evaluation import EvaluationRunner

    # Get clients
    dataup_client = get_dataup_client()
    cvat_client = get_cvat_client()

    # Get provider and load model
    inference_provider = get_provider(provider)

    console.print(f"Loading model: [cyan]{weights}[/cyan]")
    try:
        inference_provider.load_model(weights)
    except ImportError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        sys.exit(1)

    console.print(f"Model loaded. Classes: {len(inference_provider.class_names)}")

    # Create runner
    runner = EvaluationRunner(dataup_client, cvat_client, inference_provider)

    # Run evaluation with progress bar
    console.print(f"\nRunning evaluation on CVAT task [cyan]{task}[/cyan]...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_progress = progress.add_task("Processing frames...", total=None)

        def update_progress(current: int, total: int):
            progress.update(task_progress, completed=current, total=total)

        try:
            result = runner.run_and_submit(
                task_id=task,
                agent_id=agent_id,
                agent_name=agent_name or weights,
                agent_version=agent_version,
                conf=conf,
                iou=iou,
                batch_size=batch_size,
                progress_callback=update_progress,
            )
        except Exception as e:
            console.print(f"\n[red]Error running evaluation:[/red] {e}")
            sys.exit(1)

    display_evaluation_results(result)

    # Cleanup
    dataup_client.close()
    cvat_client.close()


eval_alias.add_command(evaluation_run, name="run")


@evaluation.command("get")
@click.option(
    "--evaluation-id",
    required=True,
    help="Evaluation ID to retrieve.",
)
def evaluation_get(evaluation_id: str):
    """Get evaluation results from DataUp.

    Example:

        dataup evaluation get --evaluation-id eval_abc123
    """
    dataup_client = get_dataup_client()

    try:
        eval_result = dataup_client.evaluations.get(evaluation_id)
        display_evaluation_results(eval_result)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        dataup_client.close()


eval_alias.add_command(evaluation_get, name="get")


@evaluation.command("list")
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of evaluations to list.",
)
def evaluation_list(limit: int):
    """List evaluations from DataUp.

    Example:

        dataup evaluation list --limit 20
    """
    dataup_client = get_dataup_client()

    try:
        result = dataup_client.evaluations.list(size=limit)

        table = Table(title="Evaluations")
        table.add_column("ID", style="cyan")
        table.add_column("Agent Name")
        table.add_column("Version")
        table.add_column("Dataset ID")
        table.add_column("Status")
        table.add_column("Frames")
        table.add_column("AP", justify="right")
        table.add_column("Created")

        for ev in result.items:
            ap_str = "-"
            if ev.summary_metrics:
                ap_str = f"{ev.summary_metrics.ap:.4f}"
            table.add_row(
                ev.id,
                ev.agent_name or "-",
                ev.agent_version or "-",
                str(ev.dataset_id),
                ev.status,
                str(ev.received_frames),
                ap_str,
                ev.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
        console.print(f"\nShowing {len(result.items)} evaluations")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        dataup_client.close()


eval_alias.add_command(evaluation_list, name="list")


@evaluation.command("delete")
@click.option(
    "--evaluation-id",
    required=True,
    help="Evaluation ID to delete.",
)
@click.confirmation_option(prompt="Are you sure you want to delete this evaluation?")
def evaluation_delete(evaluation_id: str):
    """Delete an evaluation from DataUp.

    Example:

        dataup evaluation delete --evaluation-id eval_abc123
    """
    dataup_client = get_dataup_client()

    try:
        dataup_client.evaluations.delete(evaluation_id)
        console.print(f"[green]Evaluation {evaluation_id} deleted successfully.[/green]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        dataup_client.close()


eval_alias.add_command(evaluation_delete, name="delete")


@evaluation.command("from-checkpoint")
@click.option(
    "--provider",
    type=click.Choice(["ultralytics", "roboflow"]),
    required=True,
    help="Inference provider to use.",
)
@click.option(
    "--task",
    type=int,
    required=True,
    help="CVAT task ID to evaluate against.",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to model checkpoint/weights file.",
)
@click.option(
    "--agent-id",
    required=False,
    default=None,
    help="Optional DataUp agent ID for this evaluation.",
)
@click.option(
    "--agent-name",
    required=False,
    default=None,
    help="Name of the model being evaluated.",
)
@click.option(
    "--agent-version",
    default="1.0.0",
    help="Agent version string.",
)
@click.option(
    "--conf",
    type=float,
    default=0.25,
    help="Confidence threshold for detections.",
)
@click.option(
    "--iou",
    type=float,
    default=0.5,
    help="IoU threshold for NMS.",
)
@click.option(
    "--batch-size",
    type=int,
    default=10,
    help="Number of frames per batch.",
)
def evaluation_from_checkpoint(
    provider: str,
    task: int,
    checkpoint: str,
    agent_id: str | None,
    agent_name: str | None,
    agent_version: str,
    conf: float,
    iou: float,
    batch_size: int,
):
    """Run evaluation using a model checkpoint, iterating over CVAT jobs.

    This command uses async I/O for efficient image fetching and batch
    submission while keeping inference synchronous. It:
    1. Creates an evaluation and gets its ID
    2. Loads the model from the checkpoint
    3. Iterates over each job in the CVAT task
    4. For each frame: fetches the image (async), runs inference (sync), gets ground truth
    5. Submits batches to DataUp via batch_ingest (async)
    6. Finalizes the evaluation

    Example:

        dataup evaluation from-checkpoint --provider ultralytics --task 121 --checkpoint best.pt
    """
    import asyncio

    # Get provider and load model first (sync)
    inference_provider = get_provider(provider)

    console.print(f"Loading model checkpoint: [cyan]{checkpoint}[/cyan]")
    try:
        inference_provider.load_model(checkpoint)
    except ImportError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        sys.exit(1)

    console.print(f"Model loaded. Classes: {len(inference_provider.class_names)}")

    # Run the async evaluation
    try:
        asyncio.run(
            _run_evaluation_from_checkpoint(
                inference_provider=inference_provider,
                task_id=task,
                checkpoint=checkpoint,
                agent_id=agent_id,
                agent_name=agent_name,
                agent_version=agent_version,
                conf=conf,
                iou=iou,
                batch_size=batch_size,
            )
        )
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


async def _run_evaluation_from_checkpoint(
    inference_provider,
    task_id: int,
    checkpoint: str,
    agent_id: str | None,
    agent_name: str | None,
    agent_version: str,
    conf: float,
    iou: float,
    batch_size: int,
):
    """Async implementation of evaluation from checkpoint."""
    import io

    from PIL import Image

    from dataup_models.labels import Label
    from dataup.models.evaluations import (
        BatchIngestRequest,
        EvaluationCreate,
        FrameData,
        PredictionSource,
    )

    # Get async clients
    dataup_client = get_async_dataup_client()
    cvat_client = get_async_cvat_client()

    try:
        # Fetch task details to get jobs
        console.print(f"\nFetching CVAT task [cyan]{task_id}[/cyan]...")
        cvat_task = await cvat_client.tasks.get(task_id)
        jobs_to_process = cvat_task.jobs

        # Calculate total frames and get job frame ranges
        total_frames = 0
        job_frame_ranges: dict[int, tuple[int, int]] = {}
        for job_summary in jobs_to_process:
            job = await cvat_client.jobs.get(job_summary.id)
            frame_count = job.stop_frame - job.start_frame + 1
            total_frames += frame_count
            job_frame_ranges[job_summary.id] = (job.start_frame, job.stop_frame)

        console.print(f"Found {len(jobs_to_process)} jobs with {total_frames} total frames")

        # Create the evaluation
        console.print("\nCreating evaluation...")
        create_response = await dataup_client.evaluations.create(
            EvaluationCreate(
                agent_id=agent_id,
                dataset_id=task_id,
                agent_name=agent_name or checkpoint,
                agent_version=agent_version,
                prediction_source=PredictionSource.CLIENT_SDK,
                total_frames=total_frames,
            )
        )
        evaluation_id = create_response.evaluation_id
        console.print(f"Created evaluation: [cyan]{evaluation_id}[/cyan]")

        # Process frames and submit in batches
        processed_frames = 0
        current_batch: list[FrameData] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_progress = progress.add_task("Processing frames...", total=total_frames)

            for job_summary in jobs_to_process:
                job_id = job_summary.id
                start_frame, stop_frame = job_frame_ranges[job_id]

                # Get frame metadata and annotations for this job (async)
                meta = await cvat_client.jobs.get_data_meta(job_id)
                annotations_list = await cvat_client.jobs.get_annotations(job_id)

                # Build frame_id -> ground truth lookup
                frame_annotations: dict[int, list[Label]] = {}
                for frame_labels in annotations_list:
                    frame_annotations[frame_labels.frame_id] = frame_labels.labels

                for frame_id in range(start_frame, stop_frame + 1):
                    # Skip deleted frames
                    if frame_id in meta.deleted_frames:
                        continue

                    # Fetch frame image from CVAT (async)
                    frame_image = await cvat_client.jobs.get_frame(job_id, frame_id)

                    # Convert to PIL Image for inference
                    pil_image = Image.open(io.BytesIO(frame_image.data))

                    # Run inference on the image (sync - blocks event loop but that's OK)
                    predictions = inference_provider.predict(pil_image, conf=conf, iou=iou)

                    # Get ground truth from CVAT annotations
                    ground_truth = frame_annotations.get(frame_id, [])

                    # Create frame data
                    frame_data = FrameData(
                        job_id=job_id,
                        frame_id=frame_id,
                        ground_truth=ground_truth,
                        predictions=predictions,
                        image_width=pil_image.width,
                        image_height=pil_image.height,
                    )
                    current_batch.append(frame_data)

                    processed_frames += 1
                    progress.update(task_progress, completed=processed_frames)

                    # Submit batch when full (async)
                    if len(current_batch) >= batch_size:
                        await dataup_client.evaluations.ingest_batch(
                            evaluation_id,
                            BatchIngestRequest(frames=current_batch),
                        )
                        current_batch = []

            # Submit any remaining frames (async)
            if current_batch:
                await dataup_client.evaluations.ingest_batch(
                    evaluation_id,
                    BatchIngestRequest(frames=current_batch),
                )

        # Finalize the evaluation (async)
        console.print("\nFinalizing evaluation...")
        await dataup_client.evaluations.finalize(evaluation_id)

        # Get and display results (async)
        result = await dataup_client.evaluations.get(evaluation_id)
        display_evaluation_results(result)

    finally:
        await dataup_client.aclose()
        await cvat_client.aclose()


eval_alias.add_command(evaluation_from_checkpoint, name="from-checkpoint")
