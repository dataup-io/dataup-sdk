"""Evaluation runner for orchestrating evaluation creation and submission."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Callable

from PIL import Image

from dataup.models.evaluations import (
    BatchIngestRequest,
    EvaluationCreate,
    EvaluationRead,
    FrameData,
    PredictionSource,
)

if TYPE_CHECKING:
    from dataup.client import DataUpClient
    from dataup.cvat import CVATClient
    from dataup.evaluation.providers.base import InferenceProvider


class EvaluationRunner:
    """Orchestrates evaluation creation and submission.

    The EvaluationRunner coordinates between:
    - CVAT client (fetches images and ground truth annotations)
    - Inference provider (runs model predictions)
    - DataUp client (submits evaluation results via streaming API)

    The new evaluation system uses a streaming approach:
    1. Create an evaluation
    2. Ingest batches of frames with predictions
    3. Finalize the evaluation to compute metrics

    Example:
        >>> from dataup import DataUpClient
        >>> from dataup.cvat import CVATClient
        >>> from dataup.evaluation import EvaluationRunner, UltralyticsProvider
        >>>
        >>> dataup = DataUpClient(api_key="...")
        >>> cvat = CVATClient(api_token="...")
        >>> provider = UltralyticsProvider()
        >>> provider.load_model("yolov8n.pt")
        >>>
        >>> runner = EvaluationRunner(dataup, cvat, provider)
        >>> result = runner.run_and_submit(
        ...     task_id=123,
        ...     agent_name="my-model",
        ...     agent_version="1.0.0"
        ... )
    """

    def __init__(
        self, dataup_client: DataUpClient, cvat_client: CVATClient, provider: InferenceProvider
    ) -> None:
        """Initialize the evaluation runner.

        Args:
            dataup_client: DataUp API client for submitting evaluations.
            cvat_client: CVAT client for fetching images and annotations.
            provider: Inference provider for running predictions.
        """
        self._dataup = dataup_client
        self._cvat = cvat_client
        self._provider = provider

    def run_and_submit(
        self,
        task_id: int,
        *,
        agent_id: str | None = None,
        agent_name: str | None = None,
        agent_version: str | None = None,
        job_ids: list[int] | None = None,
        conf: float = 0.0,
        iou: float = 0.5,
        batch_size: int = 10,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EvaluationRead:
        """Run evaluation on a CVAT task and submit to DataUp.

        This method:
        1. Creates an evaluation on the DataUp API
        2. Uses iter_jobs_with_annotations to efficiently iterate over jobs
        3. For each job:
           a. Iterates through frames using iter_frames
           b. Runs inference on each frame image
           c. Submits batches to the API
        4. Finalizes the evaluation to compute COCO metrics
        5. Returns the complete evaluation with metrics

        Args:
            task_id: CVAT task ID to evaluate against.
            agent_id: Optional DataUp agent ID being evaluated.
            agent_name: Name of the model/agent being evaluated.
            agent_version: Version string for the agent (e.g., "1.0.0").
            job_ids: Optional list of specific job IDs to process.
                    If None, processes all jobs in the task.
            conf: Confidence threshold for inference.
            iou: IoU threshold for NMS during inference.
            batch_size: Number of frames to process before sending to API.
            progress_callback: Optional callback function called with
                             (current_frame, total_frames) for progress updates.

        Returns:
            EvaluationRead object with computed COCO metrics.
        """
        # First pass: calculate total frames for progress tracking and filter jobs
        total_frames = 0
        jobs_to_process_ids: set[int] | None = set(job_ids) if job_ids is not None else None

        for job_summary, _ in self._cvat.jobs.iter_jobs_with_annotations(task_id):
            if jobs_to_process_ids is not None and job_summary.id not in jobs_to_process_ids:
                continue
            job = self._cvat.jobs.get(job_summary.id)
            meta = self._cvat.jobs.get_data_meta(job_summary.id)
            frame_count = job.stop_frame - job.start_frame + 1 - len(meta.deleted_frames)
            total_frames += frame_count

        # Create the evaluation
        create_response = self._dataup.evaluations.create(
            EvaluationCreate(
                agent_id=agent_id,
                dataset_id=task_id,
                agent_name=agent_name,
                agent_version=agent_version,
                prediction_source=PredictionSource.CLIENT_SDK,
                total_frames=total_frames,
            )
        )
        evaluation_id = create_response.evaluation_id

        # Process frames and submit in batches
        processed_frames = 0
        current_batch: list[FrameData] = []

        try:
            # Second pass: process frames using iterators
            for job_summary, frame_labels_list in self._cvat.jobs.iter_jobs_with_annotations(
                task_id
            ):
                job_id = job_summary.id

                # Skip jobs not in the filter list
                if jobs_to_process_ids is not None and job_id not in jobs_to_process_ids:
                    continue

                # Build frame_id -> labels lookup from FrameLabels
                frame_labels_lookup: dict[int, list] = {}
                for frame_labels in frame_labels_list:
                    frame_labels_lookup[frame_labels.frame_id] = frame_labels.labels

                # Iterate through frames using iter_frames
                for frame_image in self._cvat.jobs.iter_frames(job_id):
                    frame_id = frame_image.frame_id

                    # Convert bytes to PIL Image
                    pil_image = Image.open(io.BytesIO(frame_image.data))

                    # Run inference
                    predictions = self._provider.predict(pil_image, conf=conf, iou=iou)

                    # Get ground truth labels (already converted from CVAT annotations)
                    ground_truth = frame_labels_lookup.get(frame_id, [])
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
                    if progress_callback:
                        progress_callback(processed_frames, total_frames)

                    # Submit batch when full
                    if len(current_batch) >= batch_size:
                        self._dataup.evaluations.ingest_batch(
                            evaluation_id,
                            BatchIngestRequest(frames=current_batch),
                        )
                        current_batch = []

            # Submit any remaining frames
            if current_batch:
                self._dataup.evaluations.ingest_batch(
                    evaluation_id,
                    BatchIngestRequest(frames=current_batch),
                )

            # Finalize the evaluation
            self._dataup.evaluations.finalize(evaluation_id)

            # Return the complete evaluation with metrics
            return self._dataup.evaluations.get(evaluation_id)

        except Exception:
            # If something goes wrong, the evaluation will be in a failed state
            # Re-raise the exception for the caller to handle
            raise
