"""Synchronous CVAT API client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import httpx

from dataup.cvat._base import DEFAULT_CVAT_URL, DEFAULT_TIMEOUT, BaseCVATClient
from dataup.cvat.exceptions import (
    CVATAPIError,
    CVATAuthenticationError,
    CVATConnectionError,
    CVATNotFoundError,
    CVATPermissionDeniedError,
    CVATRateLimitError,
    CVATTimeoutError,
    CVATValidationError,
)
from dataup.cvat.models.annotations import Annotations, FrameLabels
from dataup.cvat.models.common import PaginatedResponse
from dataup.cvat.models.frames import DataMetaInfo, FrameImage
from dataup.cvat.models.jobs import Job, JobSummary
from dataup.cvat.models.tasks import CVATLabel, Task, TaskSummary
from dataup.cvat.utils import shape_to_label

if TYPE_CHECKING:
    from dataup.cvat.models.enums import AnnotationFormat


class TasksResource:
    """Tasks API resource."""

    def __init__(self, client: CVATClient) -> None:
        self._client = client

    def list(
        self,
        *,
        page: int = 1,
        page_size: int = 10,
        search: str | None = None,
        project_id: int | None = None,
        status: str | None = None,
    ) -> PaginatedResponse[TaskSummary]:
        """List tasks with optional filters and pagination.

        Args:
            page: Page number (1-indexed).
            page_size: Number of items per page.
            search: Search string for task names.
            project_id: Filter by project ID.
            status: Filter by task status.

        Returns:
            Paginated response containing task summaries.
        """
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if search:
            params["search"] = search
        if project_id is not None:
            params["project_id"] = project_id
        if status:
            params["status"] = status

        response = self._client._request("GET", "/tasks", params=params)
        data = response.json()
        return PaginatedResponse[TaskSummary](
            count=data["count"],
            next=data.get("next"),
            previous=data.get("previous"),
            results=[TaskSummary.model_validate(t) for t in data["results"]],
        )

    def get(self, task_id: int) -> Task:
        """Get a specific task by ID.

        Args:
            task_id: The task ID.

        Returns:
            The task details with jobs populated.
        """
        response = self._client._request("GET", f"/tasks/{task_id}")
        task = Task.model_validate(response.json())

        # Fetch jobs separately since API doesn't include them in task response
        jobs_response = self._client.jobs.list(task_id=task_id, page_size=1000)
        task.jobs = jobs_response.results

        return task

    def get_task_labels(self, task_id: int) -> list[CVATLabel]:
        """Get all labels for a task.

        Args:
            task_id: The task ID.

        Returns:
            List of CVATLabel objects for the task.
        """
        response = self._client._request("GET", "/labels", params={"task_id": task_id})
        data = response.json()
        # Handle both paginated response and direct list
        if isinstance(data, dict) and "results" in data:
            # Paginated response
            return [CVATLabel.model_validate(label_data) for label_data in data["results"]]
        elif isinstance(data, list):
            # Direct list response
            return [CVATLabel.model_validate(label_data) for label_data in data]
        else:
            # Single item or unexpected format
            return [CVATLabel.model_validate(data)]

    def get_data_meta(self, task_id: int) -> DataMetaInfo:
        """Get data metadata for a task (frame info, dimensions, etc.).

        Args:
            task_id: The task ID.

        Returns:
            Data metadata including frame information.
        """
        response = self._client._request("GET", f"/tasks/{task_id}/data/meta")
        return DataMetaInfo.model_validate(response.json())

    def get_frame(self, task_id: int, frame_id: int, *, quality: str = "original") -> FrameImage:
        """Get a single frame/image from a task.

        Args:
            task_id: The task ID.
            frame_id: The frame number.
            quality: Image quality - "original" or "compressed".

        Returns:
            Frame image data.
        """
        params = {"quality": quality, "type": "frame", "number": frame_id}
        response = self._client._request(
            "GET",
            f"/tasks/{task_id}/data",
            params=params,
            accept="image/*",
        )

        content_type = response.headers.get("content-type", "image/jpeg")
        return FrameImage(
            frame_id=frame_id,
            data=response.content,
            content_type=content_type,
        )

    def iter_frames(
        self,
        task_id: int,
        *,
        start_frame: int | None = None,
        stop_frame: int | None = None,
        quality: str = "original",
    ) -> Iterator[FrameImage]:
        """Iterate through frames in a task.

        Args:
            task_id: The task ID.
            start_frame: Starting frame number (inclusive). Defaults to task's start frame.
            stop_frame: Ending frame number (inclusive). Defaults to task's stop frame.
            quality: Image quality - "original" or "compressed".

        Yields:
            Frame images in order.
        """
        meta = self.get_data_meta(task_id)
        start = start_frame if start_frame is not None else meta.start_frame
        stop = stop_frame if stop_frame is not None else meta.stop_frame

        for frame_id in range(start, stop + 1):
            if frame_id not in meta.deleted_frames:
                yield self.get_frame(task_id, frame_id, quality=quality)

    def get_annotations(self, task_id: int) -> list[FrameLabels]:
        """Get all annotations for a task.

        Args:
            task_id: The task ID.

        Returns:
            List of FrameLabels, one per frame with annotations.
        """
        # Get annotations
        response = self._client._request("GET", f"/tasks/{task_id}/annotations")
        annotations = Annotations.model_validate(response.json())

        # Get task to resolve label names and get jobs
        task = self.get(task_id)

        # Get CVAT labels for attribute resolution
        cvat_labels = self.get_task_labels(task_id)

        # Create frame -> job_id mapping
        # Get all jobs for the task to map frames to jobs
        jobs_response = self._client.jobs.list(task_id=task_id, page_size=1000)
        frame_to_job: dict[int, int] = {}
        for job_summary in jobs_response.results:
            for frame_id in range(job_summary.start_frame, job_summary.stop_frame + 1):
                frame_to_job[frame_id] = job_summary.id

        # Group shapes by frame
        frames_shapes: dict[int, list[dict[str, Any]]] = {}
        for shape in annotations.shapes:
            frame_id = shape.frame
            if frame_id not in frames_shapes:
                frames_shapes[frame_id] = []
            # Convert Shape to dict for shape_to_label
            shape_dict = shape.model_dump()
            frames_shapes[frame_id].append(shape_dict)

        # Convert to FrameLabels
        frame_labels_list: list[FrameLabels] = []
        for frame_id, shapes in frames_shapes.items():
            # Get job_id for this frame (default to first job if not found)
            job_id = frame_to_job.get(frame_id)
            if job_id is None and task.jobs:
                # Fallback: use first job if frame not mapped
                job_id = task.jobs[0].id
            elif job_id is None:
                # If no jobs, we can't create FrameLabels - skip
                continue

            # Convert shapes to Labels
            labels = []
            for shape_dict in shapes:
                try:
                    label = shape_to_label(shape_dict, cvat_labels)
                    labels.append(label)
                except Exception:
                    # Skip shapes that can't be converted
                    continue

            frame_labels_list.append(FrameLabels(frame_id=frame_id, job_id=job_id, labels=labels))

        return frame_labels_list

    def get_frame_annotations(self, task_id: int, frame_id: int) -> FrameLabels:
        """Get annotations for a specific frame in a task.

        Args:
            task_id: The task ID.
            frame_id: The frame number.

        Returns:
            FrameLabels for the specified frame.
        """
        frame_labels_list = self.get_annotations(task_id)
        # Find the FrameLabels for this frame_id
        for frame_labels in frame_labels_list:
            if frame_labels.frame_id == frame_id:
                return frame_labels
        # If not found, return empty FrameLabels
        # Need to get job_id - use first job from task
        task = self.get(task_id)
        job_id = task.jobs[0].id if task.jobs else 0
        return FrameLabels(frame_id=frame_id, job_id=job_id, labels=[])

    def export_annotations(
        self,
        task_id: int,
        format: AnnotationFormat | str,
    ) -> bytes:
        """Export annotations in a specific format.

        Args:
            task_id: The task ID.
            format: Export format (e.g., AnnotationFormat.COCO).

        Returns:
            Exported annotation data as bytes (typically a ZIP archive).
        """
        format_str = format.value if hasattr(format, "value") else format
        params = {"format": format_str, "action": "download"}

        response = self._client._request(
            "GET",
            f"/tasks/{task_id}/annotations",
            params=params,
            accept="*/*",
        )
        return response.content


class JobsResource:
    """Jobs API resource."""

    def __init__(self, client: CVATClient) -> None:
        self._client = client

    def list(
        self,
        *,
        page: int = 1,
        page_size: int = 10,
        task_id: int | None = None,
    ) -> PaginatedResponse[JobSummary]:
        """List jobs with optional filters and pagination.

        Args:
            page: Page number (1-indexed).
            page_size: Number of items per page.
            task_id: Filter by task ID.

        Returns:
            Paginated response containing job summaries.
        """
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if task_id is not None:
            params["task_id"] = task_id

        response = self._client._request("GET", "/jobs", params=params)
        data = response.json()
        return PaginatedResponse[JobSummary](
            count=data["count"],
            next=data.get("next"),
            previous=data.get("previous"),
            results=[JobSummary.model_validate(j) for j in data["results"]],
        )

    def get(self, job_id: int) -> Job:
        """Get a specific job by ID.

        Args:
            job_id: The job ID.

        Returns:
            The job details.
        """
        response = self._client._request("GET", f"/jobs/{job_id}")
        return Job.model_validate(response.json())

    def get_job_labels(self, job_id: int) -> list[CVATLabel]:
        """Get all labels for a job.

        Args:
            job_id: The job ID.

        Returns:
            List of CVATLabel objects for the job.
        """
        response = self._client._request("GET", "/labels", params={"job_id": job_id})
        data = response.json()
        if isinstance(data, dict) and "results" in data:
            return [CVATLabel.model_validate(label_data) for label_data in data["results"]]
        elif isinstance(data, list):
            return [CVATLabel.model_validate(label_data) for label_data in data]
        else:
            return [CVATLabel.model_validate(data)]

    def get_data_meta(self, job_id: int) -> DataMetaInfo:
        """Get data metadata for a job.

        Args:
            job_id: The job ID.

        Returns:
            Data metadata including frame information.
        """
        response = self._client._request("GET", f"/jobs/{job_id}/data/meta")
        return DataMetaInfo.model_validate(response.json())

    def get_frame(
        self,
        job_id: int,
        frame_id: int,
        *,
        quality: str = "original",
    ) -> FrameImage:
        """Get a single frame/image from a job.

        Args:
            job_id: The job ID.
            frame_id: The frame number.
            quality: Image quality - "original" or "compressed".

        Returns:
            Frame image data.
        """
        params = {"quality": quality, "type": "frame", "number": frame_id}
        response = self._client._request(
            "GET",
            f"/jobs/{job_id}/data",
            params=params,
            accept="image/*",
        )

        content_type = response.headers.get("content-type", "image/jpeg")
        return FrameImage(
            frame_id=frame_id,
            data=response.content,
            content_type=content_type,
        )

    def iter_frames(
        self,
        job_id: int,
        *,
        quality: str = "original",
    ) -> Iterator[FrameImage]:
        """Iterate through frames in a job.

        Args:
            job_id: The job ID.
            quality: Image quality - "original" or "compressed".

        Yields:
            Frame images in order.
        """
        job = self.get(job_id)
        meta = self.get_data_meta(job_id)

        for frame_id in range(job.start_frame, job.stop_frame + 1):
            if frame_id not in meta.deleted_frames:
                yield self.get_frame(job_id, frame_id, quality=quality)

    def get_annotations(
        self, job_id: int, *, cvat_labels: list[CVATLabel] | None = None
    ) -> list[FrameLabels]:
        """Get all annotations for a job.

        Args:
            job_id: The job ID.
            cvat_labels: Optional pre-fetched CVAT labels for the job.
                If not provided, labels will be fetched automatically.

        Returns:
            List of FrameLabels, one per frame with annotations.
        """
        response = self._client._request("GET", f"/jobs/{job_id}/annotations")
        annotations = Annotations.model_validate(response.json())
        # Fetch labels if not provided
        if cvat_labels is None:
            cvat_labels = self.get_job_labels(job_id)

        # Group shapes by frame
        frames_shapes: dict[int, list[dict[str, Any]]] = {}
        for shape in annotations.shapes:
            frame_id = shape.frame
            if frame_id not in frames_shapes:
                frames_shapes[frame_id] = []
            frames_shapes[frame_id].append(shape.model_dump())

        # Convert to FrameLabels
        frame_labels_list: list[FrameLabels] = []
        for frame_id, shapes in sorted(frames_shapes.items()):
            labels = []
            for shape_dict in shapes:
                try:
                    label = shape_to_label(shape_dict, cvat_labels)
                    labels.append(label)
                except Exception:
                    continue

            frame_labels_list.append(FrameLabels(frame_id=frame_id, job_id=job_id, labels=labels))

        return frame_labels_list

    def get_annotations_raw(self, job_id: int) -> Annotations:
        """Get raw CVAT annotations for a job.

        Args:
            job_id: The job ID.

        Returns:
            Raw CVAT Annotations object (shapes, tracks, tags).
        """
        response = self._client._request("GET", f"/jobs/{job_id}/annotations")
        return Annotations.model_validate(response.json())

    def get_frame_annotations(
        self,
        job_id: int,
        frame_id: int,
        *,
        cvat_labels: list[CVATLabel] | None = None,
    ) -> FrameLabels:
        """Get annotations for a specific frame in a job.

        Args:
            job_id: The job ID.
            frame_id: The frame number.
            cvat_labels: Optional pre-fetched CVAT labels for the job.

        Returns:
            FrameLabels for the specified frame.
        """
        frame_labels_list = self.get_annotations(job_id, cvat_labels=cvat_labels)
        for frame_labels in frame_labels_list:
            if frame_labels.frame_id == frame_id:
                return frame_labels
        return FrameLabels(frame_id=frame_id, job_id=job_id, labels=[])

    def iter_jobs_with_annotations(
        self,
        task_id: int,
        *,
        page_size: int = 100,
    ) -> Iterator[tuple[JobSummary, list[FrameLabels]]]:
        """Iterate over all jobs for a task with their annotations.

        This method is optimized for fetching annotations across multiple jobs,
        useful for submitting evaluations. It fetches labels once per task
        and reuses them for all jobs.

        Args:
            task_id: The task ID.
            page_size: Number of jobs to fetch per page.

        Yields:
            Tuples of (JobSummary, list[FrameLabels]) for each job.
        """
        # Fetch labels once for the task (shared across all jobs)
        cvat_labels = self._client.tasks.get_task_labels(task_id)

        # Paginate through all jobs
        page = 1
        while True:
            jobs_response = self.list(task_id=task_id, page=page, page_size=page_size)
            for job_summary in jobs_response.results:
                annotations = self.get_annotations(job_id=job_summary.id, cvat_labels=cvat_labels)
                yield job_summary, annotations

            if jobs_response.next is None:
                break
            page += 1


class CVATClient(BaseCVATClient):
    """Synchronous CVAT API client.

    Example:
        >>> from dataup.cvat import CVATClient
        >>> client = CVATClient(api_token="your-token")
        >>> tasks = client.tasks.list()
        >>> for task in tasks.results:
        ...     print(task.name)

        # Using context manager
        >>> with CVATClient(api_token="your-token") as client:
        ...     task = client.tasks.get(task_id=123)
    """

    tasks: TasksResource
    jobs: JobsResource

    def __init__(
        self,
        api_token: str,
        *,
        base_url: str = DEFAULT_CVAT_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the CVAT client.

        Args:
            api_token: CVAT API token for authentication.
            base_url: CVAT server URL. Defaults to https://app.cvat.ai.
            timeout: Request timeout in seconds. Defaults to 60.
            http_client: Optional custom httpx.Client instance.
        """
        super().__init__(api_token, base_url=base_url, timeout=timeout)

        self._client = http_client or httpx.Client(
            timeout=self.timeout,
            headers=self._headers,
        )

        # Initialize resources
        self.tasks = TasksResource(self)
        self.jobs = JobsResource(self)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        accept: str = "application/json",
    ) -> httpx.Response:
        """Make HTTP request and handle errors."""
        url = self._build_url(path)
        headers = dict(self._headers)
        if json is not None:
            headers["Content-Type"] = "application/json"

        try:
            response = self._client.request(
                method,
                url,
                params=params,
                json=json,
                headers=headers,
            )
        except httpx.ConnectError as e:
            raise CVATConnectionError(f"Failed to connect to CVAT: {e}") from e
        except httpx.TimeoutException as e:
            raise CVATTimeoutError(f"Request timed out: {e}") from e

        self._handle_response(response)
        return response

    def _handle_response(self, response: httpx.Response) -> None:
        """Handle HTTP response and raise appropriate exceptions."""
        if response.is_success:
            return

        status_code = response.status_code
        try:
            error_data = response.json()
            message = error_data.get("detail", error_data.get("message", response.text))
        except Exception:
            message = response.text

        if status_code == 400:
            raise CVATValidationError(message, status_code=status_code)
        elif status_code == 401:
            raise CVATAuthenticationError(message, status_code=status_code)
        elif status_code == 403:
            raise CVATPermissionDeniedError(message, status_code=status_code)
        elif status_code == 404:
            raise CVATNotFoundError(message, status_code=status_code)
        elif status_code == 429:
            raise CVATRateLimitError(message, status_code=status_code)
        else:
            raise CVATAPIError(message, status_code=status_code)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> CVATClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
