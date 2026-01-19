"""CVAT client for DataUp SDK - Interact with CVAT annotation platform.

This module provides synchronous and asynchronous clients for interacting
with the CVAT (Computer Vision Annotation Tool) API.

Example:
    Synchronous usage::

        from dataup.cvat import CVATClient

        client = CVATClient(api_token="your-cvat-token")

        # List tasks
        tasks = client.tasks.list()
        for task in tasks.results:
            print(task.name)

        # Get annotations
        annotations = client.tasks.get_annotations(task_id=123)

        # Download frames
        for frame in client.tasks.iter_frames(task_id=123):
            with open(f"frame_{frame.frame_id}.jpg", "wb") as f:
                f.write(frame.data)

    Asynchronous usage::

        import asyncio
        from dataup.cvat import AsyncCVATClient

        async def main():
            async with AsyncCVATClient(api_token="your-token") as client:
                task = await client.tasks.get(123)
                annotations = await client.tasks.get_annotations(123)

        asyncio.run(main())
"""

from dataup.cvat.async_client import AsyncCVATClient
from dataup.cvat.client import CVATClient
from dataup.cvat.exceptions import (
    CVATAPIError,
    CVATAuthenticationError,
    CVATConnectionError,
    CVATError,
    CVATNotFoundError,
    CVATPermissionDeniedError,
    CVATRateLimitError,
    CVATTimeoutError,
    CVATValidationError,
)
from dataup.cvat.models.annotations import (
    Annotations,
    AttributeValue,
    FrameAnnotations,
    LabeledShape,
    Shape,
    Tag,
    Track,
)
from dataup.cvat.models.common import FrameData, PaginatedResponse
from dataup.cvat.models.enums import (
    AnnotationFormat,
    JobStage,
    JobStatus,
    ShapeType,
    TaskStatus,
)
from dataup.cvat.models.frames import DataMetaInfo, FrameImage, FrameMeta
from dataup.cvat.models.jobs import Job, JobSummary
from dataup.cvat.models.tasks import (
    CVATLabel,
    CVATLabelAttribute,
    Task,
    TaskOwner,
    TaskSummary,
)

__all__ = [
    # Clients
    "CVATClient",
    "AsyncCVATClient",
    # Exceptions
    "CVATError",
    "CVATAPIError",
    "CVATAuthenticationError",
    "CVATPermissionDeniedError",
    "CVATNotFoundError",
    "CVATValidationError",
    "CVATRateLimitError",
    "CVATConnectionError",
    "CVATTimeoutError",
    # Enums
    "TaskStatus",
    "JobStatus",
    "JobStage",
    "ShapeType",
    "AnnotationFormat",
    # Task models
    "Task",
    "TaskSummary",
    "TaskOwner",
    "CVATLabel",
    "CVATLabelAttribute",
    # Job models
    "Job",
    "JobSummary",
    # Frame models
    "FrameData",
    "FrameImage",
    "FrameMeta",
    "DataMetaInfo",
    # Annotation models
    "Annotations",
    "FrameAnnotations",
    "Shape",
    "LabeledShape",
    "Track",
    "Tag",
    "AttributeValue",
    # Common
    "PaginatedResponse",
]
