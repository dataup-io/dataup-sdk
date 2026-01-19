"""CVAT models package."""

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
    DataMeta,
    Task,
    TaskOwner,
    TaskSummary,
)

__all__ = [
    # Enums
    "TaskStatus",
    "JobStatus",
    "JobStage",
    "ShapeType",
    "AnnotationFormat",
    # Common
    "PaginatedResponse",
    "FrameData",
    # Tasks
    "Task",
    "TaskSummary",
    "TaskOwner",
    "CVATLabel",
    "CVATLabelAttribute",
    "DataMeta",
    # Jobs
    "Job",
    "JobSummary",
    # Frames
    "FrameImage",
    "FrameMeta",
    "DataMetaInfo",
    # Annotations
    "Annotations",
    "FrameAnnotations",
    "Shape",
    "LabeledShape",
    "Track",
    "Tag",
    "AttributeValue",
]
