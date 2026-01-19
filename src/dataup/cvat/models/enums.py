"""Enum definitions for the CVAT client."""

from enum import Enum


class TaskStatus(str, Enum):
    """Status of a CVAT task."""

    ANNOTATION = "annotation"
    VALIDATION = "validation"
    COMPLETED = "completed"


class JobStatus(str, Enum):
    """Status of a CVAT job."""

    ANNOTATION = "annotation"
    VALIDATION = "validation"
    COMPLETED = "completed"
    REJECTED = "rejected"


class JobStage(str, Enum):
    """Stage of a CVAT job."""

    ANNOTATION = "annotation"
    VALIDATION = "validation"
    ACCEPTANCE = "acceptance"


class ShapeType(str, Enum):
    """Type of annotation shape."""

    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    POLYLINE = "polyline"
    POINTS = "points"
    ELLIPSE = "ellipse"
    CUBOID = "cuboid"
    SKELETON = "skeleton"
    MASK = "mask"


class AnnotationFormat(str, Enum):
    """Supported annotation export formats."""

    CVAT_FOR_IMAGES = "CVAT for images 1.1"
    CVAT_FOR_VIDEO = "CVAT for video 1.1"
    COCO = "COCO 1.0"
    YOLO = "YOLO 1.1"
    PASCAL_VOC = "PASCAL VOC 1.1"
