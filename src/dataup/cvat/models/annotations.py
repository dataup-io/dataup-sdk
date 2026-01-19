"""Annotation models for the CVAT client."""

from __future__ import annotations

from dataup_models.labels import Label
from pydantic import BaseModel, ConfigDict, Field

from dataup.cvat.models.enums import ShapeType


class AttributeValue(BaseModel):
    """Attribute value on a shape or track."""

    spec_id: int
    value: str


class Shape(BaseModel):
    """Base shape annotation."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    type: ShapeType
    frame: int
    label_id: int
    group: int | None = 0
    source: str = "manual"
    occluded: bool = False
    outside: bool = False
    z_order: int = 0
    rotation: float = 0.0
    points: list[float] = Field(default_factory=list)
    attributes: list[AttributeValue] = Field(default_factory=list)


class LabeledShape(Shape):
    """Shape with label information resolved."""

    label: str | None = None  # Resolved label name


class Track(BaseModel):
    """Track annotation (for video)."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    label_id: int
    group: int | None = 0
    source: str = "manual"
    shapes: list[Shape] = Field(default_factory=list)
    attributes: list[AttributeValue] = Field(default_factory=list)


class Tag(BaseModel):
    """Tag annotation (frame-level)."""

    id: int | None = None
    frame: int
    label_id: int
    group: int | None = 0
    source: str = "manual"
    attributes: list[AttributeValue] = Field(default_factory=list)


class Annotations(BaseModel):
    """Complete annotations for a task or job."""

    model_config = ConfigDict(extra="ignore")

    version: int = 0
    tags: list[Tag] = Field(default_factory=list)
    shapes: list[Shape] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)


class FrameAnnotations(BaseModel):
    """Annotations for a single frame."""

    frame_id: int
    shapes: list[Shape] = Field(default_factory=list)
    tags: list[Tag] = Field(default_factory=list)


# class FrameLabels(BaseModel):
#     frame_id: int
#     labels: list[Label]


class FrameLabels(BaseModel):
    frame_id: int
    job_id: int
    labels: list[Label]
