"""Inference models for the DataUp SDK."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel, frozen=True):
    """Bounding box coordinates."""

    x: int
    y: int
    width: int
    height: int


class Polygon(BaseModel, frozen=True):
    """Polygon coordinates."""

    points: list[tuple[int, int]] = Field(description="List of points in the polygon")

    def to_bbox(self) -> BoundingBox:
        """Convert polygon to bounding box."""
        x_coords = [x for x, y in self.points]
        y_coords = [y for x, y in self.points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        return BoundingBox(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)


class LabelAttribute(BaseModel, frozen=True):
    """Label attribute key-value pair."""

    key: str
    value: str


class Label(BaseModel, frozen=True):
    """Detection label."""

    label: str
    score: float
    bbox: BoundingBox
    polygon: Polygon | None = None
    rle_mask: str = ""
    attributes: list[LabelAttribute] = Field(default_factory=list)


class DetectorParams(BaseModel):
    """Detector model parameters."""

    param_type: Literal["detector"] = "detector"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="IoU threshold")
    max_detections: int = Field(default=100, ge=1, le=1000, description="Max detections per image")
    prompt: str | None = None


class SAM3Params(BaseModel):
    """SAM3 model parameters."""

    param_type: Literal["sam3"] = "sam3"
    text_prompt: str | None = Field(default=None, description="Text prompt for SAM3 inference")
    geom_prompt: dict[str, Any] | None = Field(
        default=None, description="Geometric prompt for SAM3 inference"
    )


ModelParams = Annotated[DetectorParams | SAM3Params, Field(discriminator="param_type")]


class InferenceRequest(BaseModel, frozen=True):
    """Inference request schema."""

    request_id: str | None = Field(default=None, description="Unique request ID")
    image_urls: list[str] | None = Field(
        default=None, description="List of image URLs to run inference on"
    )
    images_b64: list[str] | None = Field(
        default=None, description="List of base64 encoded images to run inference on"
    )
    session_id: str | None = Field(default=None, description="Session ID")
    params: DetectorParams | SAM3Params = Field(
        default_factory=DetectorParams, description="Model inference parameters"
    )


class DetectionResults(BaseModel):
    """Detection results for a single image."""

    image_id: str = Field(description="Image ID")
    labels: list[Label] = Field(description="List of detected labels")


class InferenceResponse(BaseModel):
    """Inference response schema."""

    data: list[DetectionResults] | dict[str, Any]
    success: bool = True
    session_id: str | None = Field(default=None, description="Session ID")
    error: str = ""
