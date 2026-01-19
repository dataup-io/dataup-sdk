"""Evaluation models for the DataUp SDK."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from dataup_models.labels import Label
from pydantic import BaseModel, Field, computed_field


class PredictionSource(str, Enum):
    """Enum for prediction source."""

    SERVER_AGENT = "SERVER_AGENT"
    CLIENT_SDK = "CLIENT_SDK"


class EvaluationStatus(str, Enum):
    """Enum for evaluation status."""

    CREATED = "CREATED"
    INGESTING = "INGESTING"
    FINALIZING = "FINALIZING"
    DONE = "DONE"
    FAILED = "FAILED"


# --- Request schemas ---


class EvaluationCreate(BaseModel):
    """Schema for creating a new evaluation."""

    agent_id: str | None = Field(
        default=None, description="Agent ID (optional if provided in URL path)"
    )
    dataset_id: int = Field(ge=0, description="Dataset identifier")
    agent_name: str | None = Field(
        default=None, max_length=256, description="Name of the agent (optional)"
    )
    agent_version: str | None = Field(
        default=None, max_length=40, description="Version of the agent (e.g., '1.0.0')"
    )
    prediction_source: PredictionSource = Field(
        default=PredictionSource.CLIENT_SDK,
        description="Source of predictions: SERVER_AGENT or CLIENT_SDK",
    )
    total_frames: int | None = Field(
        default=None, ge=0, description="Expected total number of frames (set when known)"
    )
    settings: dict[str, Any] | None = Field(
        default=None, description="COCO config, label map version, and other settings"
    )


class FrameData(BaseModel):
    """Schema for frame data in a batch."""

    job_id: int = Field(ge=0, description="Job identifier")
    frame_id: int = Field(ge=0, description="Frame identifier within the job")
    ground_truth: list[Label] = Field(
        default_factory=list, description="List of ground truth labels"
    )
    predictions: list[Label] | None = Field(
        default=None,
        description=(
            "List of predicted labels (required for CLIENT_SDK mode, ignored for SERVER_AGENT mode)"
        ),
    )
    image_width: int | None = Field(default=None, ge=0, description="Width of the image in pixels")
    image_height: int | None = Field(
        default=None, ge=0, description="Height of the image in pixels"
    )


class BatchIngestRequest(BaseModel):
    """Schema for batch ingestion request."""

    frames: list[FrameData] = Field(description="List of frame data to ingest")


# --- Response schemas ---


class ThresholdMetrics(BaseModel):
    """Metrics at a specific confidence threshold."""

    confidence: float = Field(ge=0.0, le=1.0, description="Confidence threshold")
    true_positive: int = Field(ge=0, description="Number of true positive detections")
    false_positive: int = Field(ge=0, description="Number of false positive detections")
    false_negative: int = Field(ge=0, description="Number of false negative detections")
    precision: float = Field(ge=0.0, le=1.0, description="Precision at this threshold")
    recall: float = Field(ge=0.0, le=1.0, description="Recall at this threshold")
    f1: float = Field(ge=0.0, le=1.0, description="F1 score at this threshold")


class ClassConfusionMatrix(BaseModel):
    """Confusion matrix showing how predictions are distributed across actual classes."""

    classes: list[str] = Field(
        description="Ordered list of class names (includes 'background' for unmatched)"
    )
    matrix: dict[str, dict[str, int]] = Field(
        description="Nested dict: {predicted_class: {actual_class: count}}"
    )


class EvaluationCreateResponse(BaseModel):
    """Response for evaluation creation."""

    evaluation_id: str = Field(description="Evaluation ID")


class BatchIngestResponse(BaseModel):
    """Response for batch ingestion."""

    evaluation_id: str = Field(description="Evaluation ID")
    frames_in_batch: int = Field(ge=0, description="Number of frames processed in this batch")
    received_frames: int = Field(ge=0, description="Total frames received so far")


class COCOSummaryMetrics(BaseModel):
    """COCO summary metrics computed at finalize time."""

    ap: float = Field(ge=0.0, le=1.0, description="AP @ IoU=0.50:0.95")
    ap_50: float = Field(ge=0.0, le=1.0, description="AP @ IoU=0.50")
    ap_75: float = Field(ge=0.0, le=1.0, description="AP @ IoU=0.75")
    ap_small: float | None = Field(default=None, ge=0.0, le=1.0, description="AP for small")
    ap_medium: float | None = Field(default=None, ge=0.0, le=1.0, description="AP for medium")
    ap_large: float | None = Field(default=None, ge=0.0, le=1.0, description="AP for large")
    ar_1: float = Field(ge=0.0, le=1.0, description="AR @ max detections=1")
    ar_10: float = Field(ge=0.0, le=1.0, description="AR @ max detections=10")
    ar_100: float = Field(ge=0.0, le=1.0, description="AR @ max detections=100")
    ar_small: float | None = Field(default=None, ge=0.0, le=1.0, description="AR for small")
    ar_medium: float | None = Field(default=None, ge=0.0, le=1.0, description="AR for medium")
    ar_large: float | None = Field(default=None, ge=0.0, le=1.0, description="AR for large")
    threshold_metrics: list[ThresholdMetrics] = Field(
        default_factory=list,
        description=(
            "Precision, recall, and F1 at different confidence thresholds "
            "aggregated across all classes"
        ),
    )
    confusion_matrix: ClassConfusionMatrix | None = Field(
        default=None,
        description="Class confusion matrix (computed at IoU=0.5)",
    )


class PerClassCOCOMetrics(BaseModel):
    """Per-class COCO metrics."""

    class_name: str = Field(max_length=256, description="Name of the class")
    ap: float = Field(ge=0.0, le=1.0, description="AP @ IoU=0.50:0.95")
    ap_50: float = Field(ge=0.0, le=1.0, description="AP @ IoU=0.50")
    ap_75: float = Field(ge=0.0, le=1.0, description="AP @ IoU=0.75")
    ar_100: float = Field(ge=0.0, le=1.0, description="AR @ max detections=100")
    ground_truth_count: int = Field(ge=0, description="Ground truth annotations count")
    detection_count: int = Field(ge=0, description="Number of detections for this class")
    threshold_metrics: list[ThresholdMetrics] = Field(
        default_factory=list, description="Metrics at different confidence thresholds"
    )


class EvaluationRead(BaseModel):
    """Schema for reading an evaluation."""

    id: str = Field(description="Evaluation ID")
    owner_id: str = Field(description="Owner user ID")
    agent_id: str | None = Field(
        default=None,
        description="Agent ID (required for SERVER_AGENT mode, optional for CLIENT_SDK mode)",
    )
    agent_name: str | None = Field(default=None, description="Name of the agent")
    agent_version: str | None = Field(default=None, description="Version of the agent")
    dataset_id: int = Field(ge=0, description="Dataset identifier")
    prediction_source: PredictionSource = Field(description="Source of predictions")
    status: EvaluationStatus = Field(description="Evaluation status")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    total_frames: int | None = Field(default=None, description="Expected total number of frames")
    received_frames: int = Field(description="Number of frames received so far")
    settings: dict[str, Any] | None = Field(default=None, description="Evaluation settings")
    summary_metrics: COCOSummaryMetrics | None = Field(
        default=None, description="COCO summary metrics"
    )
    per_class_summary_metrics: list[PerClassCOCOMetrics] | None = Field(
        default=None, description="Per-class COCO summary metrics"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def progress(self) -> float | None:
        """
        Progress as a ratio (0.0 to 1.0) representing processed frames / total frames.

        Returns None if total_frames is unknown (None or 0).
        Returns 1.0 if received_frames >= total_frames (capped at 100%).
        """
        if self.total_frames is None or self.total_frames == 0:
            return None
        return min(self.received_frames / self.total_frames, 1.0)


class EvaluationFrameRead(BaseModel):
    """Schema for reading an evaluation frame."""

    id: str = Field(description="Frame record ID")
    evaluation_id: str = Field(description="Evaluation ID")
    job_id: int = Field(ge=0, description="Job identifier")
    frame_id: int = Field(ge=0, description="Frame identifier within the job")
    ground_truth: list[Label] = Field(description="List of ground truth labels")
    predictions: list[Label] = Field(description="List of predicted labels")
    threshold_metrics: list[ThresholdMetrics] = Field(
        description="Metrics at different confidence thresholds"
    )
    per_class_threshold_metrics: dict[str, list[ThresholdMetrics]] | None = Field(
        default=None, description="Per-class threshold metrics"
    )
    image_width: int | None = Field(default=None, description="Width of the image in pixels")
    image_height: int | None = Field(default=None, description="Height of the image in pixels")
    created_at: datetime = Field(description="Creation timestamp")


class FinalizeResponse(BaseModel):
    """Response for finalize endpoint."""

    evaluation_id: str = Field(description="Evaluation ID")
    status: EvaluationStatus = Field(description="Final evaluation status")
    summary_metrics: COCOSummaryMetrics | None = Field(
        default=None, description="COCO summary metrics"
    )
    per_class_summary_metrics: list[PerClassCOCOMetrics] | None = Field(
        default=None, description="Per-class COCO summary metrics"
    )


class JobMetrics(BaseModel):
    """Job-level metrics for an evaluation."""

    job_id: int = Field(ge=0, description="Job identifier")
    job_name: str | None = Field(default=None, description="Name of the job (if available)")
    total_frames: int = Field(ge=0, description="Total number of frames in this job")
    frames_with_errors: int = Field(
        ge=0, description="Number of frames with errors (frames where FP > 0 or FN > 0)"
    )
    f1_score: float = Field(
        ge=0.0, le=1.0, description="F1 score at the specified confidence threshold"
    )
    precision: float = Field(
        ge=0.0, le=1.0, description="Precision at the specified confidence threshold"
    )
    recall: float = Field(
        ge=0.0, le=1.0, description="Recall at the specified confidence threshold"
    )
    total_fp: int = Field(ge=0, description="Total false positives across all frames")
    total_fn: int = Field(ge=0, description="Total false negatives across all frames")


class JobMetricsResponse(BaseModel):
    """Response for job metrics endpoint."""

    confidence_threshold: float = Field(
        ge=0.0, le=1.0, description="Confidence threshold used to compute metrics"
    )
    jobs: list[JobMetrics] = Field(default_factory=list, description="List of job metrics")
