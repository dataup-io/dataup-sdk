"""Task models for the CVAT client."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dataup.cvat.models.enums import TaskStatus


class CVATLabelAttribute(BaseModel):
    """Label attribute definition."""

    id: int
    name: str
    mutable: bool = False
    input_type: str = "text"
    default_value: str = ""
    values: list[str] = Field(default_factory=list)


class CVATLabel(BaseModel):
    """Task label definition."""

    id: int
    name: str
    color: str | None = None
    type: str | None = None
    attributes: list[CVATLabelAttribute] = Field(default_factory=list)
    sublabels: list[CVATLabel] = Field(default_factory=list)


class TaskOwner(BaseModel):
    """Task owner information."""

    id: int
    username: str
    url: str | None = None


class DataMeta(BaseModel):
    """Task data metadata."""

    chunk_size: int | None = None
    size: int = 0  # Total number of frames
    image_quality: int = 70
    start_frame: int = 0
    stop_frame: int = 0
    frame_filter: str = ""


class JobSummary(BaseModel):
    """Summary of a job within a task."""

    id: int
    url: str | None = None
    status: str
    stage: str
    assignee: TaskOwner | None = None


class Task(BaseModel):
    """CVAT Task model."""

    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    url: str | None = None
    project_id: int | None = None
    status: TaskStatus
    size: int = 0  # Number of frames
    mode: str = ""  # "annotation" or "interpolation"
    owner: TaskOwner | None = None
    assignee: TaskOwner | None = None
    created_date: datetime | None = None
    updated_date: datetime | None = None
    overlap: int | None = None
    segment_size: int = 0
    labels: list[CVATLabel] = Field(default_factory=list)
    jobs: list[JobSummary] = Field(default_factory=list)
    dimension: str = "2d"
    subset: str = ""
    organization: int | None = None
    target_storage: dict | None = None
    source_storage: dict | None = None

    @field_validator("jobs", mode="before")
    @classmethod
    def extract_jobs_from_paginated_response(cls, v: list[JobSummary] | dict) -> list[JobSummary]:
        """Extract jobs list from paginated response if needed."""
        if isinstance(v, dict):
            # If it's a paginated response, extract the results
            if "results" in v:
                return v["results"]
            # If it's a dict but not paginated, return empty list
            return []
        return v

    @field_validator("labels", mode="before")
    @classmethod
    def extract_labels_from_paginated_response(cls, v: list[CVATLabel] | dict) -> list[CVATLabel]:
        """Extract labels list from paginated response if needed."""
        if isinstance(v, dict):
            # If it's a paginated response, extract the results
            if "results" in v:
                return v["results"]
            # If it's a dict but not paginated, return empty list
            return []
        return v


class TaskSummary(BaseModel):
    """Summary view of a task (for list operations)."""

    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    project_id: int | None = None
    status: TaskStatus
    size: int = 0
    mode: str = ""
    owner: TaskOwner | None = None
    assignee: TaskOwner | None = None
    created_date: datetime | None = None
    updated_date: datetime | None = None


# Forward reference resolution
# This is needed to resolve forward references (i.e., references to classes
# not yet fully defined) in the CVATLabel model.
CVATLabel.model_rebuild()
