"""Job models for the CVAT client."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dataup.cvat.models.enums import JobStage, JobStatus
from dataup.cvat.models.tasks import CVATLabel, TaskOwner

# example = {
#     "url": "https://app.data-up.io/api/jobs/19517",
#     "id": 19517,
#     "task_id": 473,
#     "project_id": 68,
#     "assignee": None,
#     "guide_id": None,
#     "dimension": "2d",
#     "bug_tracker": "",
#     "status": "annotation",
#     "stage": "annotation",
#     "state": "new",
#     "mode": "annotation",
#     "frame_count": 1,
#     "start_frame": 75,
#     "stop_frame": 75,
#     "data_chunk_size": 19,
#     "data_compressed_chunk_type": "imageset",
#     "data_original_chunk_type": "imageset",
#     "created_date": "2025-12-24T18:49:39.375176Z",
#     "updated_date": "2025-12-24T18:49:39.375199Z",
#     "issues": {"url": "https://app.data-up.io/api/issues?job_id=19517", "count": 0},
#     "labels": {"url": "https://app.data-up.io/api/labels?job_id=19517"},
#     "type": "annotation",
#     "organization": 1,
#     "target_storage": {"id": 604, "location": "local", "cloud_storage_id": None},
#     "source_storage": {"id": 603, "location": "local", "cloud_storage_id": None},
#     "assignee_updated_date": None,
#     "parent_job_id": None,
#     "consensus_replicas": 0,
# }


class Job(BaseModel):
    """CVAT Job model."""

    model_config = ConfigDict(extra="ignore")

    id: int
    task_id: int
    project_id: int | None = None
    status: JobStatus
    stage: JobStage
    state: str = ""
    start_frame: int
    stop_frame: int
    assignee: TaskOwner | None = None
    updated_date: datetime | None = None
    created_date: datetime | None = None
    labels: dict | list[CVATLabel] = Field(default_factory=dict)
    dimension: str = "2d"
    data_chunk_size: int | None = None
    data_compressed_chunk_type: str = "imageset"
    data_original_chunk_type: str | None = None
    mode: str = ""
    bug_tracker: str = ""
    issues: dict | list[dict] = Field(default_factory=dict)
    url: str | None = None
    guide_id: int | None = None
    frame_count: int | None = None
    type: str = ""
    organization: int | None = None
    target_storage: dict | None = None
    source_storage: dict | None = None
    assignee_updated_date: datetime | None = None
    parent_job_id: int | None = None
    consensus_replicas: int = 0

    @field_validator("labels", mode="before")
    @classmethod
    def extract_labels_from_paginated_response(
        cls, v: list[CVATLabel] | dict
    ) -> list[CVATLabel] | dict:
        """Extract labels list from paginated response if needed."""
        if isinstance(v, dict):
            # If it's a paginated response, extract the results
            if "results" in v:
                return v["results"]
            # If it's a dict but not paginated (URL reference), return as-is
            return v
        return v

    @field_validator("issues", mode="before")
    @classmethod
    def extract_issues_from_paginated_response(cls, v: list[dict] | dict) -> list[dict] | dict:
        """Extract issues list from paginated response if needed."""
        if isinstance(v, dict):
            # If it's a paginated response, extract the results
            if "results" in v:
                return v["results"]
            # If it's a dict but not paginated (URL reference), return as-is
            return v
        return v


class JobSummary(BaseModel):
    """Summary view of a job (for list operations)."""

    model_config = ConfigDict(extra="ignore")

    id: int
    task_id: int
    project_id: int | None = None
    status: JobStatus
    stage: JobStage
    assignee: TaskOwner | None = None
    start_frame: int
    stop_frame: int
    updated_date: datetime | None = None
