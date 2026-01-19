"""Frame/image models for the CVAT client."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FrameMeta(BaseModel):
    """Metadata for a single frame."""

    width: int
    height: int
    name: str  # Filename
    related_files: int | list[str] = 0
    has_related_context: bool = False


# example = {
#     "chapters": None,
#     "chunks_updated_date": "2025-12-24T18:49:39.369314Z",
#     "chunk_size": 19,
#     "size": 1,
#     "image_quality": 70,
#     "start_frame": 75,
#     "stop_frame": 75,
#     "frame_filter": "",
#     "frames": [
#         {
#             "width": 1700,
#             "height": 2200,
#             "name": "2511.16719v1000000000.jpeg-76.jpg",
#             "related_files": 0,
#             "has_related_context": False,
#         }
#     ],
#     "deleted_frames": [],
#     "included_frames": None,
#     "storage": "local",
#     "cloud_storage_id": None,
# }


class DataMetaInfo(BaseModel):
    """Complete data metadata for a task or job."""

    model_config = ConfigDict(extra="ignore")

    chunk_size: int | None = None
    size: int = 0  # Total frames
    image_quality: int = 70
    start_frame: int = 0
    stop_frame: int = 0
    frame_filter: str = ""
    frames: list[FrameMeta] = Field(default_factory=list)
    deleted_frames: list[int] = Field(default_factory=list)
    chapters: dict | None = None
    chunks_updated_date: str | None = None
    included_frames: list[int] | None = None
    storage: str = "local"
    cloud_storage_id: int | None = None


class FrameImage(BaseModel):
    """Frame image data with metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame_id: int
    width: int | None = None
    height: int | None = None
    name: str | None = None
    image_url: str | None = None
    data: bytes | None = None
    content_type: str = "image/jpeg"
