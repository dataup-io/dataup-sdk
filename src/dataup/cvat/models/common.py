"""Common types for the CVAT client."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """CVAT paginated response format."""

    count: int
    next: str | None = None
    previous: str | None = None
    results: list[T]

    def has_next(self) -> bool:
        """Check if there's a next page."""
        return self.next is not None


class FrameData(BaseModel):
    """Binary frame/image data wrapper."""

    frame_id: int
    content: bytes
    content_type: str = "image/jpeg"
    filename: str | None = None

    class Config:
        arbitrary_types_allowed = True
