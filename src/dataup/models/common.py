"""Common types for the DataUp SDK."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class CursorPage(BaseModel, Generic[T]):
    """Cursor-based pagination response."""

    items: list[T]
    total: int | None = None
    size: int | None = None
    cursor: str | None = None
    next_page: str | None = None

    def has_next(self) -> bool:
        """Check if there's a next page."""
        return self.cursor is not None or self.next_page is not None
