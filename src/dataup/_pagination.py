"""Pagination utilities for the DataUp SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable, Iterator, TypeVar

if TYPE_CHECKING:
    from dataup.models.common import CursorPage

T = TypeVar("T")


def paginate(
    fetch_page: Callable[..., CursorPage[T]],
    **kwargs: object,
) -> Iterator[T]:
    """
    Iterate through all pages of a paginated endpoint.

    Args:
        fetch_page: A function that fetches a page of results.
        **kwargs: Additional arguments to pass to fetch_page.

    Yields:
        Items from all pages.

    Example:
        ```python
        for agent in paginate(client.agents.list, provider="ultralytics"):
            print(agent.name)
        ```
    """
    cursor = None
    while True:
        page = fetch_page(cursor=cursor, **kwargs)
        yield from page.items
        if not page.has_next():
            break
        # Use cursor if available, otherwise use next_page
        cursor = page.cursor or page.next_page


async def paginate_async(
    fetch_page: Callable[..., Awaitable[CursorPage[T]]],
    **kwargs: object,
) -> AsyncIterator[T]:
    """
    Async iterate through all pages of a paginated endpoint.

    Args:
        fetch_page: An async function that fetches a page of results.
        **kwargs: Additional arguments to pass to fetch_page.

    Yields:
        Items from all pages.

    Example:
        ```python
        async for agent in paginate_async(client.agents.list, provider="ultralytics"):
            print(agent.name)
        ```
    """
    cursor = None
    while True:
        page = await fetch_page(cursor=cursor, **kwargs)
        for item in page.items:
            yield item
        if not page.has_next():
            break
        # Use cursor if available, otherwise use next_page
        cursor = page.cursor or page.next_page
