"""
Retry handler with exponential backoff for failed operations.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class RetryHandler:
    """Handles retries with exponential backoff for failed operations."""

    def __init__(
        self, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_count = 0

    async def execute_with_retry(
        self, operation: Callable[..., Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """Execute an operation with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Operation succeeded after {attempt} retries")
                return result

            except Exception as e:
                last_exception = e
                self.retry_count += 1

                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Operation failed after {self.max_retries + 1} attempts. Last error: {e}"
                    )
                    raise last_exception

    def reset_count(self):
        """Reset the retry counter."""
        self.retry_count = 0

    @property
    def total_retries(self) -> int:
        """Get total number of retries performed."""
        return self.retry_count
