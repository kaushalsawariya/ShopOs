"""
core/retry_handler.py
---------------------
Retry logic and error handling for API calls.
"""

import time
import random
from typing import Callable, Any, Optional
from functools import wraps


class RetryHandler:
    """Handles retries with exponential backoff for API calls."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        error_msg = str(error).lower()
        return any(keyword in error_msg for keyword in [
            "rate_limit", "429", "timeout", "connection", "server_error", "502", "503", "504"
        ])

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if this is specifically a rate limit error."""
        error_msg = str(error).lower()
        return "rate_limit" in error_msg or "429" in error_msg

    def _extract_wait_time(self, error: Exception) -> float:
        """Extract wait time from rate limit error message."""
        error_msg = str(error).lower()
        if "please try again in" in error_msg:
            # Extract time from message like "Please try again in 364ms"
            import re
            match = re.search(r'please try again in (\d+)ms', error_msg)
            if match:
                return float(match.group(1)) / 1000.0  # Convert ms to seconds

            match = re.search(r'please try again in (\d+\.?\d*)s', error_msg)
            if match:
                return float(match.group(1))

        # Default wait time for rate limits
        return 1.0

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 1.0) * delay * 0.1
        return delay + jitter

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e

                if attempt == self.max_retries or not self._is_retryable_error(e):
                    break

                # Special handling for rate limit errors
                if self._is_rate_limit_error(e):
                    wait_time = self._extract_wait_time(e)
                    print(f"Rate limit hit. Waiting {wait_time:.2f} seconds before retry...")
                else:
                    wait_time = self._calculate_delay(attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f} seconds...")

                import time
                time.sleep(wait_time)

        # If we get here, all retries failed
        raise last_error


def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for functions that need retry logic."""
    def decorator(func: Callable) -> Callable:
        retry_handler = RetryHandler(max_retries, base_delay)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return retry_handler.execute_with_retry(func, *args, **kwargs)

        return wrapper
    return decorator


# Global retry handler
retry_handler = RetryHandler()