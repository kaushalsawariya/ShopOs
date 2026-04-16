"""
core/token_manager.py
---------------------
Token management and optimization utilities for OpenAI API calls.
Handles token counting, truncation, and rate limiting.
"""

import os
import tiktoken
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class TokenManager:
    """Manages token usage and optimization for OpenAI API calls."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.encoding = self._load_encoding(model)
        self.max_tokens = {
            "gpt-4o": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385
        }.get(model, 128000)
        self.rate_limiter = RateLimiter()  # Add rate limiter instance

    def _load_encoding(self, model: str):
        """Load tokenizer assets without making the app depend on network access."""
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        text = "" if text is None else str(text)
        if self.encoding is None:
            return max(1, len(text) // 4) if text else 0
        return len(self.encoding.encode(text))

    def count_messages(self, messages: List[BaseMessage]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            total += self.count_tokens(str(msg.content))
            # Add tokens for message formatting
            total += 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        total += 3  # every reply is primed with <|start|>assistant<|message|>
        return total

    def truncate_text(self, text: str, max_tokens: int, preserve_start: bool = True) -> str:
        """Truncate text to fit within token limit."""
        text = "" if text is None else str(text)
        if self.encoding is None:
            approx_chars = max_tokens * 4
            if len(text) <= approx_chars:
                return text
            if preserve_start:
                return text[: max(0, approx_chars - 3)] + "..."
            return "..." + text[-max(0, approx_chars - 3):]

        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        if preserve_start:
            # Keep the beginning, truncate the end
            truncated_tokens = tokens[:max_tokens-3]  # Leave room for "..."
            truncated_text = self.encoding.decode(truncated_tokens) + "..."
        else:
            # Keep the end, truncate the beginning
            truncated_tokens = tokens[-(max_tokens-3):]
            truncated_text = "..." + self.encoding.decode(truncated_tokens)

        return truncated_text

    def optimize_context(self, context: str, max_tokens: int = 2000) -> str:
        """Optimize context by removing redundant information and truncating."""
        if self.count_tokens(context) <= max_tokens:
            return context

        # Simple optimization: remove excessive whitespace and duplicate content
        import re
        context = re.sub(r'\n\s*\n\s*\n+', '\n\n', context)  # Multiple newlines to double
        context = re.sub(r'\s+', ' ', context)  # Multiple spaces to single

        return self.truncate_text(context, max_tokens)

    def estimate_cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        """Estimate cost in USD for a request."""
        # GPT-4o pricing (as of 2024)
        input_cost_per_1k = 0.005
        output_cost_per_1k = 0.015

        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k

        return input_cost + output_cost


class RateLimiter:
    """Simple rate limiter for OpenAI API calls."""

    def __init__(self, requests_per_minute: int = 50, tokens_per_minute: int = 30000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        # Be more conservative - leave 5% buffer
        self.effective_token_limit = int(tokens_per_minute * 0.95)
        self.request_times = []
        self.token_usage = []

    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """Check if we can make a request without hitting rate limits."""
        import time

        current_time = time.time()
        one_minute_ago = current_time - 60

        # Clean old entries
        self.request_times = [t for t in self.request_times if t > one_minute_ago]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > one_minute_ago]

        # Check limits
        recent_tokens = sum(tokens for _, tokens in self.token_usage)

        # Allow the request if we're under the effective limit
        return (len(self.request_times) < self.requests_per_minute and
                recent_tokens + estimated_tokens <= self.effective_token_limit)

    def get_wait_time(self, estimated_tokens: int = 1000) -> float:
        """Get how long to wait before making a request (in seconds)."""
        import time

        current_time = time.time()
        one_minute_ago = current_time - 60

        # Clean old entries
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > one_minute_ago]

        recent_tokens = sum(tokens for _, tokens in self.token_usage)

        if recent_tokens + estimated_tokens <= self.effective_token_limit:
            return 0.0  # Can make request now

        # Calculate how long to wait for old tokens to expire
        if not self.token_usage:
            return 0.0  # No recent usage

        # Find the oldest token usage that would allow this request
        tokens_needed = estimated_tokens
        sorted_usage = sorted(self.token_usage, key=lambda x: x[0])  # Sort by time

        for timestamp, tokens in sorted_usage:
            time_passed = current_time - timestamp
            if time_passed >= 60:
                continue  # This entry is already expired

            tokens_needed -= tokens
            if tokens_needed <= 0:
                # We need to wait until this timestamp + 60 seconds
                return max(0, (timestamp + 60) - current_time)

        # If we get here, we need to wait for all current tokens to expire
        if self.token_usage:
            oldest_time = min(t for t, _ in self.token_usage)
            return max(0, (oldest_time + 60) - current_time)

        return 0.0

    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if necessary to make a request."""
        wait_time = self.get_wait_time(estimated_tokens)
        if wait_time > 0:
            print(f"Rate limit approached. Waiting {wait_time:.2f} seconds...")
            import time
            time.sleep(wait_time)

    def record_request(self, tokens_used: int):
        """Record a completed request."""
        import time
        current_time = time.time()
        self.request_times.append(current_time)
        self.token_usage.append((current_time, tokens_used))


# Global instances
token_manager = TokenManager()
rate_limiter = RateLimiter()
