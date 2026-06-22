import time
from typing import Callable, Any

def execute_with_backoff(
    action_fn: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 0.5
) -> Any:
    """Executes a Jira API operation with exponential backoff on HTTP 429 rate limit exceptions."""
    delay = initial_delay
    last_exc = None
    for attempt in range(max_retries):
        try:
            return action_fn()
        except Exception as e:
            last_exc = e
            if "429" in str(e) or "Rate limit" in str(e):
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    raise last_exc
