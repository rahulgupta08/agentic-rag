import time
from typing import Dict, Set
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.limits: Dict[str, int] = defaultdict(lambda: max_requests)
        
    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed for the given key
        
        Args:
            key: Unique identifier for the request (e.g., IP address, user ID)
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        request_times = self.requests[key]
        
        # Remove requests outside the time window
        while request_times and request_times[0] <= now - self.window_seconds:
            request_times.popleft()
        
        # Check if we're under the limit
        if len(request_times) < self.max_requests:
            request_times.append(now)
            return True
        
        return False

# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance"""
    return rate_limiter