from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from app.core.rate_limiter import get_rate_limiter
import logging

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limiter=None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or get_rate_limiter()
    
    async def dispatch(self, request: Request, call_next):
        # Generate a key for rate limiting (using IP address for now)
        # In a production system, you might want to use user ID or API key
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:{client_ip}"
        
        if not self.rate_limiter.is_allowed(key):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests, rate limit exceeded"
            )
        
        response = await call_next(request)
        return response