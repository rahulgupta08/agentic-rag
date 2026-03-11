from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.exceptions import RAGException
import logging

logger = logging.getLogger(__name__)


async def rag_exception_handler(request: Request, exc: RAGException):
    logger.error(
        f"RAGException | path={request.url.path} | "
        f"error_code={exc.error_code} | message={exc.message}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "path": request.url.path,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(
        f"Unhandled Exception | path={request.url.path} | message={str(exc)}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "Unexpected error occurred.",
            "path": request.url.path,
        },
    )