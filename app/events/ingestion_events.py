from pydantic import Field

from app.events.base_event import BaseEvent


class DocumentSubmittedEvent(BaseEvent):
    """
    Initial ingestion request event.

    Produced when a document ingestion request
    enters the ingestion system.
    """

    event_type: str = "document.submitted"

    payload: dict = Field(
        default_factory=dict,
        description="""
        Expected payload:
        {
            "document_id": str,
            "file_path": str,
            "collection": str
        }
        """
    )


class DocumentLoadedEvent(BaseEvent):
    """
    Produced after PDF extraction and normalization.
    """

    event_type: str = "document.loaded"


class DocumentChunkedEvent(BaseEvent):
    """
    Produced after semantic chunk generation.
    """

    event_type: str = "document.chunked"


class DocumentEmbeddedEvent(BaseEvent):
    """
    Produced after embedding generation.
    """

    event_type: str = "document.embedded"


class DocumentPersistedEvent(BaseEvent):
    """
    Produced after vector persistence completes.
    """

    event_type: str = "document.persisted"


class DocumentFailedEvent(BaseEvent):
    """
    Produced whenever a stage fails irrecoverably.
    """

    event_type: str = "document.failed"