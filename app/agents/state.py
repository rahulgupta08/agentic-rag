from typing import Dict, List, Optional, Any, Annotated
from pydantic import BaseModel, Field
import operator

class AgentState(BaseModel):

    # Session
    # loading memory from SQLite
    # storing conversation history
    session_id: str

    # User question
    query: str

    # Query classification, either a RAG query, web search or hybrid query
    query_type: Optional[str] = None

    # Planner execution
    plan: List[Any] = Field(default_factory=list)

    # Generated answer
    answer: Optional[str] = None

    validation_score: Annotated[Optional[float], lambda a, b: b] = None
    needs_retry: Annotated[Optional[bool], lambda a, b: b] = None

    conversation_history: Optional[List[Any]] = None

    tool_results: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)

    next: Optional[str] = None
